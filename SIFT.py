import numpy as np
import cv2 as cv
from scipy.ndimage import maximum_filter, minimum_filter
from matplotlib import pyplot as plt

init_sigma = 0.5 # 假设初始图片自带的高斯模糊标准差
dogpyr_layers = 3 # 期望生成的高斯差分金字塔的每个八度的层数
gpyr_layers = dogpyr_layers + 3 # 高斯金字塔每个八度的层数，为啥+3，举个例子，如果s=1，差分金字塔需要高斯金字塔相邻2层相减，所以高斯金字塔需要s+1层，又因为检测关键点需要在高斯差分金字塔相邻3层，所以还需要+2
sigma_0 = 1.6
contrast_threshold = 0.04
edge_threshold = 10
R = ((edge_threshold + 1) ** 2) / edge_threshold
max_interp_steps = 5
img_border = 5  # 图像边界（去除在这个边界范围的关键点）
ori_hist_bins = 36

descr_scale_factor = 3  # 描述子的尺度因子，乘σ得到直方图区域宽度
descr_hist_bins = 8
descr_width = 4  # 特征点方向直方图的bin数
descr_mag_threshold = 0.2  # 描述子幅值阈值，限制单一方向bin的过大贡献
descr_norm_factor = 512



def build_pyramid(image,n_octaves):
    """
    当图像下采样到一半大小时（宽度和高度各减半），像素的空间间距变为原来的2倍
    在高斯模糊中，标准差sigma的单位是像素。因此，如果对下采样后的图像应用相同的sigma，其实际尺度（相对于原始图像）会因为像素间距的变化而放大
    每下一个八度，图像的像素间距翻倍，sigma也翻倍
    """
    gaussian_pyramid = []
    dog_pyramid = []
    k = 2 ** (1 / dogpyr_layers)
    img = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)  # 原图上采样作为0层
    # 假设原图自带init_sigma的高斯模糊
    sigma_diff = np.sqrt(sigma_0 ** 2 - (2 * init_sigma) ** 2)
    img = cv.GaussianBlur(img, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

    for octave in range(n_octaves):
        octave_pyramid = []
        octave_dog_pyramid = []
        if octave == 0:
            octave_pyramid.append(img)
        else:
            octave_pyramid.append(
                cv.resize(gaussian_pyramid[-1][-3], None, fx=0.5, fy=0.5,
                          interpolation=cv.INTER_LINEAR))  # 上一层的倒数第三层作为下一层第一张图

        for i in range(1, gpyr_layers):
            # σ计算公式：σ=(σi**2-(σ(i-1))**2)**0.5  σi=(k^i)σ
            sigma_to_apply = sigma_0 * k ** (i - 1) * (k ** 2 - 1) ** 0.5
            img = cv.GaussianBlur(octave_pyramid[-1], (0, 0), sigmaX=sigma_to_apply, sigmaY=sigma_to_apply)
            octave_pyramid.append(img)
            octave_dog_pyramid.append((octave_pyramid[-1] - octave_pyramid[-2]) / 255.0)  # 高斯金字塔每相邻两层相减得到DoG金字塔并归一化
        gaussian_pyramid.append(octave_pyramid)
        dog_pyramid.append(octave_dog_pyramid)
    return gaussian_pyramid, dog_pyramid


def find_keypoints(dog_pyramid):
    keypoints = []
    for o, octave in enumerate(dog_pyramid):
        edge_mask = np.ones_like(octave[0], dtype=bool)  # 过滤在边缘点
        edge_mask[:img_border, :] = False
        edge_mask[-img_border:, :] = False
        edge_mask[:, :img_border] = False
        edge_mask[:, -img_border:] = False
        for layer in range(1, len(octave) - 1):  # 跳过第一层和最后一层
            prev = octave[layer - 1]
            curr = octave[layer]
            next = octave[layer + 1]
            patch = np.stack([prev, curr, next], axis=0)
            height, width = curr.shape

            # 找到3*3*3的局部最大值和最小值
            local_max = maximum_filter(patch, size=(3, 3, 3))
            local_min = minimum_filter(patch, size=(3, 3, 3))

            extrema_mask = (curr == local_max[1]) | (curr == local_min[1])  # 是否是邻域最大值或最小值
            final_mask = extrema_mask & edge_mask
            ys, xs = np.where(final_mask)

            for x, y in zip(xs, ys):
                # 亚像素插值
                abandon, x, y, l, bx, by, bl = subpixel_interpolation(octave, layer, x, y, height, width, max_interp_steps, img_border)
                if abandon:
                    continue

                # 对比度剔除
                dx = (curr[y, x + 1] - curr[y, x - 1]) / 2
                dy = (curr[y + 1, x] - curr[y - 1, x]) / 2
                ds = (next[y, x] - prev[y, x]) / 2
                t = dx * bx + dy * by + ds * bl  # 前面收敛的条件是小于0.5，所以四舍五入为0全都没加到x,y,l上，这里要加上这部分偏移
                contr = curr[y, x] + 0.5 * t  # D(X) ≈ D + ∇DTX
                if contr * dogpyr_layers < contrast_threshold:
                    continue

                # 边缘响应剔除
                dxx = curr[y, x + 1] + curr[y, x - 1] - 2 * curr[y, x]
                dyy = curr[y + 1, x] + curr[y - 1, x] - 2 * curr[y, x]
                dxy = (curr[y + 1, x + 1] - curr[y + 1, x - 1] - curr[y - 1, x + 1] + curr[y - 1, x - 1]) / 4.0
                tr = dxx + dyy
                det = dxx * dyy - dxy ** 2
                if det <= 0 or (tr ** 2) / det >= R:
                    continue

                keypoints.append({
                    "octave": o,
                    "layer": l,
                    "x": int(x),
                    "y": int(y),
                    # 关键点的尺度必须反映在原始图像的尺度空间，因为 SIFT 的特征描述子是基于原始图像的坐标和尺度计算的，所以需要2^o
                    "sigma": sigma_0 * (2 ** o) * (2 ** (l / dogpyr_layers))
                })
    return keypoints

def subpixel_interpolation(octave, l, x, y, height, width, max_interp_steps, img_border):
    abandon = False
    bx, by, bl = 0, 0, 0
    for i in range(max_interp_steps + 1):
        """
        在点X0=(x,y,σ)附近泰勒展开 D(X) ≈ D + ∇DTX + XTHX/2 （这里的X实际上是相对于当前点X0的偏移量）
        对X求导，令 dD(X)/dX = ∇D + HX = 0 ⇒ X = -H^{-1}∇D
        """
        prev = octave[l - 1]
        curr = octave[l]
        next = octave[l + 1]
        dx = (curr[y, x + 1] - curr[y, x - 1]) / 2
        dy = (curr[y + 1, x] - curr[y - 1, x]) / 2
        ds = (next[y, x] - prev[y, x]) / 2
        dxx = curr[y, x + 1] + curr[y, x - 1] - 2 * curr[y, x]
        dyy = curr[y + 1, x] + curr[y - 1, x] - 2 * curr[y, x]
        dss = next[y, x] + prev[y, x] - 2 * curr[y, x]
        dxy = (curr[y + 1, x + 1] - curr[y + 1, x - 1] - curr[y - 1, x + 1] + curr[y - 1, x - 1]) / 4.0
        dxs = (next[y, x + 1] - next[y, x - 1] - prev[y, x + 1] + prev[y, x - 1]) / 4.0
        dys = (next[y + 1, x] - next[y - 1, x] - prev[y + 1, x] + prev[y - 1, x]) / 4.0
        H = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
        dD = np.array([dx, dy, ds])
        if np.abs(np.linalg.det(H)) < 1e-10:
            abandon = True
            break
        else:
            bx, by, bl = -np.linalg.solve(H, dD)
        if np.abs(bx) <= 0.5 and np.abs(by) <= 0.5 and np.abs(bl) <= 0.5:  # 收敛条件（注意这里已经小于0.5，所以四舍五入会变为0，不需要加）
            break
        if abs(bx) > 1e6 or abs(by) > 1e6 or abs(bl) > 1e6:  # 防止溢出
            abandon = True
            break
        x += round(bx)
        y += round(by)
        l += round(bl)
        if (l < 1 or l >= len(octave) - 1) or (
                x < img_border or x >= width - img_border) or (
                y < img_border or y >= height - img_border):
            abandon = True
            break
        if i == max_interp_steps:  # 达到最大插值次数还没收敛，放弃该点
            abandon = True
            break

    return abandon, x, y, l, bx, by, bl

# 方向分配
def orientation_assignment(keypoints, gaussian_pyramid):
    final_keypoints = []
    for kp in keypoints:
        o = kp["octave"]
        l = kp["layer"]
        x, y = kp["x"], kp["y"]
        sigma = kp["sigma"]
        img = gaussian_pyramid[o][l]

        radius = round(3 * 1.5 * sigma)  # 选取4.5σ作为关键点邻域的半径（正方形），3与3σ原则有关，1.5是经验值

        # 为每个关键点的邻域构建一个角度直方图，每10°一个bin，共36个bin
        n = ori_hist_bins
        hist = np.zeros(n)
        angle_per_bin = 360 / n
        kernel = generate_gaussian_kernel(2 * radius + 1, sigma)
        height, width = img.shape

        for r in range(max(y - radius, 1), min(y + radius + 1, height - 1)):
            for c in range(max(x - radius, 1), min(x + radius + 1, width - 1)):
                # 计算每个像素点x和y方向的梯度
                dx = img[r, c + 1] - img[r, c - 1]
                dy = img[r + 1, c] - img[r - 1, c]
                mag = np.sqrt(dx ** 2 + dy ** 2)  # 梯度幅值
                angle = np.degrees(np.arctan2(dy, dx)) % 360  # 梯度方向，以向右为起点，顺时针方向为正，范围[0, 360)
                bin_center = angle / angle_per_bin
                bin_low = int(bin_center)
                bin_high = (bin_low + 1) % n
                weight = kernel[r - y + radius, c - x + radius]  # 高斯权重，距离中心点越远权重越小
                # 加权，线性插值
                hist[bin_low] += mag * weight * (bin_high - bin_center)
                hist[bin_high] += mag * weight * (bin_center - bin_low)

        # 对直方图进行[1,4,6,4,1]/16核高斯平滑，提升鲁棒性
        smoothed_hist = np.zeros(n)
        for i in range(n):
            smoothed_hist[i] = (hist[(i - 2) % n] * 1 +
                                hist[(i - 1) % n] * 4 +
                                hist[i] * 6 +
                                hist[(i + 1) % n] * 4 +
                                hist[(i + 2) % n] * 1) / 16
        hist = smoothed_hist

        peak = np.max(hist)  # 选出峰值作为关键点的主方向
        threshold = 0.8 * peak  # 阈值，如果还有别的方向大于这个阈值，也会被认为是关键点的主方向，要把关键点复制一份，把角度改成这个方向
        for i in range(n):
            pre = hist[(i - 1) % n]
            cur = hist[i]
            nxt = hist[(i + 1) % n]
            if cur >= threshold and cur >= pre and cur >= nxt:
                delta = (pre - nxt) / (2 * (pre + nxt - 2 * cur))  # 计算峰值的位置，二次插值
                angle = (i + delta) * angle_per_bin  # bin 对应的角度
                new_kp = kp.copy()
                new_kp["angle"] = angle
                final_keypoints.append({
                    "octave": o,
                    "layer": l,
                    "x": x,
                    "y": y,
                    "sigma": sigma,
                    "angle": angle
                })
    return final_keypoints


def generate_descriptor(keypoints, gaussian_pyramid):
    descriptors = []
    n = descr_hist_bins
    d = descr_width
    for kp in keypoints:
        o = kp["octave"]
        l = kp["layer"]
        x, y = kp["x"], kp["y"]
        sigma = kp["sigma"]
        angle = kp["angle"]
        img = gaussian_pyramid[o][l]

        height, width = img.shape
        hist_width = descr_scale_factor * sigma / 2 ** o
        cos_t = np.cos(np.deg2rad(-angle))
        sin_t = np.sin(np.deg2rad(-angle))

        # 除以2是变为半径，乘√2是为了变为对角线，是理想情况下的最大采样半径，还有d+1，这样扩大范围是因为后面要进行的插值采样可能会用到边缘以外的数据
        radius = round(hist_width * np.sqrt(2) * (d + 1) / 2)
        radius = int(min(radius, np.sqrt(height ** 2 + width ** 2)))  # 防止半径过大

        pix_num = (radius * 2 + 1) ** 2  # +1是为了包含中心像素
        X = np.zeros(pix_num)
        Y = np.zeros(pix_num)
        xbin = np.zeros(pix_num)
        ybin = np.zeros(pix_num)
        w = np.zeros(pix_num)
        idx = 0
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                x_rot = (cos_t * j - sin_t * i) / hist_width  # x' = xcosθ - ysinθ  /hist_width是为了归一化为bin坐标
                y_rot = (sin_t * j + cos_t * i) / hist_width  # y' = xsinθ + ycosθ

                """
                这里的bin指的是d*d的网格里的一个格子，不是之前直方图里的bin，x_bin和y_bin是在d*d区域里格子的坐标
                x_rot能直接和d/2相加是因为前面cos_t和sin_t已经除以了hist_width
                x_rot和y_rot是偏移量，+d/2是把坐标原点移到左上角
                举个例子，若d为4，x_rot和y_rot都是-2，如果d为4，那么x_rot + d/2和y_rot + d/2都变为0，这是最左上角的格子
                此时每个格子的中心是0.5,1.5,2.5,3.5，通过减去0.5，对齐到整数位置0,1,2,3,方便计算
                """
                x_bin = x_rot + d / 2 - 0.5
                y_bin = y_rot + d / 2 - 0.5
                x_idx = x + j
                y_idx = y + i
                """
                前面-0.5，rbin和cbin的范围从[0, d]调整为[-0.5, d-0.5]，前后各扩张0.5用以插值变为[-1, d]，这就是前面radius中的(d + 1)的由来
                这时候可能就有人要问了，[-1, d]这不是有d+2个格子吗？错了，这个-1和d是边界，在[-1, d]之中包含了d-(-1)=d+1个格子
                """
                if (-1 < y_bin < d) and (-1 < x_bin < d) and (0 < x_idx < width - 1) and (0 < y_idx < height - 1):
                    dx = img[y_idx][x_idx + 1] - img[y_idx][x_idx - 1]
                    dy = img[y_idx + 1][x_idx] - img[y_idx - 1][x_idx]
                    X[idx] = dx
                    Y[idx] = dy
                    xbin[idx] = x_bin
                    ybin[idx] = y_bin
                    w[idx] = -(x_rot ** 2 + y_rot ** 2) / (d * d * 0.5)  # 高斯权重，标准差σ=d/2，这样邻域的高斯窗口刚好覆盖整个d*d区域
                    idx += 1
        ori = np.degrees(np.arctan2(Y, X))
        obin = ((ori - angle) % 360) / 360 * n
        mag = np.sqrt(X ** 2 + Y ** 2) * np.exp(w)
        xbin = xbin[:idx]  # 网格坐标（网格bin）
        ybin = ybin[:idx]
        obin = obin[:idx]  # 方向直方图的bin坐标（直方图bin）
        mag = mag[:idx]

        raw_descr = np.zeros(d * d * n)

        # 三线性插值
        for k in range(idx):
            y0 = int(np.floor(ybin[k]))
            x0 = int(np.floor(xbin[k]))
            o0 = int(np.floor(obin[k]))

            dy = ybin[k] - y0
            dx = xbin[k] - x0
            do = obin[k] - o0

            w_y = (1 - dy, dy)
            w_x = (1 - dx, dx)
            w_o = (1 - do, do)

            # 前面向下取整，[0,1]中0代表影响“左”的格子，1代表影响“右”的格子
            for yi in [0, 1]:
                for xi in [0, 1]:
                    for oi in [0, 1]:
                        y_idx = y0 + yi
                        x_idx = x0 + xi
                        o_idx = (o0 + oi) % n

                        if y_idx < 0 or y_idx >= d or x_idx < 0 or x_idx >= d:
                            continue

                        weight = w_y[yi] * w_x[xi] * w_o[oi]

                        # 方向维度n是最内层维度,在行主序中，方向索引o_idx是连续存储的，步长为1，列索引x_idx的步长为n，行索引y_idx的步长为d*n
                        raw_descr[(y_idx * d + x_idx) * n + o_idx] += mag[k] * weight  # 插值

        # 限制大值，防止单一方向bin的过大贡献，提高描述子对光照变化和噪声的鲁棒性（大值通常由强边缘或噪声引起）
        threshold = np.linalg.norm(raw_descr) * descr_mag_threshold
        raw_descr = np.clip(raw_descr, 0, threshold)

        scale = descr_norm_factor / max(np.linalg.norm(raw_descr), 1e-7)

        descriptors.append(raw_descr * scale)

    return descriptors


def detect_and_compute(image, n_octaves=-1):
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = np.array(image, dtype=np.float32)
    if n_octaves == -1:
        min_dim = min(image.shape)
        n_octaves = round(np.log2(min_dim)) - 1
    gaussian_pyramid, dog_pyramid = build_pyramid(image, n_octaves)
    keypoints = find_keypoints(dog_pyramid)
    keypoints = orientation_assignment(keypoints, gaussian_pyramid)
    descriptors = generate_descriptor(keypoints, gaussian_pyramid)

    return keypoints, descriptors


def generate_gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)




def knn_match(descr1, descr2, threshold=0.6):
    matches = []
    descr1 = np.array(descr1)
    descr2 = np.array(descr2)
    desc1_sq = np.sum(descr1 ** 2, axis=1, keepdims=True) # 形状为(N1, 1)，加keepdims=True是为了保持二维形状，方便后面广播
    desc2_sq = np.sum(descr2 ** 2, axis=1)
    inner = np.dot(descr1, descr2.T)
    # d(u,v)= √(∥u∥²+∥v∥²−2u⋅v)
    dists_squared = desc1_sq - 2 * inner + desc2_sq
    dists_squared = np.maximum(dists_squared, 0) # 由于浮点精度问题可能出现负数
    dists = np.sqrt(dists_squared) # 形状为(N1, N2)，每个元素(i,j)代表descr1[i]和descr2[j]的欧式距离

    for i in range(dists.shape[0]):
        sorted_idx = np.argsort(dists[i])  # 把第i行的距离从小到大排序，返回索引
        nearest = sorted_idx[:2]  # 取前k个最近的索引
        d1, d2 = dists[i, nearest[0]], dists[i, nearest[1]]

        # 一个好的匹配点，它和第一个最近邻的距离应该远小于第二个的，否则就不够独特、不可靠
        if d1 < threshold * d2:
            matches.append((i, nearest[0]))

    return matches


def show_pyramid(pyramid):
    for octave_idx, octave in enumerate(pyramid):
        for layer_idx, layer_img in enumerate(octave):
            normalized_img = cv.normalize(layer_img, None, 0, 255, cv.NORM_MINMAX)
            normalized_img = normalized_img.astype(np.uint8)

            window_name = f"Octave {octave_idx}, Layer {layer_idx}"
            cv.imshow(window_name, normalized_img)
            cv.waitKey(0)
            cv.destroyAllWindows()


def draw_keypoints(img, keypoints):
    image = img.copy()
    for kp in keypoints:
        # 坐标转换到原图
        scale = 2 / (2 ** kp["octave"])
        x = round(kp["x"] / scale)
        y = round(kp["y"] / scale)
        sigma = kp["sigma"]
        radius = round(sigma * 0.5)
        angle = kp["angle"]

        # 方向 → 颜色（HSV → BGR）
        hue = angle / 2  # OpenCV的Hue范围是0~180
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2BGR)[0, 0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        # 计算方向线段终点
        angle_rad = np.deg2rad(angle)
        end_x = round(x + radius * np.cos(angle_rad))
        end_y = round(y + radius * np.sin(angle_rad))

        # 绘制圆圈和方向线（抗锯齿）
        cv.circle(image, (x, y), radius, color, 1, cv.LINE_AA)
        cv.line(image, (x, y), (end_x, end_y), color, 1, cv.LINE_AA)
    return image

def draw_matches(img1, img2, kp1, kp2, matches, figsize=(15, 8), show_lines=True):
    # 确保两个图是彩色的并转换为RGB
    if img1.ndim == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    else:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)  # 转换BGR到RGB

    if img2.ndim == 2:
        img2 = np.stack([img2] * 3, axis=-1)
    else:
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)  # 转换BGR到RGB

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 拼接图像
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:] = img2

    # 绘图
    plt.figure(figsize=figsize)
    plt.imshow(out_img)
    plt.axis('off')

    for i1, i2 in matches:
        x1, y1 = kp1[i1]["x"], kp1[i1]["y"]
        scale1 = 2 / (2 ** kp1[i1]["octave"])
        x2, y2 = kp2[i2]["x"], kp2[i2]["y"]
        scale2 = 2 / (2 ** kp2[i2]["octave"])
        x1 = round(x1 / scale1)
        y1 = round(y1 / scale1)
        x2 = round(x2 / scale2) + w1
        y2 = round(y2 / scale2)

        color = np.random.rand(3,)
        plt.scatter([x1, x2], [y1, y2], c=[color], s=10)
        if show_lines:
            plt.plot([x1, x2], [y1, y2], c=color, linewidth=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    image1 = cv.imread('box.png')
    image2 = cv.imread('box_in_scene.png')

    keypoints1, descriptors1 = detect_and_compute(image1)
    keypoints2, descriptors2 = detect_and_compute(image2)
    matches = knn_match(descriptors1, descriptors2)
    img1 = draw_keypoints(image1, keypoints1)
    img2 = draw_keypoints(image2, keypoints2)
    print("匹配数量:", len(matches))
    draw_matches(image1, image2, keypoints1, keypoints2, matches)

    cv.imshow("SIFT Keypoints1", img1)
    cv.imshow("SIFT Keypoints2", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

