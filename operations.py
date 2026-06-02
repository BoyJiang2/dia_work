from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np

ParamType = Literal["int", "float", "choice", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: ParamType
    default: int | float | str | bool
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float = 1
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class Operation:
    key: str
    name: str
    category: str
    description: str
    params: tuple[ParamSpec, ...]
    apply: Callable[[np.ndarray, dict], np.ndarray]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# ===================================================================
#  灰度变换与增强  (Grayscale Transform & Enhancement)
# ===================================================================

def negative(image: np.ndarray, params: dict) -> np.ndarray:
    return 255 - image


def linear_transform(image: np.ndarray, params: dict) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def log_transform(image: np.ndarray, params: dict) -> np.ndarray:
    scale = float(params["scale"])
    src = image.astype(np.float32)
    result = scale * np.log1p(src)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result.astype(np.uint8)


def gamma_transform(image: np.ndarray, params: dict) -> np.ndarray:
    gamma = max(0.01, float(params["gamma"]))
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def histogram_equalization(image: np.ndarray, params: dict) -> np.ndarray:
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_equalization(image: np.ndarray, params: dict) -> np.ndarray:
    clip_limit = float(params["clip_limit"])
    tile = _odd(int(params["tile_size"]))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gray_level_slicing(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    low = min(int(params["low"]), int(params["high"]))
    high = max(int(params["low"]), int(params["high"]))
    keep_background = bool(params["keep_background"])
    out = gray.copy() if keep_background else np.zeros_like(gray)
    out[(gray >= low) & (gray <= high)] = 255
    return out


def histogram_specification(image: np.ndarray, params: dict) -> np.ndarray:
    """直方图规定化：将原图直方图匹配到目标分布."""
    gray = to_gray(image)
    target = params["target"]
    h, w = gray.shape
    total = h * w

    # 计算原始 CDF
    hist_src = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cdf_src = np.cumsum(hist_src) / total

    # 构造目标直方图
    if target == "Uniform":
        hist_tgt = np.ones(256) / 256.0
    elif target == "Gaussian":
        x = np.arange(256)
        hist_tgt = np.exp(-((x - 127) ** 2) / (2 * 40 ** 2))
        hist_tgt /= hist_tgt.sum()
    elif target == "Exponential":
        x = np.arange(256)
        hist_tgt = np.exp(-x / 40)
        hist_tgt /= hist_tgt.sum()
    elif target == "DarkEmphasis":
        x = np.arange(256)
        hist_tgt = np.exp(-x / 20)
        hist_tgt /= hist_tgt.sum()
    elif target == "BrightEmphasis":
        x = np.arange(256)
        hist_tgt = np.exp(-(255 - x) / 20)
        hist_tgt /= hist_tgt.sum()
    else:
        return gray

    cdf_tgt = np.cumsum(hist_tgt)

    # 单映射规则 (SML): 对每个源灰度级找到最接近的目标灰度级
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(cdf_src[i] - cdf_tgt)
        mapping[i] = np.argmin(diff)

    result = mapping[gray]
    if image.ndim == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = mapping[ycrcb[:, :, 0]]
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return result


def bit_plane_decomposition(image: np.ndarray, params: dict) -> np.ndarray:
    """提取指定位平面."""
    gray = to_gray(image)
    plane = int(params["plane"])  # 0-7, 0=LSB, 7=MSB
    return ((gray >> plane) & 1) * 255


def arithmetic_add(image: np.ndarray, params: dict) -> np.ndarray:
    """图像加法去噪：生成N个加噪版本后取平均."""
    gray = to_gray(image)
    n = int(params["n_samples"])
    sigma = float(params["noise_sigma"])
    rng = np.random.default_rng()
    accumulator = np.zeros_like(gray, dtype=np.float64)
    for _ in range(n):
        noisy = gray.astype(np.float64) + rng.normal(0, sigma, gray.shape)
        accumulator += noisy
    avg = accumulator / n
    result = _clip(avg)
    if image.ndim == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def arithmetic_subtract(image: np.ndarray, params: dict) -> np.ndarray:
    """图像减法显示差异：用高斯模糊后的图像与原图相减突出变化区域."""
    gray = to_gray(image)
    k = _odd(int(params["kernel"]))
    blurred = cv2.GaussianBlur(gray, (k, k), float(params["sigma"]))
    diff = cv2.absdiff(gray.astype(np.int16), blurred.astype(np.int16))
    result = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


# ===================================================================
#  空间滤波  (Spatial Filtering)
# ===================================================================

def mean_filter(image: np.ndarray, params: dict) -> np.ndarray:
    k = _odd(int(params["kernel"]))
    return cv2.blur(image, (k, k))


def gaussian_filter(image: np.ndarray, params: dict) -> np.ndarray:
    k = _odd(int(params["kernel"]))
    sigma = float(params["sigma"])
    return cv2.GaussianBlur(image, (k, k), sigma)


def median_filter(image: np.ndarray, params: dict) -> np.ndarray:
    return cv2.medianBlur(image, _odd(int(params["kernel"])))


def bilateral_filter(image: np.ndarray, params: dict) -> np.ndarray:
    d = _odd(int(params["diameter"]))
    return cv2.bilateralFilter(image, d, float(params["sigma_color"]), float(params["sigma_space"]))


def laplacian_sharpen(image: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params["strength"])
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=_odd(int(params["kernel"])))
    return _clip(image.astype(np.float32) - strength * lap)


def unsharp_mask(image: np.ndarray, params: dict) -> np.ndarray:
    k = _odd(int(params["kernel"]))
    amount = float(params["amount"])
    blurred = cv2.GaussianBlur(image, (k, k), float(params["sigma"]))
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


# ===================================================================
#  边缘检测  (Edge Detection)
# ===================================================================

def sobel_edge(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    k = _odd(int(params["kernel"]))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def prewitt_edge(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def roberts_edge(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def laplacian_edge(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    edge = cv2.Laplacian(gray, cv2.CV_32F, ksize=_odd(int(params["kernel"])))
    return cv2.convertScaleAbs(edge)


def canny_edge(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    low = int(params["threshold1"])
    high = max(low + 1, int(params["threshold2"]))
    return cv2.Canny(gray, low, high)


def kirsch_edge(image: np.ndarray, params: dict) -> np.ndarray:
    """8方向 Kirsch 罗盘算子."""
    gray = to_gray(image).astype(np.float32)
    # Kirsch 8-direction kernels
    k0 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32)
    k1 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32)
    k2 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32)
    k3 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32)
    k4 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32)
    k5 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32)
    k6 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32)
    k7 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32)
    kernels = [k0, k1, k2, k3, k4, k5, k6, k7]

    responses = np.stack([cv2.filter2D(gray, cv2.CV_32F, k) for k in kernels], axis=-1)
    edge = responses.max(axis=-1)
    return cv2.normalize(edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def hough_lines(image: np.ndarray, params: dict) -> np.ndarray:
    """Hough 直线检测，在结果图上绘制检测到的直线."""
    gray = to_gray(image)
    edges = cv2.Canny(gray, 50, 150)
    rho = float(params["rho"])
    theta = np.deg2rad(float(params["theta"]))
    threshold = int(params["threshold"])
    min_length = float(params["min_length"])
    max_gap = float(params["max_gap"])

    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=min_length, maxLineGap=max_gap)
    result = to_bgr(gray).copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 80), 2)
    return result


def hough_circles(image: np.ndarray, params: dict) -> np.ndarray:
    """Hough 圆检测，在结果图上绘制检测到的圆."""
    gray = to_gray(image)
    dp = float(params["dp"])
    min_dist = int(params["min_dist"])
    param1 = int(params["param1"])
    param2 = int(params["param2"])
    min_r = int(params["min_radius"])
    max_r = int(params["max_radius"])

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_r, maxRadius=max_r)
    result = to_bgr(gray).copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            cx, cy, r = circle
            cv2.circle(result, (cx, cy), r, (0, 220, 220), 2)
            cv2.circle(result, (cx, cy), 3, (0, 140, 255), -1)
    return result


# ===================================================================
#  阈值与分割  (Thresholding & Segmentation)
# ===================================================================

def thresholding(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    mode = params["mode"]
    if mode == "Otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "Triangle":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        _, out = cv2.threshold(gray, int(params["threshold"]), 255, cv2.THRESH_BINARY)
    return out


def adaptive_threshold(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    block = max(3, _odd(int(params["block_size"])))
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if params["method"] == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, block, int(params["c"]))


# ===================================================================
#  形态学  (Morphology)
# ===================================================================

def morph_operation(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    k = _odd(int(params["kernel"]))
    shape_map = {"Rect": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
    kernel = cv2.getStructuringElement(shape_map[params["shape"]], (k, k))
    mapping = {
        "Erode": cv2.MORPH_ERODE,
        "Dilate": cv2.MORPH_DILATE,
        "Open": cv2.MORPH_OPEN,
        "Close": cv2.MORPH_CLOSE,
        "Gradient": cv2.MORPH_GRADIENT,
        "TopHat": cv2.MORPH_TOPHAT,
        "BlackHat": cv2.MORPH_BLACKHAT,
    }
    return cv2.morphologyEx(gray, mapping[params["operation"]], kernel, iterations=int(params["iterations"]))


def watershed_segment(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return to_bgr(image)
    _, sure_fg = cv2.threshold(dist, float(params["distance_ratio"]) * dist.max(), 255, 0)
    unknown = cv2.subtract(sure_bg, sure_fg.astype(np.uint8))
    _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
    markers = markers + 1
    markers[unknown == 255] = 0
    result = to_bgr(image).copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = (0, 0, 255)
    return result


# ===================================================================
#  频域处理  (Frequency Domain Processing)
# ===================================================================

def frequency_filter(image: np.ndarray, params: dict) -> np.ndarray:
    gray = to_gray(image).astype(np.float32)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = max(1, int(params["radius"]))
    mode = params["mode"]
    family = params["family"]
    yy, xx = np.ogrid[:rows, :cols]
    dist = np.sqrt((yy - crow) ** 2 + (xx - ccol) ** 2)

    if family == "Gaussian":
        low = np.exp(-(dist**2) / (2 * radius**2)).astype(np.float32)
    else:
        low = (dist <= radius).astype(np.float32)

    if mode == "LowPass":
        mask = low
    elif mode == "HighPass":
        mask = 1.0 - low
    else:
        width = max(1, int(params["band_width"]))
        inner = max(1, radius - width)
        outer = radius + width
        if family == "Gaussian":
            outer_low = np.exp(-(dist**2) / (2 * outer**2)).astype(np.float32)
            inner_low = np.exp(-(dist**2) / (2 * inner**2)).astype(np.float32)
            band = np.clip(outer_low - inner_low, 0, 1)
        else:
            band = ((dist >= inner) & (dist <= outer)).astype(np.float32)
        mask = band if mode == "BandPass" else 1.0 - band

    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    restored = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    return cv2.normalize(np.abs(restored), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ===================================================================
#  噪声与复原  (Noise & Restoration)
# ===================================================================

def add_noise(image: np.ndarray, params: dict) -> np.ndarray:
    mode = params["mode"]
    rng = np.random.default_rng()
    out = image.astype(np.float32)
    if mode == "Gaussian":
        out += rng.normal(0, float(params["sigma"]), image.shape)
        return _clip(out)

    amount = float(params["amount"])
    noisy = image.copy()
    total = int(amount * image.shape[0] * image.shape[1])
    if total <= 0:
        return noisy
    ys = rng.integers(0, image.shape[0], total)
    xs = rng.integers(0, image.shape[1], total)
    noisy[ys, xs] = 255
    ys = rng.integers(0, image.shape[0], total)
    xs = rng.integers(0, image.shape[1], total)
    noisy[ys, xs] = 0
    return noisy


def motion_blur(image: np.ndarray, params: dict) -> np.ndarray:
    """模拟运动模糊."""
    angle = float(params["angle"])
    length = max(1, int(params["length"]))
    # 构建运动模糊核
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    for i in range(length):
        x = int(center + (i - center) * np.cos(np.deg2rad(angle)))
        y = int(center + (i - center) * np.sin(np.deg2rad(angle)))
        x = np.clip(x, 0, length - 1)
        y = np.clip(y, 0, length - 1)
        kernel[y, x] = 1.0
    kernel /= kernel.sum()
    if image.ndim == 2:
        return cv2.filter2D(image, -1, kernel)
    return cv2.filter2D(image, -1, kernel)


def motion_deblur(image: np.ndarray, params: dict) -> np.ndarray:
    """运动模糊图像的维纳滤波复原."""
    angle = float(params["angle"])
    length = max(1, int(params["length"]))
    noise_power = float(params["noise_power"])
    gray = to_gray(image).astype(np.float64)

    # 构建 PSF
    psf = np.zeros((length, length), dtype=np.float64)
    c = length // 2
    for i in range(length):
        x = int(c + (i - c) * np.cos(np.deg2rad(angle)))
        y = int(c + (i - c) * np.sin(np.deg2rad(angle)))
        x = np.clip(x, 0, length - 1)
        y = np.clip(y, 0, length - 1)
        psf[y, x] = 1.0
    psf /= psf.sum()

    # 频域维纳滤波
    h_pad = np.zeros_like(gray)
    h_pad[:length, :length] = psf
    H = np.fft.fft2(np.fft.ifftshift(h_pad))
    G = np.fft.fft2(gray)
    # Wiener: F = conj(H) / (|H|^2 + K) * G
    H2 = np.abs(H) ** 2
    K = noise_power
    F_hat = (np.conj(H) / (H2 + K)) * G
    restored = np.fft.ifft2(F_hat)
    result = cv2.normalize(np.abs(restored), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


def wiener_filter(image: np.ndarray, params: dict) -> np.ndarray:
    """自适应维纳滤波去噪：基于局部统计量."""
    gray = to_gray(image).astype(np.float64)
    k = _odd(int(params["kernel"]))
    noise_var = float(params["noise_variance"])

    # 局部均值
    local_mean = cv2.blur(gray, (k, k))
    # 局部方差
    local_sq_mean = cv2.blur(gray ** 2, (k, k))
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)

    # Wiener 估计: f_hat = mu + max(0, var - sigma^2) / max(var, sigma^2) * (g - mu)
    ratio = np.divide(np.maximum(local_var - noise_var, 0),
                       np.maximum(local_var, noise_var),
                       out=np.zeros_like(local_var),
                       where=np.maximum(local_var, noise_var) > 1e-10)
    result = local_mean + ratio * (gray - local_mean)
    return _clip(result)


# ===================================================================
#  颜色图像处理  (Color Image Processing)
# ===================================================================

def color_space(image: np.ndarray, params: dict) -> np.ndarray:
    channel = params["channel"]
    if image.ndim == 2:
        gray = image
        if channel == "PseudoColor":
            return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return gray

    if channel in {"B", "G", "R"}:
        index = {"B": 0, "G": 1, "R": 2}[channel]
        return image[:, :, index]
    if channel == "Gray":
        return to_gray(image)
    if channel == "HSV-H":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]
    if channel == "HSV-S":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    if channel == "HSV-V":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    if channel == "Lab-L":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    if channel == "PseudoColor":
        return cv2.applyColorMap(to_gray(image), cv2.COLORMAP_JET)
    return image.copy()


def white_balance(image: np.ndarray, params: dict) -> np.ndarray:
    """白平衡：灰度世界法或完美反射法."""
    if image.ndim == 2:
        return image.copy()
    method = params["method"]
    b, g, r = cv2.split(image.astype(np.float64))

    if method == "GrayWorld":
        # 灰度世界假设：R、G、B 通道均值应相等
        avg_b, avg_g, avg_r = b.mean(), g.mean(), r.mean()
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        b = np.clip(b * (avg_gray / max(avg_b, 1e-6)), 0, 255)
        g = np.clip(g * (avg_gray / max(avg_g, 1e-6)), 0, 255)
        r = np.clip(r * (avg_gray / max(avg_r, 1e-6)), 0, 255)
    elif method == "PerfectReflector":
        # 完美反射法：假设图像中最亮的点是白色
        pct = float(params["percentile"]) / 100.0
        max_b = np.percentile(b, 100 - pct)
        max_g = np.percentile(g, 100 - pct)
        max_r = np.percentile(r, 100 - pct)
        b = np.clip(b * (255.0 / max(max_b, 1e-6)), 0, 255)
        g = np.clip(g * (255.0 / max(max_g, 1e-6)), 0, 255)
        r = np.clip(r * (255.0 / max(max_r, 1e-6)), 0, 255)
    else:
        return image.copy()

    return cv2.merge([b, g, r]).astype(np.uint8)


# ===================================================================
#  几何处理  (Geometric Processing)
# ===================================================================

def geometric_transform(image: np.ndarray, params: dict) -> np.ndarray:
    mode = params["mode"]
    h, w = image.shape[:2]
    if mode == "Rotate":
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), float(params["angle"]), float(params["scale"]))
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    if mode == "FlipHorizontal":
        return cv2.flip(image, 1)
    if mode == "FlipVertical":
        return cv2.flip(image, 0)
    factor = float(params["scale"])
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


# ===================================================================
#  压缩与质量评价  (Compression & Quality Assessment)
# ===================================================================

def compression_preview(image: np.ndarray, params: dict) -> np.ndarray:
    quality = int(params["quality"])
    ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image.copy()
    flag = cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
    return cv2.imdecode(enc, flag)


# ===================================================================
#  小波与多分辨率  (Wavelet & Multi-resolution)
# ===================================================================

def _haar_dwt2(image: np.ndarray, level: int = 1) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Haar 2D DWT 手动实现，返回每层 (LL, LH, HL, HH) 系数列表."""
    coeffs = []
    current = image.astype(np.float64)
    for _ in range(level):
        h, w = current.shape
        # 保证偶数尺寸
        h_even = h - (h % 2)
        w_even = w - (w % 2)
        current = current[:h_even, :w_even]

        # 行变换
        lo_rows = (current[:, 0::2] + current[:, 1::2]) / np.sqrt(2)
        hi_rows = (current[:, 0::2] - current[:, 1::2]) / np.sqrt(2)
        # 列变换
        LL = (lo_rows[0::2, :] + lo_rows[1::2, :]) / np.sqrt(2)
        LH = (lo_rows[0::2, :] - lo_rows[1::2, :]) / np.sqrt(2)
        HL = (hi_rows[0::2, :] + hi_rows[1::2, :]) / np.sqrt(2)
        HH = (hi_rows[0::2, :] - hi_rows[1::2, :]) / np.sqrt(2)
        coeffs.append((LL, LH, HL, HH))
        current = LL
    return coeffs


def dwt_decompose(image: np.ndarray, params: dict) -> np.ndarray:
    """2D Haar 小波分解可视化：将 LL/LH/HL/HH 拼成一张图."""
    gray = to_gray(image)
    level = min(2, max(1, int(params["level"])))
    coeffs = _haar_dwt2(gray, level)

    # 取最后一层的四个子带
    LL, LH, HL, HH = coeffs[-1]
    # 归一化各子带到 0-255 以便显示
    def _norm(x: np.ndarray) -> np.ndarray:
        return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    LL_disp = _norm(LL)
    LH_disp = _norm(LH)
    HL_disp = _norm(HL)
    HH_disp = _norm(HH)

    # 拼成 2x2 布局
    top = np.hstack([LL_disp, LH_disp])
    bot = np.hstack([HL_disp, HH_disp])
    combined = np.vstack([top, bot]).astype(np.uint8)
    return cv2.resize(combined, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)


def dwt_denoise(image: np.ndarray, params: dict) -> np.ndarray:
    """小波阈值去噪：软阈值处理细节系数."""
    gray = to_gray(image).astype(np.float64)
    level = min(2, max(1, int(params["level"])))
    threshold = float(params["threshold_value"])

    # 需要偶数尺寸
    h_even = gray.shape[0] - (gray.shape[0] % (2 ** level))
    w_even = gray.shape[1] - (gray.shape[1] % (2 ** level))
    current = gray[:h_even, :w_even]

    h, w = current.shape
    coeffs_list = []
    current_save = current.copy()
    for lvl in range(level):
        h_c, w_c = current.shape
        lo_rows = (current[:, 0::2] + current[:, 1::2]) / np.sqrt(2)
        hi_rows = (current[:, 0::2] - current[:, 1::2]) / np.sqrt(2)
        LL = (lo_rows[0::2, :] + lo_rows[1::2, :]) / np.sqrt(2)
        LH = (lo_rows[0::2, :] - lo_rows[1::2, :]) / np.sqrt(2)
        HL = (hi_rows[0::2, :] + hi_rows[1::2, :]) / np.sqrt(2)
        HH = (hi_rows[0::2, :] - hi_rows[1::2, :]) / np.sqrt(2)
        coeffs_list.append((LL, LH, HL, HH))
        current = LL

    # 软阈值处理除最后一层 LL 以外的所有细节系数
    for lvl_idx, (LL, LH, HL, HH) in enumerate(coeffs_list):
        noise_sigma = np.median(np.abs(HH)) / 0.6745 if np.median(np.abs(HH)) > 0 else threshold
        thresh = threshold * noise_sigma
        for band in [LH, HL, HH]:
            band[:] = np.sign(band) * np.maximum(np.abs(band) - thresh, 0)

    # 逆变换
    current = coeffs_list[-1][0]  # 最粗糙层的 LL
    for lvl_idx in range(level - 1, -1, -1):
        LL, LH, HL, HH = coeffs_list[lvl_idx]
        # 上采样列
        lo_rows = np.zeros((LL.shape[0] * 2, LL.shape[1]), dtype=np.float64)
        hi_rows = np.zeros_like(lo_rows)
        lo_rows[0::2, :] = (LL + LH) / np.sqrt(2)
        lo_rows[1::2, :] = (LL - LH) / np.sqrt(2)
        hi_rows[0::2, :] = (HL + HH) / np.sqrt(2)
        hi_rows[1::2, :] = (HL - HH) / np.sqrt(2)
        # 上采样行
        current = np.zeros((lo_rows.shape[0], lo_rows.shape[1] * 2), dtype=np.float64)
        current[:, 0::2] = (lo_rows + hi_rows) / np.sqrt(2)
        current[:, 1::2] = (lo_rows - hi_rows) / np.sqrt(2)

    result = _clip(current)
    # 如果原图更大，pad 回来
    if result.shape != gray.shape:
        result = cv2.resize(result, (gray.shape[1], gray.shape[0]))
    if image.ndim == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result


# ===================================================================
#  图像压缩编码  (Image Compression Coding)
# ===================================================================

def huffman_stats(image: np.ndarray, params: dict) -> np.ndarray:
    """哈夫曼编码统计可视化：返回带有编码信息标注的图像."""
    gray = to_gray(image)
    # 计算灰度直方图和熵
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = gray.size
    prob = hist / total
    prob_nonzero = prob[prob > 0]
    entropy = -np.sum(prob_nonzero * np.log2(prob_nonzero))

    # 简单的哈夫曼编码长度估计 (Shannon 近似)
    code_lengths = np.where(prob > 0, -np.log2(np.maximum(prob, 1e-10)), 0)
    avg_code_len = np.sum(prob * code_lengths)
    compression_ratio = 8.0 / max(avg_code_len, 0.01)
    unique = np.count_nonzero(hist)

    # 在图像上绘制统计信息
    result = to_bgr(gray)
    lines = [
        f"Entropy: {entropy:.3f} bits/pixel",
        f"Est Avg Code Len: {avg_code_len:.3f} bits",
        f"Compression Ratio: {compression_ratio:.2f}:1",
        f"Unique Gray Levels: {unique}",
        f"Original: 8 bits/pixel",
    ]
    y0 = 30
    for i, line in enumerate(lines):
        cv2.putText(result, line, (12, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
    return result


def dpcm_code(image: np.ndarray, params: dict) -> np.ndarray:
    """DPCM 预测编码：显示预测误差图像."""
    gray = to_gray(image).astype(np.float64)
    predictor = params["predictor"]
    h, w = gray.shape

    predicted = np.zeros_like(gray)
    error = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            if predictor == "PrevPixel":
                # f_hat(x,y) = f(x-1, y)  (前一列)
                predicted[y, x] = gray[y, x - 1] if x > 0 else 128.0
            elif predictor == "PrevLine":
                # f_hat(x,y) = f(x, y-1)  (上一行)
                predicted[y, x] = gray[y - 1, x] if y > 0 else 128.0
            elif predictor == "Average":
                # f_hat(x,y) = (f(x-1,y) + f(x,y-1)) / 2
                a = gray[y, x - 1] if x > 0 else 128.0
                b = gray[y - 1, x] if y > 0 else 128.0
                predicted[y, x] = (a + b) / 2.0
            elif predictor == "Planar":
                # f_hat(x,y) = f(x-1,y) + f(x,y-1) - f(x-1,y-1)
                a = gray[y, x - 1] if x > 0 else 128.0
                b = gray[y - 1, x] if y > 0 else 128.0
                c = gray[y - 1, x - 1] if x > 0 and y > 0 else 128.0
                predicted[y, x] = a + b - c
            error[y, x] = gray[y, x] - predicted[y, x]

    # 将误差映射到 0-255，以灰度 128 为零点
    error_disp = np.clip(error + 128, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(error_disp, cv2.COLOR_GRAY2BGR)
    return error_disp


# ===================================================================
#  图像表示与描述  (Image Representation & Description)
# ===================================================================

def chain_code(image: np.ndarray, params: dict) -> np.ndarray:
    """提取最大轮廓的边界链码并可视化."""
    gray = to_gray(image)
    direction = int(params["direction"])  # 4 或 8

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 128:
        binary = 255 - binary

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        result = to_bgr(gray).copy()
        cv2.putText(result, "No contour found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

    # 取最大轮廓
    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze(1)

    # 计算链码
    chain = []
    if direction == 8:
        dir_map = {0: 0, 1: 1, (1, 1): 1, 2: 2, (0, 1): 2, 3: 3, (-1, 1): 3,
                   4: 4, (-1, 0): 4, 5: 5, (-1, -1): 5, 6: 6, (0, -1): 6, 7: 7, (1, -1): 7}
    else:
        dir_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3}

    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        # 量化到 8 方向
        if dx > 0 and dy == 0: d = 0
        elif dx > 0 and dy < 0: d = 1
        elif dx == 0 and dy < 0: d = 2
        elif dx < 0 and dy < 0: d = 3
        elif dx < 0 and dy == 0: d = 4
        elif dx < 0 and dy > 0: d = 5
        elif dx == 0 and dy > 0: d = 6
        elif dx > 0 and dy > 0: d = 7
        else: d = 0
        chain.append(d)

    # 可视化
    result = to_bgr(gray).copy()
    # 绘制轮廓
    cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)
    # 标注起始点
    cv2.circle(result, tuple(pts[0]), 8, (0, 0, 255), -1)

    # 显示链码摘要
    if chain:
        chain_str = "".join(str(d) for d in chain[:60])
        if len(chain) > 60:
            chain_str += "..."
        cv2.putText(result, f"Chain ({direction}-dir, len={len(chain)}):",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)
        cv2.putText(result, chain_str, (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    return result


def glcm_texture(image: np.ndarray, params: dict) -> np.ndarray:
    """GLCM 纹理特征计算与可视化."""
    gray = to_gray(image)
    distance = int(params["distance"])
    # 量化到 16 级以控制 GLCM 大小
    levels = 16
    quantized = (gray / 16).astype(np.uint8)
    h, w = quantized.shape

    # 四个方向的 GLCM
    angles = {
        "0deg": (0, 1),
        "45deg": (-1, 1),
        "90deg": (-1, 0),
        "135deg": (-1, -1),
    }

    glcm_all = np.zeros((levels, levels), dtype=np.float64)
    for dy, dx in angles.values():
        glcm = np.zeros((levels, levels), dtype=np.float64)
        for y in range(h):
            for x in range(w):
                ny, nx = y + dy * distance, x + dx * distance
                if 0 <= ny < h and 0 <= nx < w:
                    glcm[quantized[y, x], quantized[ny, nx]] += 1
        if glcm.sum() > 0:
            glcm /= glcm.sum()
        glcm_all += glcm
    glcm_all /= 4.0

    # 计算 Haralick 特征
    i, j = np.mgrid[0:levels, 0:levels]
    contrast = np.sum(glcm_all * (i - j) ** 2)
    dissimilarity = np.sum(glcm_all * np.abs(i - j))
    homogeneity = np.sum(glcm_all / (1 + (i - j) ** 2))
    energy = np.sum(glcm_all ** 2)
    entropy_glcm = -np.sum(glcm_all * np.log2(np.maximum(glcm_all, 1e-12)))
    # 相关性
    mu_i = np.sum(i * glcm_all)
    mu_j = np.sum(j * glcm_all)
    si = np.sqrt(np.sum(glcm_all * (i - mu_i) ** 2))
    sj = np.sqrt(np.sum(glcm_all * (j - mu_j) ** 2))
    correlation = np.sum(glcm_all * (i - mu_i) * (j - mu_j)) / max(si * sj, 1e-10)

    # 可视化：在图像上打印纹理特征
    result = to_bgr(gray)
    lines = [
        f"GLCM ({levels} levels, d={distance})",
        f"Contrast:    {contrast:.4f}",
        f"Homogeneity: {homogeneity:.4f}",
        f"Energy:      {energy:.4f}",
        f"Entropy:     {entropy_glcm:.4f}",
        f"Correlation: {correlation:.4f}",
        f"Dissimilarity: {dissimilarity:.4f}",
    ]
    y0 = 28
    for i_line, line in enumerate(lines):
        cv2.putText(result, line, (12, y0 + i_line * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
    return result


def distance_transform_image(image: np.ndarray, params: dict) -> np.ndarray:
    """距离变换：对二值化图像计算 D4 或 D8 距离图."""
    gray = to_gray(image)
    mode = params["mode"]  # "Binary" 或 "Gradient"
    metric = params["metric"]  # "D4 (CityBlock)" 或 "D8 (Chessboard)" 或 "Euclidean"

    if mode == "Gradient":
        # 对边缘图像做距离变换
        edges = cv2.Canny(gray, 50, 150)
        binary = edges
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 128:
            binary = 255 - binary

    dist_type = {
        "D4 (CityBlock)": cv2.DIST_C,
        "D8 (Chessboard)": cv2.DIST_C,
        "Euclidean": cv2.DIST_L2,
    }[metric]
    mask_size = cv2.DIST_MASK_3 if metric == "D4 (CityBlock)" else cv2.DIST_MASK_5

    dist = cv2.distanceTransform(binary, dist_type, mask_size)
    dist_disp = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 伪彩色增强
    result = cv2.applyColorMap(dist_disp, cv2.COLORMAP_HOT)
    return result


# ===================================================================
#  Walsh-Hadamard 变换 (补充 Ch4 频域内容)
# ===================================================================

def _hadamard_matrix(n: int) -> np.ndarray:
    """生成 2^n 阶有序 Hadamard 矩阵."""
    H = np.array([[1]], dtype=np.float64)
    for _ in range(n):
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_transform(image: np.ndarray, params: dict) -> np.ndarray:
    """2D Walsh-Hadamard 变换：显示变换域系数幅度谱."""
    gray = to_gray(image).astype(np.float64)
    h, w = gray.shape
    # 取最近的 2^N 尺寸
    n_pow = max(1, int(np.ceil(np.log2(max(h, w)))))
    size = 2 ** n_pow
    # 填充到 2^N
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:h, :w] = gray

    H = _hadamard_matrix(n_pow)
    # 2D WHT: T = H * f * H
    transform = H @ padded @ H
    # 归一化
    transform /= size

    mode = params["mode"]
    if mode == "Spectrum":
        # 对数幅度谱
        magnitude = np.abs(transform)
        display = np.log1p(magnitude)
    elif mode == "Phase":
        display = np.angle(transform) + np.pi
    elif mode == "Sequency":
        # 按列率排序后的变换系数
        # Gray code ordering for sequency
        magnitude = np.abs(transform)
        display = np.log1p(magnitude)
    else:
        display = np.abs(transform)

    result = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 缩放回原尺寸
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST)


def hadamard_filter(image: np.ndarray, params: dict) -> np.ndarray:
    """WHT 域滤波：低通/高通/带通."""
    gray = to_gray(image).astype(np.float64)
    h, w = gray.shape
    n_pow = max(1, int(np.ceil(np.log2(max(h, w)))))
    size = 2 ** n_pow
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:h, :w] = gray

    H = _hadamard_matrix(n_pow)
    transform = H @ padded @ H / size

    # 构造列率域掩模 (sequency = 零交叉次数)
    radius = max(1, int(params["radius"]))
    mode = params["mode"]
    yy, xx = np.mgrid[:size, :size]
    # 使用简化的列率近似 (空间频率的绝对值)
    sequency = np.abs(xx - size // 2) + np.abs(yy - size // 2)
    sequency = np.minimum(sequency, size - sequency)

    low = (sequency <= radius).astype(np.float64)

    if mode == "LowPass":
        mask = low
    elif mode == "HighPass":
        mask = 1.0 - low
    elif mode == "BandPass":
        width = max(1, int(params["band_width"]))
        inner = max(1, radius - width)
        outer = radius + width
        band = ((sequency >= inner) & (sequency <= outer)).astype(np.float64)
        mask = band
    else:
        mask = 1.0 - ((sequency >= max(1, radius - int(params["band_width"]))) &
                       (sequency <= radius + int(params["band_width"]))).astype(np.float64)

    filtered = H @ (transform * mask) @ H
    filtered /= size
    result = _clip(filtered[:h, :w])
    return result


# ===================================================================
#  SUSAN 检测器 (补充 Ch10 高级检测)
# ===================================================================

def susan_detector(image: np.ndarray, params: dict) -> np.ndarray:
    """SUSAN 角点/边缘检测器."""
    gray = to_gray(image).astype(np.float64)
    h, w = gray.shape
    threshold = float(params["brightness_threshold"])
    mode = params["mode"]

    # 37 像素圆形掩模 (半径 3.4)
    mask_radius = 3.4
    mr = int(np.ceil(mask_radius))
    yy, xx = np.mgrid[-mr:mr + 1, -mr:mr + 1]
    dist = np.sqrt(xx ** 2 + yy ** 2)
    mask = dist <= mask_radius
    mask_pts = np.argwhere(mask)
    n_max = mask.sum()  # 37

    # 几何阈值
    if mode == "Corner":
        geom_threshold = 0.5 * n_max  # 角点响应阈值
    else:
        geom_threshold = 0.75 * n_max  # 边缘响应阈值

    response = np.zeros_like(gray)
    pad = mr
    padded = np.pad(gray, pad, mode='reflect')

    for (dy, dx) in mask_pts:
        ry, rx = dy - mr, dx - mr
        shifted = padded[pad + ry:pad + ry + h, pad + rx:pad + rx + w]
        # 统计相似像素数（USAN面积）
        similar = np.exp(-((shifted - gray) / threshold) ** 6)
        # 累积 USAN 计数
        response += similar

    # SUSAN 响应: R = max(0, g - n)
    susan = np.maximum(0, geom_threshold - response)

    if mode == "Corner":
        # 非极大值抑制
        susan_dilated = cv2.dilate(susan, np.ones((5, 5), np.uint8))
        susan[susan < susan_dilated * 0.9] = 0

    result = cv2.normalize(susan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if mode == "Corner":
        # 绘制角点标记
        result_bgr = to_bgr(gray.astype(np.uint8))
        corners = np.argwhere(susan > susan.max() * 0.3)
        for cy, cx in corners:
            cv2.circle(result_bgr, (cx, cy), 3, (0, 255, 255), -1)
        return result_bgr

    return result


# ===================================================================
#  Hu 矩与傅里叶描述子 (补充 Ch11)
# ===================================================================

def hu_moments(image: np.ndarray, params: dict) -> np.ndarray:
    """计算 Hu 不变矩并在图像上标注 7 个矩值."""
    gray = to_gray(image)
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 128:
        binary = 255 - binary

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = to_bgr(gray)

    if not contours:
        cv2.putText(result, "No contour found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

    # 取最大轮廓
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)

    # 计算 Hu 矩
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments)

    # 对数变换以便显示
    lines = ["Hu Moments (log-scaled):"]
    for i in range(7):
        val = -np.sign(hu[i][0]) * np.log10(abs(hu[i][0]) + 1e-12)
        lines.append(f"H{i + 1}: {val:.4f}")

    y0 = 28
    for i_line, line in enumerate(lines):
        cv2.putText(result, line, (12, y0 + i_line * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
    return result


def fourier_descriptors(image: np.ndarray, params: dict) -> np.ndarray:
    """傅里叶描述子：边界频域描述与重建."""
    gray = to_gray(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 128:
        binary = 255 - binary

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = to_bgr(gray)

    if not contours:
        cv2.putText(result, "No contour found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return result

    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze(1).astype(np.float64)

    # 边界表示为复数序列 z = x + jy
    if len(pts) < 4:
        return result
    z = pts[:, 0] + 1j * pts[:, 1]

    # DFT 得到傅里叶描述子
    fd = np.fft.fft(z)
    # 取幅度
    magnitude = np.abs(fd)
    magnitude[0] = 0  # 忽略 DC

    num_descriptors = int(params["num_descriptors"])
    num_descriptors = min(num_descriptors, len(fd))

    # 用前 N 个描述子重建边界
    fd_truncated = np.zeros_like(fd, dtype=np.complex128)
    fd_truncated[0] = fd[0]  # 保留 DC（位置信息）
    fd_truncated[1:num_descriptors] = fd[1:num_descriptors]
    fd_truncated[-num_descriptors + 1:] = fd[-num_descriptors + 1:]

    z_recon = np.fft.ifft(fd_truncated)
    pts_recon = np.column_stack([np.real(z_recon), np.imag(z_recon)]).astype(np.int32)

    # 绘制原边界和重建边界
    cv2.drawContours(result, [pts.astype(np.int32)], -1, (0, 255, 0), 2)
    cv2.polylines(result, [pts_recon], True, (255, 120, 0), 2)

    # 标注
    total_desc = len(fd) // 2
    cv2.putText(result, f"FD: {num_descriptors}/{total_desc} descriptors",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100), 1)
    cv2.putText(result, "Green: original  Orange: reconstructed",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # 频谱信息
    energy_ratio = np.sum(magnitude[:num_descriptors]) / max(np.sum(magnitude), 1e-10)
    cv2.putText(result, f"Energy ratio: {energy_ratio:.3f}",
                (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    return result


# ===================================================================
#  OPERATIONS 注册表
# ===================================================================

OPERATIONS: tuple[Operation, ...] = (
    # ---- 灰度变换与增强 ----
    Operation("negative", "图像反相", "灰度变换与增强", "执行 s = 255 - r，用于观察负片效果。", (), negative),
    Operation(
        "linear",
        "亮度/对比度",
        "灰度变换与增强",
        "线性变换 g = alpha * f + beta。",
        (
            ParamSpec("alpha", "对比度 alpha", "float", 1.15, 0.1, 3.0, 0.05),
            ParamSpec("beta", "亮度 beta", "float", 12.0, -100.0, 100.0, 1.0),
        ),
        linear_transform,
    ),
    Operation("log", "对数变换", "灰度变换与增强", "压缩高灰度动态范围，增强暗部细节。", (ParamSpec("scale", "比例", "float", 45.0, 1.0, 100.0, 1.0),), log_transform),
    Operation("gamma", "伽马校正", "灰度变换与增强", "幂律变换，gamma 小于 1 时提亮暗部。", (ParamSpec("gamma", "Gamma", "float", 0.6, 0.1, 5.0, 0.1),), gamma_transform),
    Operation("hist_eq", "直方图均衡化", "灰度变换与增强", "增强全局对比度，彩色图像仅处理亮度通道。", (), histogram_equalization),
    Operation("hist_spec", "直方图规定化", "灰度变换与增强", "将原图直方图匹配到指定目标分布（均匀/高斯/指数/暗调/亮调）。",
              (ParamSpec("target", "目标分布", "choice", "Uniform",
                         choices=("Uniform", "Gaussian", "Exponential", "DarkEmphasis", "BrightEmphasis")),),
              histogram_specification),
    Operation("clahe", "自适应直方图均衡", "灰度变换与增强", "局部对比度增强，适合光照不均图像。",
              (ParamSpec("clip_limit", "裁剪阈值", "float", 2.0, 0.5, 10.0, 0.5),
               ParamSpec("tile_size", "网格大小", "int", 8, 3, 32, 1)),
              clahe_equalization),
    Operation("slice", "灰度级分层", "灰度变换与增强", "突出指定灰度范围，可选择是否保留背景。",
              (ParamSpec("low", "下限", "int", 80, 0, 255, 1),
               ParamSpec("high", "上限", "int", 180, 0, 255, 1),
               ParamSpec("keep_background", "保留背景", "bool", True)),
              gray_level_slicing),
    Operation("bit_plane", "位平面分解", "灰度变换与增强", "提取8位灰度图中指定位平面（0=LSB, 7=MSB）。",
              (ParamSpec("plane", "位平面 (0-7)", "int", 7, 0, 7, 1),),
              bit_plane_decomposition),
    Operation("arithm_add", "图像加法去噪", "灰度变换与增强", "生成N个加噪副本后取平均，演示空域加法降噪原理。",
              (ParamSpec("n_samples", "样本数 N", "int", 16, 2, 100, 1),
               ParamSpec("noise_sigma", "噪声标准差", "float", 20.0, 5.0, 80.0, 1.0)),
              arithmetic_add),
    Operation("arithm_sub", "图像减法差异", "灰度变换与增强", "原图减去高斯模糊图，突出边缘和变化区域。",
              (ParamSpec("kernel", "模糊核", "int", 11, 3, 31, 2),
               ParamSpec("sigma", "模糊 Sigma", "float", 4.0, 0.5, 15.0, 0.5)),
              arithmetic_subtract),

    # ---- 空间滤波 ----
    Operation("mean", "均值滤波", "空间滤波", "线性平滑，适合减弱随机噪声。", (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2),), mean_filter),
    Operation("gaussian", "高斯滤波", "空间滤波", "按高斯权重平滑图像。",
              (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2),
               ParamSpec("sigma", "Sigma", "float", 1.2, 0.0, 10.0, 0.1)),
              gaussian_filter),
    Operation("median", "中值滤波", "空间滤波", "对椒盐噪声有较好抑制效果。", (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2),), median_filter),
    Operation("bilateral", "双边滤波", "空间滤波", "保边去噪，同时考虑空间距离和颜色差异。",
              (ParamSpec("diameter", "直径", "int", 7, 1, 25, 2),
               ParamSpec("sigma_color", "颜色 Sigma", "float", 60.0, 1.0, 200.0, 1.0),
               ParamSpec("sigma_space", "空间 Sigma", "float", 60.0, 1.0, 200.0, 1.0)),
              bilateral_filter),
    Operation("lap_sharp", "拉普拉斯锐化", "空间滤波", "利用二阶微分增强边缘和细节。",
              (ParamSpec("kernel", "核大小", "int", 3, 1, 7, 2),
               ParamSpec("strength", "强度", "float", 0.5, 0.1, 3.0, 0.1)),
              laplacian_sharpen),
    Operation("unsharp", "反锐化掩蔽", "空间滤波", "通过原图减去模糊图得到高频增强。",
              (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2),
               ParamSpec("sigma", "Sigma", "float", 1.0, 0.0, 10.0, 0.1),
               ParamSpec("amount", "增强量", "float", 1.0, 0.1, 5.0, 0.1)),
              unsharp_mask),

    # ---- 边缘检测 ----
    Operation("sobel", "Sobel 边缘", "边缘检测", "一阶梯度边缘检测。", (ParamSpec("kernel", "核大小", "int", 3, 1, 7, 2),), sobel_edge),
    Operation("prewitt", "Prewitt 边缘", "边缘检测", "Prewitt 梯度算子。", (), prewitt_edge),
    Operation("roberts", "Roberts 边缘", "边缘检测", "2x2 交叉梯度算子。", (), roberts_edge),
    Operation("kirsch", "Kirsch 边缘", "边缘检测", "8方向 Kirsch 罗盘算子，取各方向最大响应。", (), kirsch_edge),
    Operation("lap_edge", "Laplacian 边缘", "边缘检测", "二阶边缘检测。", (ParamSpec("kernel", "核大小", "int", 3, 1, 7, 2),), laplacian_edge),
    Operation("canny", "Canny 边缘", "边缘检测", "多阶段边缘检测，包含双阈值连接。",
              (ParamSpec("threshold1", "低阈值", "int", 80, 0, 255, 1),
               ParamSpec("threshold2", "高阈值", "int", 160, 0, 255, 1)),
              canny_edge),
    Operation("hough_lines", "Hough 直线检测", "边缘检测", "Hough 概率直线检测，在图像上绘制线段。",
              (ParamSpec("rho", "Rho 精度", "float", 1.0, 0.5, 5.0, 0.5),
               ParamSpec("theta", "Theta 精度(度)", "float", 1.0, 0.1, 10.0, 0.5),
               ParamSpec("threshold", "投票阈值", "int", 50, 5, 300, 5),
               ParamSpec("min_length", "最小线长", "float", 30.0, 5.0, 500.0, 5.0),
               ParamSpec("max_gap", "最大间隙", "float", 10.0, 1.0, 100.0, 1.0)),
              hough_lines),
    Operation("susan", "SUSAN 检测器", "边缘检测",
              "SUSAN 角点/边缘检测：基于 USAN 面积的最小核值相似区检测。",
              (ParamSpec("mode", "模式", "choice", "Corner", choices=("Corner", "Edge")),
               ParamSpec("brightness_threshold", "亮度阈值", "float", 25.0, 5.0, 60.0, 1.0)),
              susan_detector),
    Operation("hough_circles", "Hough 圆检测", "边缘检测", "Hough 梯度圆检测，在图像上绘制圆。",
              (ParamSpec("dp", "累加器分辨率 dp", "float", 1.2, 0.5, 3.0, 0.1),
               ParamSpec("min_dist", "圆心最小距离", "int", 50, 5, 500, 5),
               ParamSpec("param1", "Canny 高阈值", "int", 100, 10, 300, 10),
               ParamSpec("param2", "圆心置信度", "int", 30, 5, 200, 5),
               ParamSpec("min_radius", "最小半径", "int", 10, 1, 200, 1),
               ParamSpec("max_radius", "最大半径", "int", 100, 10, 500, 10)),
              hough_circles),

    # ---- 阈值与分割 ----
    Operation("threshold", "全局阈值分割", "阈值与分割", "固定阈值、Otsu 或 Triangle 自动阈值。",
              (ParamSpec("mode", "模式", "choice", "Otsu", choices=("Manual", "Otsu", "Triangle")),
               ParamSpec("threshold", "阈值", "int", 127, 0, 255, 1)),
              thresholding),
    Operation("adaptive_threshold", "自适应阈值", "阈值与分割", "根据局部邻域计算阈值。",
              (ParamSpec("method", "方法", "choice", "Gaussian", choices=("Mean", "Gaussian")),
               ParamSpec("block_size", "块大小", "int", 11, 3, 99, 2),
               ParamSpec("c", "C", "int", 2, -20, 20, 1)),
              adaptive_threshold),
    Operation("watershed", "分水岭分割", "阈值与分割", "基于距离变换和连通域标记的分水岭。",
              (ParamSpec("distance_ratio", "前景比例", "float", 0.45, 0.1, 0.9, 0.05),),
              watershed_segment),

    # ---- 形态学 ----
    Operation("morph", "形态学处理", "形态学", "腐蚀、膨胀、开闭运算、形态学梯度、顶帽、黑帽。",
              (ParamSpec("operation", "操作", "choice", "Open",
                         choices=("Erode", "Dilate", "Open", "Close", "Gradient", "TopHat", "BlackHat")),
               ParamSpec("shape", "结构元素", "choice", "Rect", choices=("Rect", "Ellipse", "Cross")),
               ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2),
               ParamSpec("iterations", "迭代", "int", 1, 1, 10, 1)),
              morph_operation),

    # ---- 频域处理 ----
    Operation("hadamard", "Walsh-Hadamard 变换", "频域处理",
              "2D Walsh-Hadamard 变换域幅度/相位谱可视化，显示列率域特征。",
              (ParamSpec("mode", "显示模式", "choice", "Spectrum",
                         choices=("Spectrum", "Phase")),),
              hadamard_transform),
    Operation("hadamard_filter", "WHT 域滤波", "频域处理",
              "在 Walsh-Hadamard 变换域做低通/高通/带通/带阻滤波。",
              (ParamSpec("mode", "模式", "choice", "LowPass",
                         choices=("LowPass", "HighPass", "BandPass", "BandStop")),
               ParamSpec("radius", "列率半径", "int", 30, 1, 300, 1),
               ParamSpec("band_width", "带宽", "int", 10, 1, 80, 1)),
              hadamard_filter),
    Operation("fft_filter", "频域滤波", "频域处理", "傅里叶低通、高通、带通、带阻滤波，可选择理想或高斯掩模。",
              (ParamSpec("mode", "模式", "choice", "LowPass",
                         choices=("LowPass", "HighPass", "BandPass", "BandStop")),
               ParamSpec("family", "滤波器", "choice", "Ideal", choices=("Ideal", "Gaussian")),
               ParamSpec("radius", "半径", "int", 40, 1, 400, 1),
               ParamSpec("band_width", "带宽", "int", 12, 1, 100, 1)),
              frequency_filter),

    # ---- 噪声与复原 ----
    Operation("noise", "添加噪声", "噪声与复原", "添加高斯噪声或椒盐噪声。",
              (ParamSpec("mode", "模式", "choice", "Gaussian", choices=("Gaussian", "SaltPepper")),
               ParamSpec("sigma", "高斯 Sigma", "float", 20.0, 1.0, 100.0, 1.0),
               ParamSpec("amount", "椒盐比例", "float", 0.02, 0.001, 0.2, 0.001)),
              add_noise),
    Operation("motion_blur", "运动模糊模拟", "噪声与复原",
              "模拟线性运动模糊：定义运动方向和长度生成模糊核。",
              (ParamSpec("angle", "运动角度", "float", 30.0, 0.0, 180.0, 1.0),
               ParamSpec("length", "模糊长度", "int", 15, 3, 50, 1)),
              motion_blur),
    Operation("motion_deblur", "运动模糊复原", "噪声与复原",
              "使用维纳滤波在频域对运动模糊图像进行复原（需先做运动模糊）。",
              (ParamSpec("angle", "运动角度", "float", 30.0, 0.0, 180.0, 1.0),
               ParamSpec("length", "模糊长度", "int", 15, 3, 50, 1),
               ParamSpec("noise_power", "噪声功率 K", "float", 0.01, 0.0001, 1.0, 0.001)),
              motion_deblur),
    Operation("wiener", "维纳自适应滤波", "噪声与复原",
              "基于局部统计量的自适应维纳滤波去噪，保留边缘同时平滑平坦区域。",
              (ParamSpec("kernel", "估计窗口", "int", 5, 3, 21, 2),
               ParamSpec("noise_variance", "噪声方差估计", "float", 400.0, 10.0, 5000.0, 10.0)),
              wiener_filter),

    # ---- 颜色图像处理 ----
    Operation("color", "颜色空间/通道", "颜色图像处理", "观察 B/G/R、灰度、HSV、Lab 通道或伪彩色映射。",
              (ParamSpec("channel", "通道", "choice", "HSV-H",
                         choices=("B", "G", "R", "Gray", "HSV-H", "HSV-S", "HSV-V", "Lab-L", "PseudoColor")),),
              color_space),
    Operation("white_balance", "白平衡校正", "颜色图像处理",
              "灰度世界法或完美反射法自动白平衡，校正偏色图像。",
              (ParamSpec("method", "方法", "choice", "GrayWorld",
                         choices=("GrayWorld", "PerfectReflector")),
               ParamSpec("percentile", "反射百分比", "float", 1.0, 0.1, 10.0, 0.1)),
              white_balance),

    # ---- 几何处理 ----
    Operation("geometry", "几何变换", "几何处理", "旋转、翻转、缩放。",
              (ParamSpec("mode", "模式", "choice", "Rotate",
                         choices=("Rotate", "FlipHorizontal", "FlipVertical", "Resize")),
               ParamSpec("angle", "角度", "float", 30.0, -180.0, 180.0, 1.0),
               ParamSpec("scale", "比例", "float", 1.0, 0.1, 3.0, 0.1)),
              geometric_transform),

    # ---- 压缩与质量评价 ----
    Operation("jpeg", "JPEG 压缩预览", "压缩与质量评价", "模拟有损压缩效果并计算 MSE/PSNR。",
              (ParamSpec("quality", "质量", "int", 35, 1, 100, 1),),
              compression_preview),

    # ---- 小波与多分辨率 ----
    Operation("dwt_decomp", "小波分解 DWT2", "小波与多分辨率",
              "Haar 2D 离散小波分解，显示 LL/LH/HL/HH 四子带。",
              (ParamSpec("level", "分解层数", "int", 1, 1, 2, 1),),
              dwt_decompose),
    Operation("dwt_denoise", "小波阈值去噪", "小波与多分辨率",
              "小波软阈值去噪：对细节系数做阈值处理，保留低频结构。",
              (ParamSpec("level", "分解层数", "int", 1, 1, 2, 1),
               ParamSpec("threshold_value", "阈值系数", "float", 3.0, 0.5, 20.0, 0.5)),
              dwt_denoise),

    # ---- 图像压缩编码 ----
    Operation("huffman", "哈夫曼编码统计", "图像压缩编码",
              "计算图像信息熵和哈夫曼编码的估计平均码长、压缩率。",
              (), huffman_stats),
    Operation("dpcm", "DPCM 预测编码", "图像压缩编码",
              "显示预测误差图像（前像素/上行/均值/平面预测器），灰度 128=零误差。",
              (ParamSpec("predictor", "预测器", "choice", "PrevPixel",
                         choices=("PrevPixel", "PrevLine", "Average", "Planar")),),
              dpcm_code),

    # ---- 图像表示与描述 ----
    Operation("chain_code", "边界链码", "图像表示与描述",
              "提取最大轮廓的边界链码（4或8方向），并高亮轮廓与起点。",
              (ParamSpec("direction", "方向数", "choice", "8", choices=("4", "8")),),
              chain_code),
    Operation("glcm", "GLCM 纹理特征", "图像表示与描述",
              "计算灰度共生矩阵及 Haralick 纹理特征：对比度、同质性、能量、熵、相关性。",
              (ParamSpec("distance", "像素距离", "int", 2, 1, 10, 1),),
              glcm_texture),
    Operation("dist_transform", "距离变换", "图像表示与描述",
              "对二值/边缘图计算 D4(城市街区)、D8(棋盘)或欧氏距离图，热力图显示。",
              (ParamSpec("mode", "输入模式", "choice", "Binary",
                         choices=("Binary", "Gradient")),
               ParamSpec("metric", "距离度量", "choice", "Euclidean",
                         choices=("D4 (CityBlock)", "D8 (Chessboard)", "Euclidean")),),
              distance_transform_image),
    Operation("hu_moments", "Hu 不变矩", "图像表示与描述",
              "计算最大轮廓的 7 个 Hu 不变矩（对数归一化），具有平移/旋转/缩放不变性。",
              (), hu_moments),
    Operation("fourier_desc", "傅里叶描述子", "图像表示与描述",
              "边界频域描述：用前 N 个傅里叶描述子重建轮廓，对比原边界与重建结果。",
              (ParamSpec("num_descriptors", "描述子数量", "int", 20, 3, 200, 1),),
              fourier_descriptors),
)


def categories() -> list[str]:
    ordered = []
    for op in OPERATIONS:
        if op.category not in ordered:
            ordered.append(op.category)
    return ordered


def by_category(category: str) -> list[Operation]:
    return [op for op in OPERATIONS if op.category == category]
