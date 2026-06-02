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


def compression_preview(image: np.ndarray, params: dict) -> np.ndarray:
    quality = int(params["quality"])
    ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image.copy()
    flag = cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
    return cv2.imdecode(enc, flag)


OPERATIONS: tuple[Operation, ...] = (
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
    Operation("clahe", "自适应直方图均衡", "灰度变换与增强", "局部对比度增强，适合光照不均图像。", (ParamSpec("clip_limit", "裁剪阈值", "float", 2.0, 0.5, 10.0, 0.5), ParamSpec("tile_size", "网格大小", "int", 8, 3, 32, 1)), clahe_equalization),
    Operation("slice", "灰度级分层", "灰度变换与增强", "突出指定灰度范围，可选择是否保留背景。", (ParamSpec("low", "下限", "int", 80, 0, 255, 1), ParamSpec("high", "上限", "int", 180, 0, 255, 1), ParamSpec("keep_background", "保留背景", "bool", True)), gray_level_slicing),
    Operation("mean", "均值滤波", "空间滤波", "线性平滑，适合减弱随机噪声。", (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2),), mean_filter),
    Operation("gaussian", "高斯滤波", "空间滤波", "按高斯权重平滑图像。", (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2), ParamSpec("sigma", "Sigma", "float", 1.2, 0.0, 10.0, 0.1)), gaussian_filter),
    Operation("median", "中值滤波", "空间滤波", "对椒盐噪声有较好抑制效果。", (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2),), median_filter),
    Operation("bilateral", "双边滤波", "空间滤波", "保边去噪，同时考虑空间距离和颜色差异。", (ParamSpec("diameter", "直径", "int", 7, 1, 25, 2), ParamSpec("sigma_color", "颜色 Sigma", "float", 60.0, 1.0, 200.0, 1.0), ParamSpec("sigma_space", "空间 Sigma", "float", 60.0, 1.0, 200.0, 1.0)), bilateral_filter),
    Operation("lap_sharp", "拉普拉斯锐化", "空间滤波", "利用二阶微分增强边缘和细节。", (ParamSpec("kernel", "核大小", "int", 3, 1, 7, 2), ParamSpec("strength", "强度", "float", 0.5, 0.1, 3.0, 0.1)), laplacian_sharpen),
    Operation("unsharp", "反锐化掩蔽", "空间滤波", "通过原图减去模糊图得到高频增强。", (ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2), ParamSpec("sigma", "Sigma", "float", 1.0, 0.0, 10.0, 0.1), ParamSpec("amount", "增强量", "float", 1.0, 0.1, 5.0, 0.1)), unsharp_mask),
    Operation("sobel", "Sobel 边缘", "边缘检测", "一阶梯度边缘检测。", (ParamSpec("kernel", "核大小", "int", 3, 1, 7, 2),), sobel_edge),
    Operation("prewitt", "Prewitt 边缘", "边缘检测", "Prewitt 梯度算子。", (), prewitt_edge),
    Operation("roberts", "Roberts 边缘", "边缘检测", "2x2 交叉梯度算子，适合展示早期边缘检测方法。", (), roberts_edge),
    Operation("lap_edge", "Laplacian 边缘", "边缘检测", "二阶边缘检测。", (ParamSpec("kernel", "核大小", "int", 3, 1, 7, 2),), laplacian_edge),
    Operation("canny", "Canny 边缘", "边缘检测", "多阶段边缘检测，包含双阈值连接。", (ParamSpec("threshold1", "低阈值", "int", 80, 0, 255, 1), ParamSpec("threshold2", "高阈值", "int", 160, 0, 255, 1)), canny_edge),
    Operation("threshold", "全局阈值分割", "阈值与分割", "固定阈值、Otsu 或 Triangle 自动阈值。", (ParamSpec("mode", "模式", "choice", "Otsu", choices=("Manual", "Otsu", "Triangle")), ParamSpec("threshold", "阈值", "int", 127, 0, 255, 1)), thresholding),
    Operation("adaptive_threshold", "自适应阈值", "阈值与分割", "根据局部邻域计算阈值。", (ParamSpec("method", "方法", "choice", "Gaussian", choices=("Mean", "Gaussian")), ParamSpec("block_size", "块大小", "int", 11, 3, 99, 2), ParamSpec("c", "C", "int", 2, -20, 20, 1)), adaptive_threshold),
    Operation("watershed", "分水岭分割", "阈值与分割", "基于距离变换和连通域标记的分水岭。", (ParamSpec("distance_ratio", "前景比例", "float", 0.45, 0.1, 0.9, 0.05),), watershed_segment),
    Operation("morph", "形态学处理", "形态学", "腐蚀、膨胀、开闭运算、形态学梯度、顶帽、黑帽。", (ParamSpec("operation", "操作", "choice", "Open", choices=("Erode", "Dilate", "Open", "Close", "Gradient", "TopHat", "BlackHat")), ParamSpec("shape", "结构元素", "choice", "Rect", choices=("Rect", "Ellipse", "Cross")), ParamSpec("kernel", "核大小", "int", 5, 1, 31, 2), ParamSpec("iterations", "迭代", "int", 1, 1, 10, 1)), morph_operation),
    Operation("fft_filter", "频域滤波", "频域处理", "傅里叶低通、高通、带通、带阻滤波，可选择理想或高斯掩模。", (ParamSpec("mode", "模式", "choice", "LowPass", choices=("LowPass", "HighPass", "BandPass", "BandStop")), ParamSpec("family", "滤波器", "choice", "Ideal", choices=("Ideal", "Gaussian")), ParamSpec("radius", "半径", "int", 40, 1, 400, 1), ParamSpec("band_width", "带宽", "int", 12, 1, 100, 1)), frequency_filter),
    Operation("noise", "添加噪声", "噪声与复原", "添加高斯噪声或椒盐噪声，可配合滤波方法观察复原效果。", (ParamSpec("mode", "模式", "choice", "Gaussian", choices=("Gaussian", "SaltPepper")), ParamSpec("sigma", "高斯 Sigma", "float", 20.0, 1.0, 100.0, 1.0), ParamSpec("amount", "椒盐比例", "float", 0.02, 0.001, 0.2, 0.001)), add_noise),
    Operation("color", "颜色空间/通道", "颜色图像处理", "观察 B/G/R、灰度、HSV、Lab 通道或伪彩色映射。", (ParamSpec("channel", "通道", "choice", "HSV-H", choices=("B", "G", "R", "Gray", "HSV-H", "HSV-S", "HSV-V", "Lab-L", "PseudoColor")),), color_space),
    Operation("geometry", "几何变换", "几何处理", "旋转、翻转、缩放。", (ParamSpec("mode", "模式", "choice", "Rotate", choices=("Rotate", "FlipHorizontal", "FlipVertical", "Resize")), ParamSpec("angle", "角度", "float", 30.0, -180.0, 180.0, 1.0), ParamSpec("scale", "比例", "float", 1.0, 0.1, 3.0, 0.1)), geometric_transform),
    Operation("jpeg", "JPEG 压缩预览", "压缩与质量评价", "模拟有损压缩效果，并在界面中计算 MSE 和 PSNR。", (ParamSpec("quality", "质量", "int", 35, 1, 100, 1),), compression_preview),
)


def categories() -> list[str]:
    ordered = []
    for op in OPERATIONS:
        if op.category not in ordered:
            ordered.append(op.category)
    return ordered


def by_category(category: str) -> list[Operation]:
    return [op for op in OPERATIONS if op.category == category]
