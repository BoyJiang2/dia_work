"""数字图像处理操作单元测试 — 无需 pytest 依赖."""
from __future__ import annotations

import numpy as np

from operations import (
    OPERATIONS,
    _odd,
    adaptive_threshold,
    add_noise,
    arithmetic_add,
    arithmetic_subtract,
    bilateral_filter,
    bit_plane_decomposition,
    by_category,
    canny_edge,
    categories,
    chain_code,
    clahe_equalization,
    color_space,
    compression_preview,
    distance_transform_image,
    dpcm_code,
    dwt_decompose,
    dwt_denoise,
    fourier_descriptors,
    frequency_filter,
    gamma_transform,
    gaussian_filter,
    geometric_transform,
    glcm_texture,
    gray_level_slicing,
    hadamard_filter,
    hadamard_transform,
    histogram_equalization,
    histogram_specification,
    hough_circles,
    hough_lines,
    hu_moments,
    huffman_stats,
    kirsch_edge,
    laplacian_edge,
    laplacian_sharpen,
    linear_transform,
    log_transform,
    mean_filter,
    median_filter,
    morph_operation,
    motion_blur,
    motion_deblur,
    negative,
    prewitt_edge,
    roberts_edge,
    sobel_edge,
    susan_detector,
    thresholding,
    unsharp_mask,
    watershed_segment,
    white_balance,
    wiener_filter,
)


def gray() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(40, 200, (128, 128), dtype=np.uint8)


def color() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(40, 200, (128, 128, 3), dtype=np.uint8)


# ---- Helpers ----
def test_odd():
    assert _odd(3) == 3
    assert _odd(4) == 5

# ---- 灰度变换 ----
def test_negative(): assert negative(gray(), {}).shape == gray().shape
def test_linear(): assert linear_transform(gray(), {"alpha": 1.5, "beta": 10.0}).shape == gray().shape
def test_log(): assert log_transform(gray(), {"scale": 45.0}).shape == gray().shape
def test_gamma(): assert gamma_transform(gray(), {"gamma": 0.6}).shape == gray().shape
def test_hist_eq(): assert histogram_equalization(color(), {}).shape == color().shape
def test_hist_spec():
    for t in ("Uniform", "Gaussian", "Exponential", "DarkEmphasis", "BrightEmphasis"):
        assert histogram_specification(gray(), {"target": t}).shape == gray().shape
def test_clahe(): assert clahe_equalization(color(), {"clip_limit": 2.0, "tile_size": 8}).shape == color().shape
def test_slice(): assert gray_level_slicing(gray(), {"low": 80, "high": 180, "keep_background": True}).shape == gray().shape
def test_bit_plane():
    for p in range(8):
        r = bit_plane_decomposition(gray(), {"plane": p})
        assert set(np.unique(r)).issubset({0, 255})
def test_arithm_add(): assert arithmetic_add(color(), {"n_samples": 4, "noise_sigma": 15.0}).shape == color().shape
def test_arithm_sub(): assert arithmetic_subtract(gray(), {"kernel": 5, "sigma": 3.0}).shape == gray().shape

# ---- 空间滤波 ----
def test_mean(): assert mean_filter(color(), {"kernel": 5}).shape == color().shape
def test_gauss(): assert gaussian_filter(color(), {"kernel": 5, "sigma": 1.2}).shape == color().shape
def test_median(): assert median_filter(color(), {"kernel": 5}).shape == color().shape
def test_bilateral(): assert bilateral_filter(color(), {"diameter": 7, "sigma_color": 60.0, "sigma_space": 60.0}).shape == color().shape
def test_lap_sharp(): assert laplacian_sharpen(color(), {"kernel": 3, "strength": 0.5}).shape == color().shape
def test_unsharp(): assert unsharp_mask(color(), {"kernel": 5, "sigma": 1.0, "amount": 1.0}).shape == color().shape

# ---- 边缘检测 ----
def test_sobel(): assert sobel_edge(color(), {"kernel": 3}).ndim == 2
def test_prewitt(): assert prewitt_edge(color(), {}).ndim == 2
def test_roberts(): assert roberts_edge(color(), {}).ndim == 2
def test_lap_edge(): assert laplacian_edge(color(), {"kernel": 3}).ndim == 2
def test_canny(): assert canny_edge(color(), {"threshold1": 80, "threshold2": 160}).ndim == 2
def test_kirsch(): assert kirsch_edge(color(), {}).ndim == 2
def test_hough_lines(): assert hough_lines(color(), {"rho": 1.0, "theta": 1.0, "threshold": 50, "min_length": 20.0, "max_gap": 10.0}).shape == color().shape
def test_hough_circles(): assert hough_circles(color(), {"dp": 1.2, "min_dist": 30, "param1": 100, "param2": 30, "min_radius": 5, "max_radius": 60}).shape == color().shape
def test_susan():
    assert susan_detector(color(), {"mode": "Corner", "brightness_threshold": 25.0}) is not None
    assert susan_detector(color(), {"mode": "Edge", "brightness_threshold": 25.0}) is not None

# ---- 阈值与分割 ----
def test_threshold():
    for m in ("Manual", "Otsu", "Triangle"):
        assert thresholding(color(), {"mode": m, "threshold": 127}).ndim == 2
def test_adaptive_thresh(): assert adaptive_threshold(color(), {"method": "Gaussian", "block_size": 11, "c": 2}).ndim == 2
def test_watershed(): assert watershed_segment(color(), {"distance_ratio": 0.45}).shape == color().shape

# ---- 形态学 ----
def test_morph():
    for op in ("Erode", "Dilate", "Open", "Close", "Gradient", "TopHat", "BlackHat"):
        assert morph_operation(color(), {"operation": op, "shape": "Rect", "kernel": 5, "iterations": 1}).ndim == 2

# ---- 频域 ----
def test_fft_filter():
    for m in ("LowPass", "HighPass", "BandPass", "BandStop"):
        assert frequency_filter(color(), {"mode": m, "family": "Gaussian", "radius": 30, "band_width": 12}).ndim == 2
def test_hadamard(): assert hadamard_transform(gray(), {"mode": "Spectrum"}).shape == gray().shape
def test_hadamard_filt():
    for m in ("LowPass", "HighPass", "BandPass", "BandStop"):
        assert hadamard_filter(gray(), {"mode": m, "radius": 20, "band_width": 10}).shape == gray().shape

# ---- 噪声与复原 ----
def test_noise():
    assert add_noise(color(), {"mode": "Gaussian", "sigma": 20.0, "amount": 0.02}).shape == color().shape
    assert add_noise(color(), {"mode": "SaltPepper", "sigma": 20.0, "amount": 0.02}).shape == color().shape
def test_motion_blur(): assert motion_blur(gray(), {"angle": 30.0, "length": 10}).shape == gray().shape
def test_motion_deblur():
    blurred = motion_blur(gray(), {"angle": 30.0, "length": 10})
    assert motion_deblur(blurred, {"angle": 30.0, "length": 10, "noise_power": 0.01}).shape == gray().shape
def test_wiener(): assert wiener_filter(gray(), {"kernel": 5, "noise_variance": 400.0}).shape == gray().shape

# ---- 颜色 ----
def test_color():
    for ch in ("B", "G", "R", "Gray", "HSV-H", "HSV-S", "HSV-V", "Lab-L", "PseudoColor"):
        assert color_space(color(), {"channel": ch}) is not None
def test_wb():
    assert white_balance(color(), {"method": "GrayWorld", "percentile": 1.0}).shape == color().shape
    assert white_balance(color(), {"method": "PerfectReflector", "percentile": 1.0}).shape == color().shape

# ---- 几何 ----
def test_geometry():
    for mode in ("Rotate", "FlipHorizontal", "FlipVertical", "Resize"):
        assert geometric_transform(color(), {"mode": mode, "angle": 30.0, "scale": 1.0}) is not None

# ---- 压缩 ----
def test_jpeg(): assert compression_preview(color(), {"quality": 35}).shape == color().shape

# ---- 小波 ----
def test_dwt():
    for lv in (1, 2):
        assert dwt_decompose(gray(), {"level": lv}).shape == gray().shape
        assert dwt_denoise(gray(), {"level": lv, "threshold_value": 3.0}).shape == gray().shape

# ---- 压缩编码 ----
def test_huffman(): assert huffman_stats(gray(), {}) is not None
def test_dpcm():
    for pred in ("PrevPixel", "PrevLine", "Average", "Planar"):
        assert dpcm_code(gray(), {"predictor": pred}).shape == gray().shape

# ---- 图像表示与描述 ----
def test_chain():
    for d in ("4", "8"):
        assert chain_code(color(), {"direction": d}).shape == color().shape
def test_glcm(): assert glcm_texture(gray(), {"distance": 2}) is not None
def test_dist_trans():
    for metric in ("D4 (CityBlock)", "D8 (Chessboard)", "Euclidean"):
        assert distance_transform_image(gray(), {"mode": "Binary", "metric": metric}) is not None
def test_hu(): assert hu_moments(color(), {}).shape == color().shape
def test_fd(): assert fourier_descriptors(color(), {"num_descriptors": 15}).shape == color().shape

# ---- 注册表完整性 ----
def test_registry():
    assert len(OPERATIONS) == 50
    seen = set()
    for op in OPERATIONS:
        assert op.key not in seen, f"Duplicate: {op.key}"
        seen.add(op.key)
        assert callable(op.apply)
        assert op.category and op.name

def test_cats():
    cats = categories()
    assert len(cats) == 13
    assert sum(len(by_category(c)) for c in cats) == len(OPERATIONS)

# ---- 全部使用默认参数 ----
def test_all_defaults():
    g = gray()
    for op in OPERATIONS:
        defaults = {p.name: p.default for p in op.params}
        try:
            r = op.apply(g.copy(), defaults)
            assert r is not None and r.size > 0, f"{op.key} returned empty"
        except Exception as e:
            raise AssertionError(f"{op.key} FAILED: {e}") from e


if __name__ == "__main__":
    import sys, traceback
    all_tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = fail = 0
    for fn in all_tests:
        try:
            fn()
            ok += 1
            print(f"  OK  {fn.__name__}")
        except Exception:
            fail += 1
            print(f"  FAIL {fn.__name__}: {traceback.format_exc().strip().split(chr(10))[-1]}")
    print(f"\n{ok} passed, {fail} failed out of {len(all_tests)}")
    sys.exit(1 if fail else 0)
