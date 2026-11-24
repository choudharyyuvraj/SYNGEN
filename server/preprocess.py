"""Style-reference preprocessing for the SYNGEN web demo.

Turns an arbitrary photo / scan of handwriting into the two tensors the
model's style encoder expects:

* ``style``   - grayscale word/line image, height 64, width <= 352, values 0..1
* ``laplace`` - high-frequency (Laplacian edge) map of the same image, 0..1

It also computes a few interpretable handwriting statistics (slant, stroke
width, ink density) that the frontend shows in the "Style analysis" panel.
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import cv2
import numpy as np

STYLE_H = 64
STYLE_W = 352          # max style width used by data_loader/loader.py
MIN_STYLE_W = 128      # loader.py prefers references wider than 128 px

LAPLACE_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


@dataclass
class StyleResult:
    original: np.ndarray   # uint8 grayscale, downscaled for preview
    cleaned: np.ndarray    # uint8, cropped to ink, background whitened
    style: np.ndarray      # uint8 (64 x W) - what the model sees
    laplace: np.ndarray    # uint8 (64 x W) - edge map fed to the high-freq encoder
    metrics: dict


# --------------------------------------------------------------------------- io
def decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read the image. Please upload a PNG or JPG file.")
    if img.shape[0] < 16 or img.shape[1] < 16:
        raise ValueError("Image is too small. Use a reference at least 16x16 pixels.")
    # keep memory/time bounded for very large phone photos
    max_side = 2000
    h, w = img.shape
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img


def to_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------- segmentation
def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask (255 = ink) that ignores ruled lines, shadows and specks."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # adaptive threshold copes with uneven phone-camera lighting
    block = max(15, (min(h, w) // 8) | 1)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, block, 10)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(adapt, otsu)

    # remove ruled / margin lines: long, *thin* horizontal or vertical runs.
    # Pen strokes are thicker than printed rules, so thick runs are protected.
    rect = cv2.getStructuringElement
    ink = mask > 0
    if ink.any():
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        pen = 2.0 * float(np.percentile(dt[ink], 95))      # thick part of pen strokes
    else:
        pen = 8.0
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (max(20, w // 15), 1)))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (1, max(20, h // 3))))
    long_runs = cv2.bitwise_or(horiz, vert)
    if long_runs.any():
        dl = cv2.distanceTransform(long_runs, cv2.DIST_L2, 3)
        rule = 2.0 * float(np.percentile(dl[long_runs > 0], 90))  # printed rule thickness
    else:
        rule = 2.0
    thin = int(round(min(max(3.0, rule + 2.0), max(3.0, 0.85 * pen))))
    horiz = cv2.subtract(horiz, cv2.morphologyEx(horiz, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (1, thin))))
    vert = cv2.subtract(vert, cv2.morphologyEx(vert, cv2.MORPH_OPEN, rect(cv2.MORPH_RECT, (thin, 1))))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), rect(cv2.MORPH_RECT, (3, 3)))
    mask = cv2.subtract(mask, lines)
    # re-join pen strokes that were cut where they crossed a rule
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rect(cv2.MORPH_RECT, (1, thin + 4)))

    # drop dust / noise components
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(10, 0.02 * float(areas.max()))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def crop_and_clean(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ink_mask(gray)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No handwriting found in the image. Try a clearer, closer photo.")
    pad = max(4, int(0.04 * (ys.max() - ys.min() + 1)))
    y0, y1 = max(0, ys.min() - pad), min(gray.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(gray.shape[1], xs.max() + pad + 1)
    crop, m = gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    # whiten everything that is not ink; stretch ink contrast
    m_d = cv2.dilate(m, np.ones((3, 3), np.uint8))
    ink_vals = crop[m > 0]
    lo = float(np.percentile(ink_vals, 2)) if ink_vals.size else 0.0
    hi = float(np.percentile(crop[m_d == 0], 50)) if np.any(m_d == 0) else 255.0
    hi = max(hi, lo + 1)
    stretched = np.clip((crop.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    clean = np.where(m_d > 0, stretched, 255).astype(np.uint8)
    return clean, m


# ---------------------------------------------------------------- model input
def repo_enhance(img: np.ndarray) -> np.ndarray:
    """Same filter chain as Random_StyleIAMDataset.get_style_ref in data_loader/loader.py."""
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


def fit_to_style(clean: np.ndarray, fit: str) -> tuple[np.ndarray, bool]:
    h, w = clean.shape
    if fit == "stretch":   # behaviour of custom_5percent/test_single.py
        return cv2.resize(clean, (STYLE_W, STYLE_H), interpolation=cv2.INTER_CUBIC), False
    new_w = max(1, round(w * STYLE_H / h))
    interp = cv2.INTER_AREA if h > STYLE_H else cv2.INTER_CUBIC
    out = cv2.resize(clean, (new_w, STYLE_H), interpolation=interp)
    truncated = new_w > STYLE_W
    if truncated:
        out = out[:, :STYLE_W]
    if out.shape[1] < MIN_STYLE_W:   # pad with white paper, like the loader does
        pad = np.full((STYLE_H, MIN_STYLE_W - out.shape[1]), 255, np.uint8)
        out = np.hstack([out, pad])
    return out, truncated


def laplace_map(style: np.ndarray) -> np.ndarray:
    resp = cv2.filter2D(style.astype(np.float32) / 255.0, -1, LAPLACE_KERNEL)
    edges = (np.abs(resp) > 0.15).astype(np.uint8) * 255
    return edges


# -------------------------------------------------------------------- metrics
def estimate_slant(ink: np.ndarray) -> float:
    """Shear-search slant estimate (degrees, positive = leaning right).

    Each candidate shear straightens strokes of one slant; the shear whose
    vertical projection is sharpest (largest sum of squared column counts)
    matches the writer's slant.
    """
    if ink.sum() < 20:
        return 0.0
    h, w = ink.shape
    cy = h / 2.0
    pad = h
    src = cv2.copyMakeBorder(ink.astype(np.float32), 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    best, best_score = 0.0, -1.0
    for deg in np.arange(-45, 45.5, 1.5):
        t = math.tan(math.radians(deg))
        m = np.float32([[1, t, -t * cy], [0, 1, 0]])
        sheared = cv2.warpAffine(src, m, (src.shape[1], h), flags=cv2.INTER_NEAREST)
        score = float(np.square(sheared.sum(axis=0)).sum())
        if score > best_score:
            best, best_score = float(deg), score
    return best


def style_metrics(style: np.ndarray, truncated: bool) -> dict:
    ink = (style < 128).astype(np.uint8)
    density = float(ink.mean())
    slant = estimate_slant(ink)
    stroke = 0.0
    if ink.any():
        dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        stroke = float(2.0 * np.percentile(dt[ink > 0], 90))
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    return {
        "slant_deg": round(slant, 1),
        "stroke_px": round(stroke, 2),
        "ink_density": round(density * 100, 1),
        "ink_width_px": int(cols.max() - cols.min() + 1) if cols.size else 0,
        "ink_height_px": int(rows.max() - rows.min() + 1) if rows.size else 0,
        "style_width_px": int(style.shape[1]),
        "truncated": truncated,
    }


def process_style(data: bytes, enhance: bool = True, fit: str = "aspect") -> StyleResult:
    gray = decode_upload(data)
    clean, _ = crop_and_clean(gray)
    style, truncated = fit_to_style(clean, fit)
    if enhance:
        style = repo_enhance(style)
    lap = laplace_map(style)

    preview = gray
    if preview.shape[1] > 720:
        s = 720 / preview.shape[1]
        preview = cv2.resize(preview, (720, round(preview.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return StyleResult(preview, clean, style, lap, style_metrics(style, truncated))
