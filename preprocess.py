"""
Image pre-processing pipeline: resize, color convert, CLAHE, blur.
"""
import cv2
import numpy as np

from config import (
    RESIZE_WIDTH,
    USE_CLAHE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    USE_GAUSSIAN_BLUR,
    GAUSSIAN_KERNEL,
)


def resize_keep_ratio(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize frame to target width, keeping aspect ratio."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    ratio = width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB color space."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE
    )
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full pre-processing pipeline.

    Returns RGB image ready for face_recognition library.
    """
    processed = resize_keep_ratio(frame, RESIZE_WIDTH)

    if USE_CLAHE:
        processed = apply_clahe(processed)

    if USE_GAUSSIAN_BLUR:
        processed = cv2.GaussianBlur(processed, GAUSSIAN_KERNEL, 0)

    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return rgb


def get_display_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame for OpenCV display (keep BGR)."""
    return resize_keep_ratio(frame, RESIZE_WIDTH)
