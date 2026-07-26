from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_AUTO_WIDTH = 1280
MAX_AUTO_WIDTH = 2560
SIZE_MULTIPLE = 32


@dataclass(frozen=True)
class PreparedOCRImage:
    """OCR-ready image and its coordinate scale relative to the source image."""

    image: np.ndarray
    scale_x: float = 1.0
    scale_y: float = 1.0


def _edge_length(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=np.float32) - second))


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def resolve_rectified_size(
    corners: np.ndarray,
    *,
    requested_width: int = 0,
    requested_height: int = 0,
) -> tuple[int, int]:
    """Choose a stable 16:9 working resolution for a detected slide."""

    if requested_width and requested_height:
        return int(requested_width), int(requested_height)

    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("corners must have shape (4, 2)")

    top = _edge_length(points[0], points[1])
    bottom = _edge_length(points[3], points[2])
    left = _edge_length(points[0], points[3])
    right = _edge_length(points[1], points[2])
    measured_width = max(top, bottom, ((left + right) * 0.5) * 16 / 9)
    target_width = float(np.clip(measured_width, MIN_AUTO_WIDTH, MAX_AUTO_WIDTH))
    width = _round_to_multiple(target_width, SIZE_MULTIPLE)
    height = width * 9 // 16
    return int(width), int(height)


def _gray_world_balance(image: np.ndarray) -> np.ndarray:
    float_image = image.astype(np.float32)
    channel_means = float_image.reshape(-1, 3).mean(axis=0)
    target = float(channel_means.mean())
    gains = np.clip(target / np.maximum(channel_means, 1.0), 0.86, 1.16)
    return np.clip(float_image * gains, 0, 255).astype(np.uint8)


def _automatic_gamma(image: np.ndarray) -> np.ndarray:
    gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    if 92 <= gray_mean <= 188:
        return image
    desired = 142.0
    gamma = np.log(desired / 255.0) / np.log(max(gray_mean, 1.0) / 255.0)
    gamma = float(np.clip(gamma, 0.72, 1.35))
    table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)]).astype(
        np.uint8
    )
    return cv2.LUT(image, table)


def enhance_slide_image(image: np.ndarray) -> np.ndarray:
    """Apply conservative projector/photo correction while preserving slide colors."""

    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image")

    balanced = _gray_world_balance(image)
    balanced = _automatic_gamma(balanced)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.85)
    sharpened = cv2.addWeighted(enhanced, 1.34, blurred, -0.34, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def prepare_ocr_image(image: np.ndarray) -> PreparedOCRImage:
    """Create a same-size, high-contrast BGR image for OCR coordinate stability."""

    if image is None or image.size == 0:
        raise ValueError("image is empty")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(12, 12))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 0.7)
    contrast = cv2.addWeighted(contrast, 1.4, blurred, -0.4, 0)
    ocr_image = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    return PreparedOCRImage(image=ocr_image)
