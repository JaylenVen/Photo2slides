from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


@dataclass(frozen=True)
class OcclusionRepair:
    """Conservative border-occlusion repair result."""

    image: np.ndarray
    mask: np.ndarray
    regions: tuple[tuple[int, int, int, int], ...]


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


def _smooth_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float32).reshape(-1, 1)
    return cv2.GaussianBlur(
        values,
        (1, 0),
        sigmaX=0,
        sigmaY=max(0.8, float(sigma)),
        borderType=cv2.BORDER_REFLECT,
    ).reshape(-1)


def _periodic_profile_component(profile: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the periodic part of a one-dimensional illumination profile."""

    values = np.asarray(profile, dtype=np.float32)
    length = int(values.size)
    if length < 64:
        return np.zeros_like(values), 0.0

    baseline = _smooth_profile(values, max(10.0, length / 24.0))
    residual = values - baseline
    residual -= float(np.mean(residual))
    windowed = residual * np.hanning(length).astype(np.float32)
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(frequencies > 0, 1.0 / frequencies, np.inf)
    candidate = (periods >= 5.0) & (periods <= min(96.0, length / 3.0))
    candidate[:2] = False
    candidate_power = power[candidate]
    if candidate_power.size < 4:
        return np.zeros_like(values), 0.0

    noise_floor = max(1e-6, float(np.median(candidate_power)))
    ranked = np.argsort(power)[::-1]
    selected: list[int] = []
    for index in ranked:
        if not candidate[index] or power[index] < noise_floor * 7.0:
            continue
        if any(abs(index - current) <= 2 for current in selected):
            continue
        selected.append(int(index))
        if len(selected) >= 4:
            break
    if not selected:
        return np.zeros_like(values), 0.0

    filtered = np.zeros_like(spectrum)
    for index in selected:
        start = max(1, index - 1)
        stop = min(len(spectrum), index + 2)
        filtered[start:stop] = spectrum[start:stop]
    component = np.fft.irfft(filtered, n=length).astype(np.float32)
    component *= 2.0 / max(0.35, float(np.mean(np.hanning(length))))

    periodic_energy = float(np.sum(power[[index for index in selected]]))
    total_energy = max(1e-6, float(np.sum(candidate_power)))
    score = float(np.clip(periodic_energy / total_energy, 0.0, 1.0))
    if float(np.std(component)) < 1.15 or score < 0.34:
        return np.zeros_like(values), score
    return np.clip(component, -28.0, 28.0), score


def _dominant_periodic_bins(profile: np.ndarray) -> tuple[list[int], float]:
    values = np.asarray(profile, dtype=np.float32)
    length = int(values.size)
    if length < 64:
        return [], 0.0
    baseline = _smooth_profile(values, max(10.0, length / 24.0))
    residual = (values - baseline) * np.hanning(length).astype(np.float32)
    power = np.abs(np.fft.rfft(residual)) ** 2
    frequencies = np.fft.rfftfreq(length)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(frequencies > 0, 1.0 / frequencies, np.inf)
    candidate = (periods >= 5.0) & (periods <= min(96.0, length / 3.0))
    candidate[:2] = False
    candidate_power = power[candidate]
    if candidate_power.size < 4:
        return [], 0.0
    noise_floor = max(1e-6, float(np.median(candidate_power)))
    dominant = int(np.argmax(np.where(candidate, power, -1.0)))
    if power[dominant] < noise_floor * 12.0:
        return [], 0.0

    selected: list[int] = []
    multiple = 1
    while dominant * multiple < len(power):
        center = dominant * multiple
        if periods[center] < 5.0:
            break
        start = max(1, center - 2)
        stop = min(len(power), center + 3)
        local = start + int(np.argmax(power[start:stop]))
        if power[local] >= noise_floor * (7.0 if multiple == 1 else 5.0):
            selected.extend(
                range(max(1, local - 1), min(len(power), local + 2))
            )
        multiple += 1
    selected = sorted(set(selected))
    score = float(
        np.clip(
            np.sum(power[selected]) / max(1e-6, float(np.sum(candidate_power))),
            0.0,
            1.0,
        )
    )
    return selected, score


def _remove_axis_banding(image: np.ndarray, axis: int) -> np.ndarray:
    """Remove globally repeated row or column banding without blurring glyph edges."""

    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = np.median(luminance, axis=axis)
    bins, score = _dominant_periodic_bins(profile)
    fundamental = float(np.mean(bins[: min(3, len(bins))])) if bins else 0.0
    period = len(profile) / fundamental if fundamental > 0 else float("inf")
    if score < 0.42 or not 5.0 <= period <= 42.0:
        return image

    transform_axis = 0 if axis == 1 else 1
    spectrum = np.fft.rfft(image.astype(np.float32), axis=transform_axis)
    if transform_axis == 0:
        spectrum[bins, :, :] *= 0.08
    else:
        spectrum[:, bins, :] *= 0.08
    corrected = np.fft.irfft(
        spectrum,
        n=image.shape[transform_axis],
        axis=transform_axis,
    )
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _profile_banding_score(profile: np.ndarray) -> float:
    bins, score = _dominant_periodic_bins(profile)
    fundamental = float(np.mean(bins[: min(3, len(bins))])) if bins else 0.0
    period = len(profile) / fundamental if fundamental > 0 else float("inf")
    return score if 5.0 <= period <= 42.0 else 0.0


def _equalize_axis_profile(image: np.ndarray, axis: int) -> np.ndarray:
    """Remove residual row/column exposure wobble after the Fourier notch."""

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    profile_length = image.shape[0] if axis == 1 else image.shape[1]
    sigma = max(12.0, profile_length / 18.0)
    for channel, strength, limit in (
        (0, 0.90, 20.0),
        (1, 0.42, 8.0),
        (2, 0.42, 8.0),
    ):
        field = lab[:, :, channel]
        profile = np.median(field, axis=axis)
        baseline = _smooth_profile(profile, sigma)
        correction = np.clip(baseline - profile, -limit, limit) * strength
        if axis == 1:
            field += correction[:, None]
        else:
            field += correction[None, :]
        lab[:, :, channel] = np.clip(field, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def suppress_periodic_banding(image: np.ndarray) -> np.ndarray:
    """Suppress camera/display moire that appears as repeated horizontal or vertical bars."""

    horizontal = _remove_axis_banding(image, axis=1)
    return _remove_axis_banding(horizontal, axis=0)


def periodic_banding_score(image: np.ndarray) -> float:
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    scores = [
        _profile_banding_score(np.median(luminance, axis=1)),
        _profile_banding_score(np.median(luminance, axis=0)),
    ]
    return float(max(scores))


def _equalize_masked_row_profile(
    image: np.ndarray,
    excluded_boxes: Iterable[tuple[int, int, int, int]],
) -> np.ndarray:
    height, width = image.shape[:2]
    valid = np.ones((height, width), dtype=bool)
    pad_x = max(3, width // 256)
    pad_y = max(2, height // 240)
    for box in excluded_boxes:
        x1, y1, x2, y2 = (int(value) for value in box)
        valid[
            max(0, y1 - pad_y) : min(height, y2 + pad_y),
            max(0, x1 - pad_x) : min(width, x2 + pad_x),
        ] = False

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    corrected = lab.copy()
    for channel, strength, limit in (
        (0, 1.0, 32.0),
        (1, 0.55, 10.0),
        (2, 0.55, 10.0),
    ):
        profile = np.full(height, np.nan, dtype=np.float32)
        for row in range(height):
            samples = lab[row, :, channel][valid[row]]
            if samples.size >= 32:
                profile[row] = float(np.median(samples))
        known = np.flatnonzero(np.isfinite(profile))
        if known.size < max(16, height // 4):
            return image.copy()
        profile = np.interp(np.arange(height), known, profile[known]).astype(np.float32)
        baseline = _smooth_profile(profile, max(12.0, height / 18.0))
        correction = np.clip(baseline - profile, -limit, limit) * strength
        corrected[:, :, channel] = np.clip(
            corrected[:, :, channel] + correction[:, None],
            0,
            255,
        )
    return cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)


def descreen_presentation_image(
    image: np.ndarray,
    excluded_boxes: Iterable[tuple[int, int, int, int]] = (),
) -> np.ndarray:
    """Smooth subtle residual screen bands after OCR has already been collected."""

    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_profile = np.median(luminance, axis=1)
    score = _profile_banding_score(row_profile)
    baseline = _smooth_profile(row_profile, max(12.0, len(row_profile) / 18.0))
    residual = row_profile - baseline
    residual_span = float(
        np.percentile(residual, 98) - np.percentile(residual, 2)
    )
    if score < 0.14 or residual_span > 32.0:
        return image.copy()
    boxes = tuple(excluded_boxes)
    balanced = (
        _equalize_masked_row_profile(image, boxes)
        if score >= 0.42 and boxes
        else image
    )
    denoise_strength = 10 if score >= 0.42 else 12
    return cv2.fastNlMeansDenoisingColored(
        balanced,
        None,
        denoise_strength,
        denoise_strength,
        7,
        21,
    )


def _dominant_color_fraction(image: np.ndarray) -> float:
    sample_width = min(192, image.shape[1])
    sample_height = min(108, image.shape[0])
    sample = cv2.resize(
        image,
        (sample_width, sample_height),
        interpolation=cv2.INTER_AREA,
    )
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    quantized = np.column_stack(
        (
            (lab[:, :, 0].reshape(-1) // 16),
            (lab[:, :, 1].reshape(-1) // 12),
            (lab[:, :, 2].reshape(-1) // 12),
        )
    )
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    return float(counts.max() / max(1, counts.sum()))


def correct_projector_illumination(image: np.ndarray) -> np.ndarray:
    """Reduce smooth projector/camera shading while retaining authored slide graphics."""

    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    dominant_fraction = _dominant_color_fraction(image)
    strength = float(
        np.interp(
            dominant_fraction,
            [0.08, 0.30, 0.62],
            [0.22, 0.56, 0.88],
        )
    )
    sigma = max(18.0, min(width, height) / 8.5)

    lightness = lab[:, :, 0]
    closing_size = max(17, int(round(min(width, height) / 18.0)))
    if closing_size % 2 == 0:
        closing_size += 1
    closed = cv2.morphologyEx(
        lightness,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_size, closing_size),
        ),
    )
    illumination = cv2.GaussianBlur(
        closed,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    target = float(np.median(illumination))
    correction = np.clip(target - illumination, -30.0, 30.0) * strength
    lab[:, :, 0] = np.clip(lightness + correction, 0, 255)

    chroma_strength = strength * 0.30
    for channel in (1, 2):
        chroma = lab[:, :, channel]
        field = cv2.GaussianBlur(
            chroma,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        chroma_target = float(np.median(field))
        lab[:, :, channel] = np.clip(
            chroma + np.clip(chroma_target - field, -10.0, 10.0) * chroma_strength,
            0,
            255,
        )
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def normalize_nearly_solid_background(image: np.ndarray) -> np.ndarray:
    """Pull the dominant authored background color toward a clean, even fill."""

    height, width = image.shape[:2]
    sample = cv2.resize(
        image,
        (min(192, width), min(108, height)),
        interpolation=cv2.INTER_AREA,
    )
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    quantized = np.column_stack(
        (
            sample_lab[:, :, 0].reshape(-1) // 16,
            sample_lab[:, :, 1].reshape(-1) // 12,
            sample_lab[:, :, 2].reshape(-1) // 12,
        )
    )
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_index = int(np.argmax(counts))
    dominant_fraction = float(counts[dominant_index] / max(1, counts.sum()))
    if dominant_fraction < 0.18:
        return image.copy()

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    full_quantized = np.stack(
        (
            lab[:, :, 0] // 16,
            lab[:, :, 1] // 12,
            lab[:, :, 2] // 12,
        ),
        axis=2,
    )
    dominant_selector = np.all(
        full_quantized == colors[dominant_index],
        axis=2,
    )
    if int(dominant_selector.sum()) < 64:
        return image.copy()
    target_bgr = np.median(image[dominant_selector], axis=0).astype(np.uint8)
    target_lab = cv2.cvtColor(
        target_bgr.reshape(1, 1, 3),
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)[0, 0]
    distance = np.linalg.norm(lab.astype(np.float32) - target_lab, axis=2)
    distance_limit = 44.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1),
    )
    selector = ((distance < distance_limit) & (gradient < 32.0)).astype(np.uint8)
    selector = cv2.morphologyEx(
        selector,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    alpha = cv2.GaussianBlur(
        selector.astype(np.float32),
        (0, 0),
        1.5,
    )[:, :, None] * (0.94 if dominant_fraction >= 0.55 else 0.82)
    return np.clip(
        image.astype(np.float32) * (1.0 - alpha)
        + target_bgr.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _bbox_overlap_fraction(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return intersection / first_area


def repair_border_occlusions(
    image: np.ndarray,
    protected_boxes: Iterable[tuple[int, int, int, int]] = (),
    forced_boxes: Iterable[tuple[int, int, int, int]] = (),
    solid_forced_boxes: Iterable[tuple[int, int, int, int]] = (),
) -> OcclusionRepair:
    """Repair compact foreground objects entering from a slide edge.

    This deliberately ignores wide bands and any candidate overlapping recognized
    slide content. It is intended for heads, shoulders, lecterns, and similar
    border-connected occlusions, not for inventing covered text.
    """

    height, width = image.shape[:2]
    slide_area = float(width * height)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[:, :, 0]
    row_median = np.median(luminance, axis=1)[:, None]
    column_median = np.median(luminance, axis=0)[None, :]
    expected = (row_median + column_median) * 0.5
    dark_delta = expected - luminance

    chroma_a = lab[:, :, 1]
    chroma_b = lab[:, :, 2]
    chroma_distance = np.sqrt(
        (chroma_a - np.median(chroma_a, axis=1)[:, None]) ** 2
        + (chroma_b - np.median(chroma_b, axis=1)[:, None]) ** 2
    )
    candidate = ((dark_delta > 26.0) | ((dark_delta > 15.0) & (chroma_distance > 18.0))).astype(
        np.uint8
    )
    forced_candidate_source = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    protected = [tuple(int(value) for value in box) for box in protected_boxes]
    protected_mask = np.zeros((height, width), dtype=np.uint8)
    protected_padding = max(2, int(round(min(width, height) * 0.003)))
    for x1, y1, x2, y2 in protected:
        cv2.rectangle(
            protected_mask,
            (
                max(0, x1 - protected_padding),
                max(0, y1 - protected_padding),
            ),
            (
                min(width - 1, x2 + protected_padding),
                min(height - 1, y2 + protected_padding),
            ),
            1,
            -1,
        )

    border_zone = np.zeros((height, width), dtype=np.uint8)
    border_zone[int(height * 0.76) :, :] = 1
    candidate &= border_zone
    candidate[protected_mask > 0] = 0
    edge_guard = max(3, int(round(height * 0.006)))
    candidate[height - edge_guard :, :] = 0
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
    )
    candidate = cv2.dilate(
        candidate,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    candidate[protected_mask > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    bottom_component_count = sum(
        int(stats[component, cv2.CC_STAT_TOP])
        + int(stats[component, cv2.CC_STAT_HEIGHT])
        >= height - 2
        and int(stats[component, cv2.CC_STAT_AREA]) >= slide_area * 0.0008
        for component in range(1, count)
    )
    cluttered_authored_border = bottom_component_count > 8
    accepted_mask = np.zeros_like(candidate)
    regions: list[tuple[int, int, int, int]] = []
    if cluttered_authored_border:
        band_top = int(round(height * 0.93))
        accepted_mask[band_top:, :] = 255
        accepted_mask[protected_mask > 0] = 0
        regions.append((0, band_top, width, height))
    solid_forced = {
        tuple(int(value) for value in box)
        for box in solid_forced_boxes
    }
    all_forced = dict.fromkeys(
        [
            *(tuple(int(value) for value in box) for box in forced_boxes),
            *solid_forced,
        ]
    )
    for forced in all_forced:
        x1, y1, x2, y2 = [
            int(value)
            for value in forced
        ]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        bbox = (x1, y1, x2, y2)
        if any(_bbox_overlap_fraction(bbox, box) > 0.24 for box in protected):
            continue
        if bbox in solid_forced:
            cv2.rectangle(
                accepted_mask,
                (x1, y1),
                (x2 - 1, y2 - 1),
                255,
                -1,
            )
            regions.append(bbox)
            continue
        forced_candidate = cv2.dilate(
            forced_candidate_source[y1:y2, x1:x2],
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        ) > 0
        if int(forced_candidate.sum()) >= max(12, int((x2 - x1) * (y2 - y1) * 0.08)):
            local_y, local_x = np.nonzero(forced_candidate)
            hull = cv2.convexHull(
                np.column_stack((local_x, local_y)).astype(np.int32)
            )
            refined_candidate = np.zeros(forced_candidate.shape, dtype=np.uint8)
            cv2.fillConvexPoly(refined_candidate, hull, 1)
            refined_candidate = cv2.dilate(
                refined_candidate,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 11)),
            )
            forced_candidate = refined_candidate > 0
            accepted_mask[y1:y2, x1:x2][forced_candidate] = 255
            local_y, local_x = np.nonzero(forced_candidate)
            regions.append(
                (
                    x1 + int(local_x.min()),
                    y1 + int(local_y.min()),
                    x1 + int(local_x.max()) + 1,
                    y1 + int(local_y.max()) + 1,
                )
            )
        else:
            cv2.rectangle(accepted_mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
            regions.append(bbox)
    for component in range(1, count):
        x, y, box_width, box_height, pixel_area = [int(value) for value in stats[component]]
        bbox = (x, y, x + box_width, y + box_height)
        bbox_area = max(1, box_width * box_height)
        touches_edge = (
            y + box_height >= height - 2
            or x <= 1
            or x + box_width >= width - 2
        )
        if not touches_edge:
            continue
        if cluttered_authored_border:
            continue
        if not (slide_area * 0.0015 <= pixel_area <= slide_area * 0.12):
            continue
        if box_width > width * 0.34 or box_height > height * 0.48:
            continue
        component_density = pixel_area / bbox_area
        if component_density < 0.28:
            continue
        touches_side = x <= 1 or x + box_width >= width - 2
        side_foreground = (
            touches_side
            and y >= height * 0.70
            and box_width <= width * 0.16
            and component_density >= 0.55
        )
        bottom_foreground = (
            y + box_height >= height - 2
            and y >= height * 0.74
            and box_width <= width * 0.20
            and component_density >= 0.55
        )
        solid_foreground = side_foreground or bottom_foreground
        if solid_foreground:
            horizontal_padding = max(
                6,
                min(
                    int(round(width * 0.02)),
                    int(round(box_width * 0.25)),
                ),
            )
            vertical_padding = max(
                4,
                min(
                    int(round(height * 0.02)),
                    int(round(box_height * 0.20)),
                ),
            )
            repair_bbox = (
                max(0, x - horizontal_padding),
                max(0, y - vertical_padding),
                min(width, x + box_width + horizontal_padding),
                height,
            )
        else:
            repair_bbox = bbox
        if any(
            _bbox_overlap_fraction(repair_bbox, box) > 0.12
            for box in protected
        ):
            continue
        if solid_foreground:
            sx1 = max(0, repair_bbox[0] - protected_padding * 4)
            sy1 = max(0, repair_bbox[1] - protected_padding * 4)
            sx2 = min(width, repair_bbox[2] + protected_padding * 4)
            sy2 = min(height, repair_bbox[3] + protected_padding * 4)
            ring_selector = protected_mask[sy1:sy2, sx1:sx2] == 0
            ring_selector[
                repair_bbox[1] - sy1 : repair_bbox[3] - sy1,
                repair_bbox[0] - sx1 : repair_bbox[2] - sx1,
            ] = False
            ring_pixels = lab[sy1:sy2, sx1:sx2][ring_selector]
            if len(ring_pixels) >= 12:
                spread = np.percentile(ring_pixels, 90, axis=0) - np.percentile(
                    ring_pixels,
                    10,
                    axis=0,
                )
                if spread[0] > 75.0 or max(spread[1:]) > 50.0:
                    continue
        if solid_foreground:
            cv2.rectangle(
                accepted_mask,
                (repair_bbox[0], repair_bbox[1]),
                (repair_bbox[2] - 1, repair_bbox[3] - 1),
                255,
                -1,
            )
            if repair_bbox not in regions:
                regions.append(repair_bbox)
            continue
        component_mask = (labels == component).astype(np.uint8)
        component_y, component_x = np.nonzero(component_mask)
        hull = cv2.convexHull(
            np.column_stack((component_x, component_y)).astype(np.int32)
        )
        refined_mask = np.zeros_like(component_mask)
        cv2.fillConvexPoly(refined_mask, hull, 255)
        refined_mask = cv2.dilate(
            refined_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)),
        )
        refined_mask[protected_mask > 0] = 0
        accepted_mask[refined_mask > 0] = 255
        refined_y, refined_x = np.nonzero(refined_mask)
        refined_bbox = (
            int(refined_x.min()),
            int(refined_y.min()),
            int(refined_x.max()) + 1,
            int(refined_y.max()) + 1,
        )
        if refined_bbox not in regions:
            regions.append(refined_bbox)

    if not regions:
        return OcclusionRepair(image.copy(), accepted_mask, ())

    repaired = image.copy()
    for x1, y1, x2, y2 in regions:
        region_mask = accepted_mask[y1:y2, x1:x2] > 0
        if not region_mask.any():
            continue
        padding = max(8, int(round(min(width, height) * 0.018)))
        sx1 = max(0, x1 - padding)
        sx2 = min(width, x2 + padding)
        if x1 == 0 and x2 == width:
            source_y1 = max(0, y1 - max(3, padding // 2))
            source_rows = repaired[source_y1:y1]
            if source_rows.size:
                reference_row = np.median(source_rows, axis=0).astype(np.uint8)
                reference_row = cv2.GaussianBlur(
                    reference_row[None, :, :],
                    (0, 0),
                    sigmaX=max(12.0, width / 48.0),
                    sigmaY=0,
                )[0]
                fill = np.repeat(reference_row[None, :, :], y2 - y1, axis=0)
                repaired[y1:y2][region_mask] = fill[region_mask]
            continue
        for y in range(y1, y2):
            row_mask = accepted_mask[y, x1:x2] > 0
            if not row_mask.any():
                continue
            left_samples = repaired[y, sx1:x1][
                (accepted_mask[y, sx1:x1] == 0)
                & (protected_mask[y, sx1:x1] == 0)
            ]
            right_samples = repaired[y, x2:sx2][
                (accepted_mask[y, x2:sx2] == 0)
                & (protected_mask[y, x2:sx2] == 0)
            ]
            if len(left_samples) >= 4 or len(right_samples) >= 4:
                left_color = (
                    np.median(left_samples, axis=0)
                    if len(left_samples) >= 4
                    else np.median(right_samples, axis=0)
                )
                right_color = (
                    np.median(right_samples, axis=0)
                    if len(right_samples) >= 4
                    else left_color
                )
                blend = np.linspace(
                    0.0,
                    1.0,
                    x2 - x1,
                    dtype=np.float32,
                )[:, None]
                fill_row = np.clip(
                    left_color * (1.0 - blend) + right_color * blend,
                    0,
                    255,
                ).astype(np.uint8)
                repaired[y, x1:x2][row_mask] = fill_row[row_mask]
                continue
            else:
                ring_y1 = max(0, y - padding)
                ring_y2 = min(height, y + padding + 1)
                samples = repaired[ring_y1:ring_y2, sx1:sx2][
                    (accepted_mask[ring_y1:ring_y2, sx1:sx2] == 0)
                    & (protected_mask[ring_y1:ring_y2, sx1:sx2] == 0)
                ]
                fill = (
                    np.median(samples, axis=0).astype(np.uint8)
                    if len(samples)
                    else np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
                )
            repaired[y, x1:x2][row_mask] = fill

    feather = cv2.GaussianBlur(
        accepted_mask.astype(np.float32) / 255.0,
        (0, 0),
        sigmaX=2.2,
        sigmaY=2.2,
    )[:, :, None]
    blended = np.clip(
        image.astype(np.float32) * (1.0 - feather)
        + repaired.astype(np.float32) * feather,
        0,
        255,
    ).astype(np.uint8)
    return OcclusionRepair(blended, accepted_mask, tuple(regions))


def enhance_slide_image(image: np.ndarray) -> np.ndarray:
    """Correct projector shading and display moire, then enhance slide legibility."""

    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image")

    banding_score = periodic_banding_score(image)
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    horizontal_banding = _profile_banding_score(np.median(luminance, axis=1))
    vertical_banding = _profile_banding_score(np.median(luminance, axis=0))
    balanced = suppress_periodic_banding(image)
    if banding_score >= 0.42:
        balanced = cv2.fastNlMeansDenoisingColored(
            balanced,
            None,
            6,
            6,
            7,
            21,
        )
        if horizontal_banding >= 0.42:
            balanced = _equalize_axis_profile(balanced, axis=1)
        if vertical_banding >= 0.42:
            balanced = _equalize_axis_profile(balanced, axis=0)
    balanced = correct_projector_illumination(balanced)
    balanced = _gray_world_balance(balanced)
    balanced = _automatic_gamma(balanced)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe_limit = 1.08 if banding_score >= 0.42 else 1.30
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.82)
    sharpen_amount = 0.10 if banding_score >= 0.42 else 0.24
    sharpened = cv2.addWeighted(
        enhanced,
        1.0 + sharpen_amount,
        blurred,
        -sharpen_amount,
        0,
    )
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
