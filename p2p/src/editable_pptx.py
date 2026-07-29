from __future__ import annotations

import base64
import html
import json
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from image_enhancement import (
    descreen_presentation_image,
    normalize_nearly_solid_background,
    prepare_ocr_image,
    repair_border_occlusions,
)


SLIDE_SIZE = (1280, 720)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECONSTRUCTION_DIR = (
    PROJECT_ROOT / "data" / "intermediate" / "runs" / "latest" / "reconstruction"
)
MAX_NATIVE_SHAPES = 80
MAX_VISUAL_ASSETS = 24
PADDLEOCR_VL_PIPELINE_VERSION = "v1.6"
PADDLEOCR_VL_MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"
DEFAULT_OPENAI_VISION_MODEL = "gpt-5.6"
VISUAL_LAYOUT_LABELS = {
    "algorithm",
    "chart",
    "display_formula",
    "figure",
    "formula",
    "footer_image",
    "header_image",
    "image",
    "inline_formula",
    "seal",
    "table",
}
VLM_TEXT_LAYOUT_LABELS = {
    "abstract",
    "aside_text",
    "caption",
    "chart_title",
    "content",
    "doc_title",
    "figure_title",
    "footer",
    "footnote",
    "formula_number",
    "header",
    "number",
    "ocr",
    "paragraph_title",
    "list",
    "list_item",
    "reference",
    "reference_content",
    "spotting",
    "subtitle",
    "table_title",
    "text",
    "title",
    "vertical_text",
    "vision_footnote",
}
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _configure_windows_torch_dll_search_path() -> None:
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("torch")
    package_locations = list(spec.submodule_search_locations or []) if spec else []
    if not package_locations:
        return
    torch_lib = Path(package_locations[0]) / "lib"
    if not torch_lib.is_dir():
        return
    torch_lib_text = str(torch_lib)
    current = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    os.environ["PATH"] = os.pathsep.join(
        [torch_lib_text, *(entry for entry in current if entry.casefold() != torch_lib_text.casefold())]
    )
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(torch_lib_text))


@dataclass(frozen=True)
class OCRLine:
    polygon: np.ndarray
    text: str
    confidence: float
    label: str = "text"
    style: dict[str, Any] | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        x_values = self.polygon[:, 0]
        y_values = self.polygon[:, 1]
        return (
            int(np.floor(x_values.min())),
            int(np.floor(y_values.min())),
            int(np.ceil(x_values.max())),
            int(np.ceil(y_values.max())),
        )


@dataclass(frozen=True)
class RegionAnalysis:
    mask: np.ndarray
    raw_mask: np.ndarray
    foreground_bgr: tuple[int, int, int]
    background_bgr: tuple[int, int, int]
    background_uniform: bool


def _package_major_version(package: str) -> int:
    try:
        return int(version(package).split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 0


def _result_mapping(value: Any) -> dict[str, Any]:
    if type(value) is dict:
        data = value
    else:
        data = {}
        json_value = getattr(value, "json", None)
        if callable(json_value):
            json_value = json_value()
        if isinstance(json_value, dict):
            data = json_value
        elif isinstance(value, dict):
            data = value
        if not data and hasattr(value, "__getitem__"):
            for key in (
                "res",
                "overall_ocr_res",
                "layout_det_res",
                "parsing_res_list",
                "model_settings",
                "spotting_res",
                "width",
                "height",
                "rec_texts",
                "rec_scores",
                "rec_polys",
                "rec_boxes",
                "boxes",
            ):
                try:
                    data[key] = value[key]
                except (KeyError, IndexError, TypeError):
                    continue
    nested = data.get("res")
    return nested if isinstance(nested, dict) else data


def _polygon_from_v3(value: Any) -> np.ndarray | None:
    points = np.asarray(value, dtype=np.float32)
    if points.shape == (4, 2):
        return points
    if points.shape == (4,):
        x1, y1, x2, y2 = points.tolist()
        return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return None


def _ocr_lines_from_v3_output(
    output: Any,
    *,
    language: str,
    min_confidence: float,
) -> list[OCRLine]:
    lines: list[OCRLine] = []
    for result in output or []:
        data = _result_mapping(result)
        overall = data.get("overall_ocr_res")
        if overall is not None:
            data = _result_mapping(overall)
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        polygons = data.get("rec_polys")
        if polygons is None:
            polygons = data.get("rec_boxes")
        if polygons is None:
            polygons = []
        for text, score, polygon in zip(texts, scores, polygons):
            normalized_text = str(text).strip()
            if language == "en":
                normalized_text = _repair_english_spacing(normalized_text)
            confidence = float(score)
            points = _polygon_from_v3(polygon)
            if not normalized_text or confidence < min_confidence or points is None:
                continue
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            lines.append(OCRLine(points, normalized_text, confidence))
    return _deduplicate_ocr_lines(lines)


def _layout_regions_from_v3_output(output: Any) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for result in output or []:
        data = _result_mapping(result)
        layout = _result_mapping(data.get("layout_det_res", {}))
        for box in layout.get("boxes") or []:
            if not isinstance(box, dict):
                continue
            coordinate = box.get("coordinate")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in coordinate)
            if x2 <= x1 or y2 <= y1:
                continue
            regions.append(
                {
                    "label": str(box.get("label", "unknown")).casefold(),
                    "score": round(float(box.get("score", 0.0)), 6),
                    "bbox": [
                        int(math.floor(x1)),
                        int(math.floor(y1)),
                        int(math.ceil(x2)),
                        int(math.ceil(y2)),
                    ],
                }
            )
    return regions


def _normalize_vlm_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|li|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>\n]+>", "", text)
    text = re.sub(
        r"\$(.+?)\$",
        lambda match: _plain_text_from_inline_latex(match.group(1)),
        text,
        flags=re.DOTALL,
    )
    text = text.replace("$", "")
    text = re.sub(r"(?<=\d)\s+(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^[-*+]\s+", "• ", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _plain_text_from_inline_latex(value: str) -> str:
    text = value
    text = re.sub(
        r"\\(?:mathrm|text|operatorname)\s*\{([^{}]*)\}",
        r"\1",
        text,
    )
    replacements = {
        r"\,": " ",
        r"\;": " ",
        r"\:": " ",
        r"\times": "×",
        r"\cdot": "·",
        r"\circ": "°",
        r"\%": "%",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\^\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\^\s*\(([^()]+)\)", r"\1", text)
    text = re.sub(r"_\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"_\s*\(([^()]+)\)", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _text_character_units(text: str) -> float:
    return sum(0.53 if ord(character) < 128 else 1.0 for character in text)


def _wrap_text_for_rows(text: str, row_count: int) -> str:
    normalized = " ".join(text.split())
    if row_count <= 1 or not normalized:
        return normalized

    words = normalized.split()
    if len(words) > 1:
        target_units = _text_character_units(normalized) / row_count
        rows: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if (
                current
                and len(rows) < row_count - 1
                and _text_character_units(candidate) > target_units
            ):
                rows.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            rows.append(" ".join(current))
        return "\n".join(rows)

    if row_count == 2:
        midpoint = len(normalized) / 2
        punctuation_breaks = [
            index + 1
            for index, character in enumerate(normalized)
            if character in "：:；;。！？!?"
        ]
        if punctuation_breaks:
            best_break = min(
                punctuation_breaks,
                key=lambda index: abs(index - midpoint),
            )
            if abs(best_break - midpoint) <= len(normalized) * 0.20:
                return f"{normalized[:best_break]}\n{normalized[best_break:]}"

    chunk_size = max(1, math.ceil(len(normalized) / row_count))
    return "\n".join(
        normalized[index : index + chunk_size]
        for index in range(0, len(normalized), chunk_size)
    )


def _polygon_from_vlm_block(block: dict[str, Any]) -> np.ndarray | None:
    polygon = block.get("block_polygon_points")
    if polygon is not None:
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 4 and points.shape[1] == 2:
            return points
    bbox = block.get("block_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def _vlm_blocks_from_output(
    output: Any,
) -> tuple[list[OCRLine], list[dict[str, Any]], list[dict[str, Any]]]:
    lines: list[OCRLine] = []
    layout_regions = _layout_regions_from_v3_output(output)
    blocks_for_audit: list[dict[str, Any]] = []

    for result in output or []:
        data = _result_mapping(result)
        for raw_block in data.get("parsing_res_list") or []:
            if not isinstance(raw_block, dict):
                continue
            label = str(raw_block.get("block_label", "unknown")).casefold()
            polygon = _polygon_from_vlm_block(raw_block)
            if polygon is None:
                continue
            x1, y1 = polygon.min(axis=0)
            x2, y2 = polygon.max(axis=0)
            bbox = (
                int(math.floor(float(x1))),
                int(math.floor(float(y1))),
                int(math.ceil(float(x2))),
                int(math.ceil(float(y2))),
            )
            matching_regions = [
                region
                for region in layout_regions
                if region["label"] == label
            ]
            if not matching_regions:
                matching_regions = layout_regions
            best_region = max(
                matching_regions,
                key=lambda region: _bbox_iou(bbox, tuple(region["bbox"])),
                default=None,
            )
            confidence = float(best_region["score"]) if best_region else 1.0
            content = _normalize_vlm_text(raw_block.get("block_content", ""))
            blocks_for_audit.append(
                {
                    "label": label,
                    "bbox": list(bbox),
                    "confidence": round(confidence, 6),
                    "content": content,
                }
            )
            if label not in VLM_TEXT_LAYOUT_LABELS or not content:
                continue
            lines.append(OCRLine(polygon, content, confidence, label))
    return _deduplicate_ocr_lines(lines), layout_regions, blocks_for_audit


def _responses_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"].strip()
    raise RuntimeError("OpenAI vision response did not contain structured output text")


def _scaled_bbox(
    values: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    bbox = _clip_bbox(
        (
            int(round(x1 * width / 1000.0)),
            int(round(y1 * height / 1000.0)),
            int(round(x2 * width / 1000.0)),
            int(round(y2 * height / 1000.0)),
        ),
        width,
        height,
    )
    if bbox[2] - bbox[0] < 3 or bbox[3] - bbox[1] < 3:
        return None
    return bbox


def _openai_slide_schema() -> dict[str, Any]:
    text_region = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "text",
            "confidence",
            "label",
            "bbox",
            "fontFamily",
            "fontWeight",
            "fontStyle",
            "color",
            "align",
            "editable",
            "uncertain",
        ],
        "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "label": {
                "type": "string",
                "enum": [
                    "doc_title",
                    "paragraph_title",
                    "subtitle",
                    "text",
                    "list",
                    "caption",
                    "footer",
                    "number",
                ],
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
            "fontFamily": {"type": "string"},
            "fontWeight": {"type": "string", "enum": ["regular", "bold"]},
            "fontStyle": {"type": "string", "enum": ["normal", "italic"]},
            "color": {"type": "string"},
            "align": {
                "type": "string",
                "enum": ["left", "center", "right", "justify"],
            },
            "editable": {"type": "boolean"},
            "uncertain": {"type": "boolean"},
        },
    }
    occlusion = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "confidence", "bbox"],
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["person", "lectern", "foreground_object", "unknown"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "bbox": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["regions", "occlusions", "background"],
        "properties": {
            "regions": {"type": "array", "items": text_region},
            "occlusions": {"type": "array", "items": occlusion},
            "background": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "nearlyUniform",
                    "hasLightingGradient",
                    "hasMoire",
                ],
                "properties": {
                    "nearlyUniform": {"type": "boolean"},
                    "hasLightingGradient": {"type": "boolean"},
                    "hasMoire": {"type": "boolean"},
                },
            },
        },
    }


class OpenAIVisionEngine:
    """High-detail slide understanding through the OpenAI Responses API."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_VISION_MODEL,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 180.0,
    ) -> None:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is required for --analysis-engine openai. "
                "Use --analysis-engine vlm to keep all processing local."
            )
        self.api_key = api_key
        self.model_name = model
        self.timeout_seconds = float(timeout_seconds)
        self.last_language = "multilingual"
        self.last_layout_regions: list[dict[str, Any]] = []
        self.last_blocks: list[dict[str, Any]] = []
        self.last_occlusions: list[dict[str, Any]] = []
        self.last_background_analysis: dict[str, Any] = {}

    def recognize(self, image: np.ndarray, min_confidence: float) -> list[OCRLine]:
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Could not encode slide image for OpenAI vision")
        image_url = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")
        prompt = (
            "Analyze this photographed presentation slide for faithful PPT reconstruction. "
            "Transcribe every visible text region exactly, preserving visible line breaks. "
            "Return tight bounding boxes around the visible glyphs using 0-1000 coordinates. "
            "Estimate font family, bold/italic style, color, and paragraph alignment from the image. "
            "Mark a region uncertain when small or blurred characters cannot be read reliably; do not "
            "invent text hidden by a person, lectern, or other foreground object. Identify only real "
            "foreground occlusions that are not authored slide content. A low-confidence region may "
            "contain a best reading, but editable must be false when confidence is below "
            f"{float(min_confidence):.2f}."
        )
        request_body = {
            "model": self.model_name,
            "reasoning": {"effort": "high"},
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "original",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "slide_reconstruction",
                    "strict": True,
                    "schema": _openai_slide_schema(),
                }
            },
        }
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        request = urllib.request.Request(
            f"{base_url}/responses",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "photo2slide/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI vision request failed with HTTP {exc.code}: {detail[:800]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI vision request failed: {exc.reason}") from exc

        analysis = json.loads(_responses_output_text(payload))
        height, width = image.shape[:2]
        lines: list[OCRLine] = []
        layout_regions: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        for region in analysis.get("regions") or []:
            if not isinstance(region, dict):
                continue
            text = _normalize_vlm_text(region.get("text", ""))
            bbox = _scaled_bbox(region.get("bbox"), width, height)
            if not text or bbox is None:
                continue
            confidence = float(np.clip(region.get("confidence", 0.0), 0.0, 1.0))
            label = str(region.get("label", "text")).casefold()
            if label not in VLM_TEXT_LAYOUT_LABELS:
                label = "text"
            x1, y1, x2, y2 = bbox
            polygon = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )
            style = {
                "fontFamily": str(region.get("fontFamily", "")).strip(),
                "bold": region.get("fontWeight") == "bold",
                "italic": region.get("fontStyle") == "italic",
                "color": str(region.get("color", "")).strip(),
                "align": str(region.get("align", "left")),
                "editable": bool(region.get("editable", True)),
                "uncertain": bool(region.get("uncertain", False)),
            }
            lines.append(OCRLine(polygon, text, confidence, label, style))
            layout_regions.append(
                {
                    "label": label,
                    "score": round(confidence, 6),
                    "bbox": list(bbox),
                }
            )
            blocks.append(
                {
                    "label": label,
                    "bbox": list(bbox),
                    "confidence": round(confidence, 6),
                    "content": text,
                    "style": style,
                }
            )

        occlusions: list[dict[str, Any]] = []
        for region in analysis.get("occlusions") or []:
            if not isinstance(region, dict):
                continue
            bbox = _scaled_bbox(region.get("bbox"), width, height)
            confidence = float(np.clip(region.get("confidence", 0.0), 0.0, 1.0))
            if bbox is None or confidence < 0.55:
                continue
            occlusions.append(
                {
                    "kind": str(region.get("kind", "unknown")),
                    "confidence": round(confidence, 6),
                    "bbox": list(bbox),
                }
            )
        self.last_layout_regions = layout_regions
        self.last_blocks = blocks
        self.last_occlusions = occlusions
        self.last_background_analysis = dict(analysis.get("background") or {})
        return _deduplicate_ocr_lines(lines)


class PaddleOCREngine:
    def __init__(self, language: str, device: str) -> None:
        self.language = language
        self.last_language = language
        self.api_major = _package_major_version("paddleocr")
        self.last_layout_regions: list[dict[str, Any]] = []
        _configure_windows_torch_dll_search_path()
        # PaddleOCR imports Albumentations, which may import torch after Paddle has
        # already loaded conflicting Windows DLLs. Loading torch first is stable.
        if os.name == "nt":
            import torch  # noqa: F401

        from paddleocr import PaddleOCR

        if self.api_major >= 3:
            options = {
                "lang": language,
                "ocr_version": "PP-OCRv5",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            }
            resolved_device = self._resolve_v3_device(device)
            if resolved_device is not None:
                options["device"] = resolved_device
            self.engine = PaddleOCR(**options)
        else:
            use_gpu = self._resolve_gpu(device)
            options = {
                "use_angle_cls": True,
                "lang": language,
                "use_gpu": use_gpu,
                "show_log": False,
                "det_limit_side_len": 2560,
                "det_limit_type": "max",
                "det_db_thresh": 0.20,
                "det_db_box_thresh": 0.45,
                "det_db_unclip_ratio": 1.65,
                "use_dilation": True,
                "rec_batch_num": 8,
            }
            try:
                self.engine = PaddleOCR(**options)
            except TypeError:
                # Keep compatibility with installations that expose a narrower API.
                for key in ("use_dilation", "rec_batch_num"):
                    options.pop(key, None)
                self.engine = PaddleOCR(**options)

    @staticmethod
    def _resolve_gpu(device: str) -> bool:
        if device == "cpu":
            return False
        try:
            import paddle

            available = bool(paddle.device.is_compiled_with_cuda())
        except Exception:
            available = False
        if device == "gpu" and not available:
            raise RuntimeError("PaddleOCR GPU was requested, but Paddle is a CPU build")
        return available

    @classmethod
    def _resolve_v3_device(cls, device: str) -> str | None:
        if device == "auto":
            return None
        if device == "cpu":
            return "cpu"
        cls._resolve_gpu("gpu")
        return "gpu:0"

    def recognize(self, image: np.ndarray, min_confidence: float) -> list[OCRLine]:
        if self.api_major >= 3:
            output = self.engine.predict(
                input=image,
                text_rec_score_thresh=min_confidence,
            )
            return _ocr_lines_from_v3_output(
                output,
                language=self.language,
                min_confidence=min_confidence,
            )

        raw = self.engine.ocr(image, cls=True)
        if not raw:
            return []
        records = raw[0] if len(raw) == 1 and isinstance(raw[0], list) else raw
        lines: list[OCRLine] = []
        for record in records or []:
            if not isinstance(record, (list, tuple)) or len(record) != 2:
                continue
            polygon, recognition = record
            if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
                continue
            text = str(recognition[0]).strip()
            if self.language == "en":
                text = _repair_english_spacing(text)
            confidence = float(recognition[1])
            points = np.asarray(polygon, dtype=np.float32)
            if not text or confidence < min_confidence or points.shape != (4, 2):
                continue
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            lines.append(OCRLine(points, text, confidence))
        return _deduplicate_ocr_lines(lines)


class PPStructureV3Engine:
    """PaddleOCR 3.x layout analysis plus OCR for slide reconstruction."""

    def __init__(self, language: str, device: str) -> None:
        if _package_major_version("paddleocr") < 3:
            raise RuntimeError(
                "PP-StructureV3 requires PaddleOCR 3.x; install p2p/requirements.txt"
            )
        self.language = language
        self.last_language = "ch+en" if language in {"auto", "ch"} else language
        self.last_layout_regions: list[dict[str, Any]] = []
        _configure_windows_torch_dll_search_path()
        if os.name == "nt":
            import torch  # noqa: F401

        from paddleocr import PPStructureV3

        options: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "use_seal_recognition": False,
            "use_table_recognition": False,
            "use_formula_recognition": False,
            "use_chart_recognition": False,
            "use_region_detection": False,
        }
        if language == "en":
            options["lang"] = "en"
        resolved_device = PaddleOCREngine._resolve_v3_device(device)
        if resolved_device is not None:
            options["device"] = resolved_device
        try:
            self.engine = PPStructureV3(**options)
        except RuntimeError as exc:
            if "dependency error occurred during pipeline creation" not in str(exc).casefold():
                raise
            requirements_path = PROJECT_ROOT / "requirements.txt"
            raise RuntimeError(
                "PP-StructureV3 dependencies are incomplete or incompatible. "
                f'Run: "{sys.executable}" -m pip install -r "{requirements_path}"'
            ) from exc

    def recognize(self, image: np.ndarray, min_confidence: float) -> list[OCRLine]:
        output = self.engine.predict(
            input=image,
            text_rec_score_thresh=min_confidence,
            use_table_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_region_detection=False,
        )
        self.last_layout_regions = _layout_regions_from_v3_output(output)
        output_language = "en" if self.language == "en" else "ch"
        return _ocr_lines_from_v3_output(
            output,
            language=output_language,
            min_confidence=min_confidence,
        )


class PaddleOCRVLEngine:
    """PaddleOCR-VL 1.6 page understanding for slide reconstruction."""

    @staticmethod
    def _resolve_transformers_device(device: str) -> str:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        if device == "gpu" and not gpu_available:
            raise RuntimeError(
                "PaddleOCR-VL GPU was requested, but PyTorch cannot access CUDA"
            )
        if device == "cpu" or not gpu_available:
            return "cpu"
        return "gpu:0"

    def __init__(self, device: str) -> None:
        if _package_major_version("paddleocr") < 3:
            raise RuntimeError(
                "PaddleOCR-VL requires PaddleOCR 3.x; install p2p/requirements.txt"
            )
        self.last_language = "multilingual"
        self.last_layout_regions: list[dict[str, Any]] = []
        self.last_blocks: list[dict[str, Any]] = []
        self.model_name = PADDLEOCR_VL_MODEL_NAME

        from paddleocr import PaddleOCRVL

        options: dict[str, Any] = {
            "pipeline_version": PADDLEOCR_VL_PIPELINE_VERSION,
            "engine": "transformers",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_layout_detection": True,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "use_ocr_for_image_block": False,
            "format_block_content": False,
            "merge_layout_blocks": False,
            "use_queues": False,
        }
        options["device"] = self._resolve_transformers_device(device)
        try:
            self.engine = PaddleOCRVL(**options)
        except RuntimeError as exc:
            requirements_path = PROJECT_ROOT / "requirements.txt"
            raise RuntimeError(
                "PaddleOCR-VL could not be initialized. "
                f'Run: "{sys.executable}" -m pip install -r "{requirements_path}"'
            ) from exc

    def recognize(self, image: np.ndarray, min_confidence: float) -> list[OCRLine]:
        del min_confidence
        output = self.engine.predict(
            input=image,
            use_layout_detection=True,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_ocr_for_image_block=False,
            layout_shape_mode="rect",
            format_block_content=False,
            merge_layout_blocks=False,
            use_queues=False,
        )
        lines, layout_regions, blocks = _vlm_blocks_from_output(output)
        self.last_layout_regions = layout_regions
        self.last_blocks = blocks
        return lines


class AutoOCREngine:
    def __init__(self, device: str) -> None:
        self.device = device
        self.primary = PaddleOCREngine("ch", device)
        self.english: PaddleOCREngine | None = None
        self.english_unavailable = False
        self.last_language = "ch"

    def recognize(self, image: np.ndarray, min_confidence: float) -> list[OCRLine]:
        primary_lines = self.primary.recognize(image, min_confidence)
        ascii_letters = sum(
            character.isascii() and character.isalpha()
            for line in primary_lines
            for character in line.text
        )
        cjk_letters = sum(
            "\u4e00" <= character <= "\u9fff"
            for line in primary_lines
            for character in line.text
        )
        english_dominant = (
            ascii_letters >= 30
            and ascii_letters / max(1, ascii_letters + cjk_letters) >= 0.82
        )
        if not english_dominant or self.english_unavailable:
            self.last_language = "ch"
            return primary_lines

        repaired_primary = [
            OCRLine(
                line.polygon,
                _repair_english_spacing(line.text),
                line.confidence,
                line.label,
            )
            for line in primary_lines
        ]

        if self.english is None:
            try:
                self.english = PaddleOCREngine("en", self.device)
            except Exception as exc:
                self.english_unavailable = True
                print(f"  [OCR] English model unavailable; using multilingual OCR: {exc}")
                self.last_language = "ch+spacing"
                return repaired_primary
        english_lines = self.english.recognize(image, min_confidence)
        primary_mean = float(np.mean([line.confidence for line in primary_lines])) if primary_lines else 0.0
        english_mean = float(np.mean([line.confidence for line in english_lines])) if english_lines else 0.0
        if (
            len(english_lines) >= max(3, int(round(len(primary_lines) * 0.62)))
            and english_mean >= primary_mean - 0.08
        ):
            merged: list[OCRLine] = []
            used_english: set[int] = set()
            for primary_line in repaired_primary:
                matches = [
                    (_bbox_iou(primary_line.bbox, english_line.bbox), index, english_line)
                    for index, english_line in enumerate(english_lines)
                    if index not in used_english
                ]
                overlap, index, english_line = max(matches, default=(0.0, -1, primary_line))
                if overlap >= 0.52:
                    used_english.add(index)
                    merged.append(
                        english_line
                        if english_line.confidence >= primary_line.confidence + 0.035
                        else primary_line
                    )
                else:
                    merged.append(primary_line)
            merged.extend(
                line for index, line in enumerate(english_lines) if index not in used_english
            )
            self.last_language = "ch+en"
            return _deduplicate_ocr_lines(merged)
        self.last_language = "ch+spacing"
        return repaired_primary


def _repair_english_spacing(text: str) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[,;:!?])(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[a-z])\.(?=[A-Z])", ". ", text)
    text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Z][a-z])", ". ", text)
    text = re.sub(r"(?<=[)）])(?=[A-Za-z])", " ", text)

    try:
        import wordninja
    except ImportError:
        wordninja = None
    function_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
    }

    def split_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if wordninja is None or len(token) < 12 or not token.islower():
            return token
        parts = wordninja.split(token)
        if (
            len(parts) >= 2
            and any(part.casefold() in function_words for part in parts)
            and all(len(part) >= 2 or part.casefold() in {"a", "i"} for part in parts)
        ):
            return " ".join(parts)
        return token

    text = re.sub(r"[A-Za-z]{12,}", split_token, text)
    return re.sub(r"\s+", " ", text).strip()


def _natural_key(path: Path) -> list[object]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(
        0.0, (bx2 - bx1) * (by2 - by1)
    ) - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_overlap_fraction(
    first: Iterable[float],
    second: Iterable[float],
) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    return intersection / max(1.0, (ax2 - ax1) * (ay2 - ay1))


def _deduplicate_ocr_lines(lines: list[OCRLine]) -> list[OCRLine]:
    accepted: list[OCRLine] = []
    for line in sorted(lines, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for current in accepted:
            ax1, ay1, ax2, ay2 = line.bbox
            bx1, by1, bx2, by2 = current.bbox
            intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
                0, min(ay2, by2) - max(ay1, by1)
            )
            smaller_area = max(
                1, min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
            )
            line_tokens = set(re.findall(r"[a-z0-9]+", line.text.casefold()))
            current_tokens = set(re.findall(r"[a-z0-9]+", current.text.casefold()))
            shared_tokens = len(line_tokens & current_tokens) / max(
                1, min(len(line_tokens), len(current_tokens))
            )
            nested_duplicate = intersection / smaller_area >= 0.72 and shared_tokens >= 0.65
            if (
                _bbox_iou(line.bbox, current.bbox) >= 0.68
                or nested_duplicate
            ) and (
                line.text.casefold() == current.text.casefold()
                or min(len(line.text), len(current.text)) >= 4
            ):
                duplicate = True
                break
        if not duplicate:
            accepted.append(line)
    return sorted(accepted, key=lambda item: (item.bbox[1], item.bbox[0]))


def _clip_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return tuple(
        int(value)
        for value in (
            max(0, min(width - 1, x1)),
            max(0, min(height - 1, y1)),
            max(1, min(width, x2)),
            max(1, min(height, y2)),
        )
    )


def _robust_spread(pixels: np.ndarray) -> float:
    if pixels.size == 0:
        return float("inf")
    median = np.median(pixels.astype(np.float32), axis=0)
    deviation = np.median(np.abs(pixels.astype(np.float32) - median), axis=0)
    return float(np.mean(deviation) * 1.4826)


def _analyze_text_region(image: np.ndarray, line: OCRLine) -> RegionAnalysis:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
    line_height = max(1, y2 - y1)
    padding = max(4, int(round(line_height * 0.28)))
    rx1, ry1, rx2, ry2 = _clip_bbox(
        (x1 - padding, y1 - padding, x2 + padding, y2 + padding), width, height
    )
    roi = image[ry1:ry2, rx1:rx2]
    local_polygon = np.rint(line.polygon - np.array([rx1, ry1])).astype(np.int32)
    polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [local_polygon], 255)

    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
    )
    dilated = cv2.dilate(polygon_mask, ring_kernel)
    ring_mask = cv2.subtract(dilated, polygon_mask)
    background_pixels = roi[ring_mask > 0]
    if len(background_pixels) < 16:
        background_pixels = roi[polygon_mask == 0]
    if len(background_pixels) < 16:
        background_pixels = roi.reshape(-1, 3)
    background = np.median(background_pixels, axis=0).astype(np.uint8)
    uniform = _robust_spread(background_pixels) < 8.75

    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_background = cv2.cvtColor(background.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(
        np.float32
    )[0, 0]
    distance = np.linalg.norm(lab_roi - lab_background, axis=2)
    lightness_delta = np.abs(lab_roi[:, :, 0] - lab_background[0])
    threshold = 15.0 if uniform else 24.0
    ink = (polygon_mask > 0) & (distance > threshold) & (lightness_delta > 5)

    polygon_distances = distance[polygon_mask > 0]
    polygon_area = max(1, int((polygon_mask > 0).sum()))
    if ink.sum() > polygon_area * 0.46 and polygon_distances.size:
        strict_threshold = max(threshold, float(np.percentile(polygon_distances, 84)))
        ink = (polygon_mask > 0) & (distance >= strict_threshold) & (lightness_delta > 6)
    if ink.sum() < max(8, int((polygon_mask > 0).sum() * 0.012)) and polygon_distances.size:
        fallback_threshold = max(10.0, float(np.percentile(polygon_distances, 78)))
        ink = (polygon_mask > 0) & (distance >= fallback_threshold)

    undilated_ink = ink.copy()
    ink_mask = (ink.astype(np.uint8) * 255)
    dilation_size = max(1, int(round(line_height * 0.095)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size * 2 + 1, dilation_size * 2 + 1)
    )
    ink_mask = cv2.dilate(ink_mask, kernel)
    ink_mask = cv2.bitwise_and(ink_mask, polygon_mask)

    foreground_selector = undilated_ink.copy()
    if foreground_selector.any():
        selected_distances = distance[foreground_selector]
        contrast_floor = float(np.percentile(selected_distances, 62))
        foreground_selector &= distance >= contrast_floor
    foreground_pixels = roi[foreground_selector]
    if len(foreground_pixels) < 4:
        foreground_pixels = roi[polygon_mask > 0]
    if len(foreground_pixels):
        foreground = np.median(foreground_pixels, axis=0).astype(np.uint8)
    else:
        foreground = np.array([0, 0, 0], dtype=np.uint8)

    global_mask = np.zeros((height, width), dtype=np.uint8)
    global_mask[ry1:ry2, rx1:rx2] = ink_mask
    global_raw_mask = np.zeros((height, width), dtype=np.uint8)
    global_raw_mask[ry1:ry2, rx1:rx2] = undilated_ink.astype(np.uint8) * 255
    return RegionAnalysis(
        mask=global_mask,
        raw_mask=global_raw_mask,
        foreground_bgr=tuple(int(value) for value in foreground),
        background_bgr=tuple(int(value) for value in background),
        background_uniform=uniform,
    )


def _hex_from_bgr(color: Iterable[int]) -> str:
    blue, green, red = [int(np.clip(value, 0, 255)) for value in color]
    return f"#{red:02X}{green:02X}{blue:02X}"


def _text_rotation(line: OCRLine) -> float:
    x1, y1, x2, y2 = line.bbox
    if y2 - y1 > (x2 - x1) * 1.65 and len(line.text.strip()) >= 4:
        return 270.0
    vector = line.polygon[1] - line.polygon[0]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    return 0.0 if abs(angle) < 1.25 else float(np.clip(angle, -15.0, 15.0))


def _replace_text_polygon_with_row_field(
    source: np.ndarray,
    target: np.ndarray,
    line: OCRLine,
    background_bgr: tuple[int, int, int] | None,
) -> bool:
    """Rebuild a text area from same-row neighbors, preserving banded gradients."""

    height, width = source.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
    line_height = max(1, y2 - y1)
    sample_width = max(16, min(64, int(round(line_height * 0.55))))
    vertical_margin = max(8, min(24, int(round(line_height * 0.16))))
    rx1, ry1, rx2, ry2 = _clip_bbox(
        (
            x1 - sample_width * 2,
            y1 - vertical_margin,
            x2 + sample_width * 2,
            y2 + vertical_margin,
        ),
        width,
        height,
    )
    roi = source[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return False
    local_polygon = np.rint(
        line.polygon - np.asarray([rx1, ry1], dtype=np.float32)
    ).astype(np.int32)
    polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [local_polygon], 255)
    replacement_margin = max(6, min(22, int(round(line_height * 0.14))))
    polygon_mask = cv2.dilate(
        polygon_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (replacement_margin * 2 + 1, replacement_margin * 2 + 1),
        ),
    )

    field = roi.astype(np.float32).copy()
    background = (
        np.asarray(background_bgr, dtype=np.float32)
        if background_bgr is not None
        else None
    )
    filled_rows = 0
    for row in range(roi.shape[0]):
        columns = np.flatnonzero(polygon_mask[row] > 0)
        if columns.size == 0:
            continue
        left_edge = int(columns[0])
        right_edge = int(columns[-1]) + 1
        left = roi[
            row,
            max(0, left_edge - sample_width * 2) : max(0, left_edge - 2),
        ]
        right = roi[
            row,
            min(roi.shape[1], right_edge + 2) : min(
                roi.shape[1],
                right_edge + sample_width * 2,
            ),
        ]

        def representative(pixels: np.ndarray) -> np.ndarray | None:
            if len(pixels) < 4:
                return None
            values = pixels.astype(np.float32)
            if background is not None:
                similar = (
                    np.linalg.norm(values - background[None, :], axis=1)
                    <= 52.0
                )
                if int(similar.sum()) >= 4:
                    values = values[similar]
            return np.median(values, axis=0) if len(values) >= 4 else None

        left_color = representative(left)
        right_color = representative(right)
        if left_color is None and right_color is None:
            if background is None:
                continue
            left_color = right_color = background
        elif left_color is None:
            left_color = right_color
        elif right_color is None:
            right_color = left_color
        assert left_color is not None and right_color is not None
        blend = np.linspace(
            0.0,
            1.0,
            roi.shape[1],
            dtype=np.float32,
        )[:, None]
        field[row] = left_color * (1.0 - blend) + right_color * blend
        filled_rows += 1
    if filled_rows < max(4, int(round((y2 - y1) * 0.70))):
        return False
    binary_alpha = polygon_mask.astype(np.float32) / 255.0
    feathered_alpha = cv2.GaussianBlur(
        binary_alpha,
        (0, 0),
        max(2.0, line_height * 0.045),
    )
    alpha = np.maximum(binary_alpha, feathered_alpha)[:, :, None]
    target_roi = target[ry1:ry2, rx1:rx2].astype(np.float32)
    target[ry1:ry2, rx1:rx2] = np.clip(
        target_roi * (1.0 - alpha) + field * alpha,
        0,
        255,
    ).astype(np.uint8)
    return True


def _replace_text_polygon_with_plane(
    source: np.ndarray,
    target: np.ndarray,
    line: OCRLine,
    *,
    background_uniform: bool = False,
    background_bgr: tuple[int, int, int] | None = None,
) -> bool:
    """Replace a text polygon with a smooth local background model."""

    height, width = source.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
    line_height = max(1, y2 - y1)
    padding = max(12, int(round(line_height * 0.72)))
    rx1, ry1, rx2, ry2 = _clip_bbox(
        (x1 - padding, y1 - padding, x2 + padding, y2 + padding), width, height
    )
    roi = source[ry1:ry2, rx1:rx2].astype(np.float32)
    if roi.size == 0:
        return False
    local_polygon = np.rint(line.polygon - np.array([rx1, ry1])).astype(np.int32)
    polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [local_polygon], 255)
    replacement_margin = max(6, min(22, int(round(line_height * 0.14))))
    polygon_mask = cv2.dilate(
        polygon_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (replacement_margin * 2 + 1, replacement_margin * 2 + 1),
        ),
    )
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
    )
    ring_mask = cv2.subtract(cv2.dilate(polygon_mask, ring_kernel), polygon_mask)
    sample_mask = ring_mask
    if background_bgr is not None:
        lab_roi = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        lab_background = cv2.cvtColor(
            np.asarray([[background_bgr]], dtype=np.uint8),
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)[0, 0]
        background_distance = np.linalg.norm(
            lab_roi - lab_background,
            axis=2,
        )
        distance_limit = 26.0 if background_uniform else 18.0
        sample_mask = (
            (polygon_mask == 0) & (background_distance <= distance_limit)
        ).astype(np.uint8) * 255
    ys, xs = np.nonzero(sample_mask)
    if len(xs) < 40:
        return False
    samples = roi[ys, xs]
    median = np.median(samples, axis=0)
    distances = np.linalg.norm(samples - median, axis=1)
    keep = distances <= max(18.0, float(np.percentile(distances, 72)))
    if int(keep.sum()) < 30:
        return False
    xs_fit = xs[keep].astype(np.float32) / max(1, roi.shape[1] - 1)
    ys_fit = ys[keep].astype(np.float32) / max(1, roi.shape[0] - 1)
    design = np.column_stack(
        (
            xs_fit,
            ys_fit,
            np.ones_like(xs_fit),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, samples[keep], rcond=None)
    all_x = xs.astype(np.float32) / max(1, roi.shape[1] - 1)
    all_y = ys.astype(np.float32) / max(1, roi.shape[0] - 1)
    ring_design = np.column_stack(
        (
            all_x,
            all_y,
            np.ones_like(all_x),
        )
    )
    ring_error = np.linalg.norm(ring_design @ coefficients - samples, axis=1)
    if not background_uniform:
        eval_y, eval_x = np.nonzero(ring_mask)
        eval_samples = roi[eval_y, eval_x]
        eval_design = np.column_stack(
            (
                eval_x.astype(np.float32) / max(1, roi.shape[1] - 1),
                eval_y.astype(np.float32) / max(1, roi.shape[0] - 1),
                np.ones_like(eval_x, dtype=np.float32),
            )
        )
        eval_error = np.linalg.norm(
            eval_design @ coefficients - eval_samples,
            axis=1,
        )
        if (
            float(np.median(eval_error)) > 8.5
            or float(np.mean(eval_error <= 18.0)) < 0.72
        ):
            return _replace_text_polygon_with_row_field(
                source,
                target,
                line,
                background_bgr,
            )

    grid_y, grid_x = np.indices(roi.shape[:2], dtype=np.float32)
    all_design = np.stack(
        (
            grid_x / max(1, roi.shape[1] - 1),
            grid_y / max(1, roi.shape[0] - 1),
            np.ones_like(grid_x),
        ),
        axis=-1,
    )
    plane = np.clip(all_design @ coefficients, 0, 255)
    try:
        cloned = cv2.seamlessClone(
            plane.astype(np.uint8),
            target,
            polygon_mask,
            ((rx1 + rx2) // 2, (ry1 + ry2) // 2),
            cv2.NORMAL_CLONE,
        )
        target[:] = cloned
    except cv2.error:
        binary_alpha = polygon_mask.astype(np.float32) / 255.0
        feathered_alpha = cv2.GaussianBlur(
            binary_alpha,
            (0, 0),
            max(1.5, line_height * 0.10),
        )
        alpha = np.maximum(binary_alpha, feathered_alpha)[:, :, None]
        target_roi = target[ry1:ry2, rx1:rx2].astype(np.float32)
        target[ry1:ry2, rx1:rx2] = np.clip(
            target_roi * (1.0 - alpha) + plane * alpha, 0, 255
        ).astype(np.uint8)
    return True


def _text_ink_mask(
    image: np.ndarray,
    analysis: RegionAnalysis,
    line: OCRLine,
    width: int,
    height: int,
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
    roi = image[y1:y2, x1:x2]
    result = np.zeros((height, width), dtype=np.uint8)
    if roi.size == 0:
        return result

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    box_height = max(1, y2 - y1)
    kernel_width = max(5, int(round(box_height * 0.12)))
    kernel_height = max(3, int(round(box_height * 0.07)))
    if kernel_width % 2 == 0:
        kernel_width += 1
    if kernel_height % 2 == 0:
        kernel_height += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, kernel_height),
    )
    foreground_luminance = float(
        cv2.cvtColor(
            np.asarray([[analysis.foreground_bgr]], dtype=np.uint8),
            cv2.COLOR_BGR2GRAY,
        )[0, 0]
    )
    background_luminance = float(
        cv2.cvtColor(
            np.asarray([[analysis.background_bgr]], dtype=np.uint8),
            cv2.COLOR_BGR2GRAY,
        )[0, 0]
    )
    if foreground_luminance <= background_luminance:
        estimated_background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        contrast = cv2.subtract(estimated_background, gray)
    else:
        estimated_background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        contrast = cv2.subtract(gray, estimated_background)

    if int(contrast.max()) < 4:
        return analysis.raw_mask.copy()
    otsu_threshold, _ = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    threshold = max(4.0, float(otsu_threshold) * 0.72)
    ink = (contrast >= threshold).astype(np.uint8) * 255
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
    )

    local_polygon = np.round(
        line.polygon - np.asarray([x1, y1], dtype=np.float32)
    ).astype(np.int32)
    polygon_mask = np.zeros_like(ink)
    cv2.fillPoly(polygon_mask, [local_polygon], 255)
    ink = cv2.bitwise_and(ink, polygon_mask)
    raw_local = cv2.bitwise_and(
        analysis.raw_mask[y1:y2, x1:x2],
        polygon_mask,
    )
    raw_density = cv2.countNonZero(raw_local) / max(1, raw_local.size)
    if raw_density <= 0.22:
        ink = cv2.bitwise_or(ink, raw_local)
    if cv2.countNonZero(ink) < 6:
        return analysis.raw_mask.copy()
    result[y1:y2, x1:x2] = ink
    return result


def _text_row_boxes(
    ink_mask: np.ndarray,
    line: OCRLine,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
    raw = ink_mask[y1:y2, x1:x2] > 0
    expanded = cv2.dilate(
        raw.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    if not raw.any():
        raw = expanded
    projection = raw.sum(axis=1)
    nonzero = projection[projection > 0]
    if nonzero.size == 0:
        return [(x1, y1, x2, y2)]
    threshold = max(1.0, min(float(np.percentile(nonzero, 22)) * 0.42, (x2 - x1) * 0.01))
    active = (projection >= threshold).astype(np.uint8).reshape(-1, 1)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)).reshape(-1)

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))

    minimum_height = max(2, int(round((y2 - y1) * 0.035)))
    runs = [run for run in runs if run[1] - run[0] >= minimum_height]
    merged: list[tuple[int, int]] = []
    maximum_gap = max(1, int(round((y2 - y1) * 0.012)))
    for run in runs:
        if merged and run[0] - merged[-1][1] <= maximum_gap:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    maximum_rows = 24 if len(line.text.strip()) >= 200 else 8
    if not merged or len(merged) > maximum_rows:
        return [(x1, y1, x2, y2)]

    boxes: list[tuple[int, int, int, int]] = []
    for top, bottom in merged:
        top = max(0, top - 1)
        bottom = min(y2 - y1, bottom + 1)
        row_pixels = expanded[top:bottom]
        columns = np.flatnonzero(row_pixels.any(axis=0))
        if columns.size == 0:
            continue
        left = max(0, int(columns[0]) - 1)
        right = min(x2 - x1, int(columns[-1]) + 2)
        boxes.append((x1 + left, y1 + top, x1 + right, y1 + bottom))
    return boxes or [(x1, y1, x2, y2)]


def _complete_long_text_row_boxes(
    line: OCRLine,
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    """Supply row geometry when capture noise merges a long paragraph into one blob."""

    if len(line.text.strip()) < 120:
        return boxes
    x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    estimated_rows = int(
        round(
            math.sqrt(
                _text_character_units(" ".join(line.text.split()))
                * 0.58
                * box_height
                / box_width
            )
        )
    )
    explicit_rows = len([row for row in line.text.splitlines() if row.strip()])
    estimated_rows = int(np.clip(max(estimated_rows, explicit_rows), 1, 24))
    if estimated_rows < 3 or len(boxes) >= max(2, math.ceil(estimated_rows * 0.55)):
        return boxes

    edges = np.linspace(y1, y2, estimated_rows + 1)
    gap = max(1, int(round(box_height / estimated_rows * 0.08)))
    return [
        (
            x1,
            int(round(edges[index])) + gap,
            x2,
            max(
                int(round(edges[index])) + gap + 1,
                int(round(edges[index + 1])) - gap,
            ),
        )
        for index in range(estimated_rows)
    ]


def _split_tokens_for_rows(text: str, row_widths: list[float]) -> list[str]:
    row_count = len(row_widths)
    explicit = [row.strip() for row in text.splitlines() if row.strip()]
    if len(explicit) == row_count:
        return explicit
    normalized = " ".join(text.split())
    if row_count <= 1 or not normalized:
        return [normalized]

    tokens = re.findall(r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+(?:\s+)?", normalized)
    expanded_tokens: list[str] = []
    for token in tokens:
        numeric_range = re.search(r"(\d+[-–—])(\d+)", token)
        if numeric_range:
            prefix = token[: numeric_range.start()]
            suffix = token[numeric_range.end() :]
            if prefix:
                expanded_tokens.append(prefix)
            expanded_tokens.extend(
                [numeric_range.group(1), numeric_range.group(2) + suffix]
            )
        else:
            expanded_tokens.append(token)
    tokens = expanded_tokens
    if len(tokens) < row_count:
        return _wrap_text_for_rows(normalized, row_count).splitlines()
    weights = np.asarray(
        [max(0.15, _text_character_units(token.rstrip())) for token in tokens],
        dtype=np.float32,
    )
    widths = np.maximum(np.asarray(row_widths, dtype=np.float32), 1.0)
    target_cumulative = np.cumsum(widths / widths.sum())[:-1] * float(weights.sum())
    cumulative = np.cumsum(weights)
    breaks: list[int] = []
    minimum_index = 1
    for target_index, target in enumerate(target_cumulative):
        maximum_index = len(tokens) - (row_count - target_index - 1)
        choices = np.arange(minimum_index, maximum_index + 1)
        best = int(choices[np.argmin(np.abs(cumulative[choices - 1] - target))])
        breaks.append(best)
        minimum_index = best + 1

    rows: list[str] = []
    start = 0
    for stop in [*breaks, len(tokens)]:
        rows.append("".join(tokens[start:stop]).strip())
        start = stop
    return rows


def _row_has_detached_bullet(
    raw_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = bbox
    crop = (raw_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if crop.size == 0 or not crop.any():
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    components = [
        tuple(int(value) for value in stats[index])
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 3
        and stats[index, cv2.CC_STAT_HEIGHT] >= 2
    ]
    if len(components) < 2:
        return False
    components.sort(key=lambda item: item[cv2.CC_STAT_LEFT])
    first, second = components[0], components[1]
    row_height = max(1, y2 - y1)
    first_right = first[cv2.CC_STAT_LEFT] + first[cv2.CC_STAT_WIDTH]
    gap = second[cv2.CC_STAT_LEFT] - first_right
    aspect = first[cv2.CC_STAT_WIDTH] / max(1, first[cv2.CC_STAT_HEIGHT])
    return bool(
        first[cv2.CC_STAT_LEFT] <= row_height * 0.30
        and first[cv2.CC_STAT_HEIGHT] <= row_height * 0.72
        and 0.30 <= aspect <= 2.6
        and gap >= row_height * 0.42
    )


def _infer_italic(
    raw_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str,
) -> bool:
    x1, y1, x2, y2 = bbox
    crop = (raw_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    if crop.size == 0 or _text_character_units(text) < 5.0:
        return False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    slopes: list[float] = []
    minimum_height = max(5, int(round((y2 - y1) * 0.28)))
    for component in range(1, count):
        if (
            stats[component, cv2.CC_STAT_AREA] < 6
            or stats[component, cv2.CC_STAT_HEIGHT] < minimum_height
        ):
            continue
        ys, xs = np.nonzero(labels == component)
        variance = float(np.var(ys))
        if variance < 1.0:
            continue
        slope = float(np.cov(ys, xs, bias=True)[0, 1] / variance)
        if abs(slope) <= 1.0:
            slopes.append(slope)
    if len(slopes) < 5:
        return False
    median_slope = float(np.median(slopes))
    negative_fraction = float(np.mean(np.asarray(slopes) < -0.045))
    return median_slope < -0.075 and negative_fraction >= 0.58


def _infer_bold(
    raw_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str,
) -> bool:
    x1, y1, x2, y2 = bbox
    crop = (raw_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    box_height = max(1, y2 - y1)
    if (
        crop.size == 0
        or box_height < 19
        or _text_character_units(text) < 4.5
    ):
        return False
    foreground = crop[crop > 0]
    if foreground.size == 0:
        return False
    distances = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    stroke_values = distances[distances > 0]
    if stroke_values.size < 12:
        return False
    density = float(np.mean(crop))
    stroke = float(np.percentile(stroke_values, 75))
    return density >= 0.355 and stroke >= 1.75


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= character <= "\u9fff" for character in text)


def _typeface_for_text(
    text: str,
    configured_font: str,
    style: dict[str, Any] | None,
) -> str:
    suggested = str((style or {}).get("fontFamily", "")).strip()
    if suggested and suggested.casefold() not in {
        "sans",
        "sans-serif",
        "serif",
        "unknown",
    }:
        return suggested
    if _contains_cjk(text):
        return configured_font
    if configured_font.casefold() == "microsoft yahei":
        return "Arial"
    return configured_font


@lru_cache(maxsize=64)
def _font_file(typeface: str, bold: bool, italic: bool) -> str | None:
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    family = typeface.casefold().replace(" ", "")
    known = {
        "microsoftyahei": {
            (False, False): "msyh.ttc",
            (True, False): "msyhbd.ttc",
            (False, True): "msyh.ttc",
            (True, True): "msyhbd.ttc",
        },
        "arial": {
            (False, False): "arial.ttf",
            (True, False): "arialbd.ttf",
            (False, True): "ariali.ttf",
            (True, True): "arialbi.ttf",
        },
        "calibri": {
            (False, False): "calibri.ttf",
            (True, False): "calibrib.ttf",
            (False, True): "calibrii.ttf",
            (True, True): "calibriz.ttf",
        },
        "simhei": {
            (False, False): "simhei.ttf",
            (True, False): "simhei.ttf",
            (False, True): "simhei.ttf",
            (True, True): "simhei.ttf",
        },
        "simsun": {
            (False, False): "simsun.ttc",
            (True, False): "simsunb.ttf",
            (False, True): "simsun.ttc",
            (True, True): "simsunb.ttf",
        },
    }
    filename = known.get(family, {}).get((bold, italic))
    if filename and (fonts_dir / filename).is_file():
        return str(fonts_dir / filename)
    if fonts_dir.is_dir():
        compact = re.sub(r"[^a-z0-9]", "", family)
        for candidate in fonts_dir.iterdir():
            if candidate.is_file() and re.sub(
                r"[^a-z0-9]",
                "",
                candidate.stem.casefold(),
            ).startswith(compact):
                return str(candidate)
    return None


def _font_size_for_box(
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    typeface: str,
    bold: bool,
    italic: bool,
) -> float:
    box_width = max(2.0, float(bbox[2] - bbox[0]))
    box_height = max(2.0, float(bbox[3] - bbox[1]))
    path = _font_file(typeface, bold, italic)
    if path:
        try:
            from PIL import ImageFont

            reference_size = 256
            font = ImageFont.truetype(path, reference_size)
            left, top, right, bottom = font.getbbox(text or "M")
            measured_width = max(1.0, float(right - left))
            measured_height = max(1.0, float(bottom - top))
            by_height = box_height * reference_size / measured_height
            by_width = box_width * reference_size / measured_width
            return float(np.clip(min(by_height, by_width) * 0.985, 8.0, 96.0))
        except (ImportError, OSError, ValueError):
            pass
    by_height = box_height * (1.10 if _contains_cjk(text) else 1.28)
    by_width = box_width / max(1.0, _text_character_units(text) * 0.58)
    return float(np.clip(min(by_height, by_width), 8.0, 96.0))


def _row_foreground_color(
    image: np.ndarray,
    raw_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    fallback: tuple[int, int, int],
) -> tuple[int, int, int]:
    x1, y1, x2, y2 = bbox
    selector = raw_mask[y1:y2, x1:x2] > 0
    pixels = image[y1:y2, x1:x2][selector]
    if len(pixels) < 4:
        return fallback
    luminance = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2GRAY,
    ).reshape(-1)
    fallback_luminance = float(
        cv2.cvtColor(
            np.asarray([[fallback]], dtype=np.uint8),
            cv2.COLOR_BGR2GRAY,
        )[0, 0]
    )
    if fallback_luminance <= 128:
        selected = pixels[luminance <= np.percentile(luminance, 35)]
    else:
        selected = pixels[luminance >= np.percentile(luminance, 65)]
    if len(selected) < 4:
        selected = pixels
    return tuple(int(value) for value in np.median(selected, axis=0))


def _valid_hex_color(value: Any) -> str | None:
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text.upper()
    return None


def _readable_text_color(
    color: str,
    background_bgr: tuple[int, int, int],
) -> str:
    """Keep inferred text colors legible when capture noise masks the real ink."""

    parsed = _valid_hex_color(color) or "#202020"

    def relative_luminance(red: int, green: int, blue: int) -> float:
        channels = []
        for value in (red, green, blue):
            normalized = value / 255.0
            channels.append(
                normalized / 12.92
                if normalized <= 0.04045
                else ((normalized + 0.055) / 1.055) ** 2.4
            )
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    foreground = tuple(int(parsed[index : index + 2], 16) for index in (1, 3, 5))
    blue, green, red = [int(value) for value in background_bgr]
    foreground_luminance = relative_luminance(*foreground)
    background_luminance = relative_luminance(red, green, blue)
    contrast = (max(foreground_luminance, background_luminance) + 0.05) / (
        min(foreground_luminance, background_luminance) + 0.05
    )
    if contrast >= 3.0:
        return parsed

    dark = "#20242B"
    light = "#F7F9FC"
    dark_contrast = (background_luminance + 0.05) / (
        relative_luminance(0x20, 0x24, 0x2B) + 0.05
    )
    light_contrast = (relative_luminance(0xF7, 0xF9, 0xFC) + 0.05) / (
        background_luminance + 0.05
    )
    return dark if dark_contrast >= light_contrast else light


def _clean_text_and_build_specs(
    image: np.ndarray,
    lines: list[OCRLine],
    font_name: str,
    *,
    analysis_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    height, width = image.shape[:2]
    source = analysis_image if analysis_image is not None else image
    if source.shape != image.shape:
        raise ValueError("analysis_image must match image dimensions")
    safe_x = max(8.0, width * 0.012)
    safe_y = max(6.0, height * 0.012)
    analyses = [_analyze_text_region(source, line) for line in lines]
    cleaned = image.copy()
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for analysis in analyses:
        full_mask = cv2.bitwise_or(full_mask, analysis.mask)
    for line, analysis in zip(lines, analyses):
        if _replace_text_polygon_with_plane(
            image,
            cleaned,
            line,
            background_uniform=analysis.background_uniform,
            background_bgr=analysis.background_bgr,
        ):
            continue
        inpaint_mask = cv2.bitwise_or(inpaint_mask, analysis.mask)
    if inpaint_mask.any():
        radius = max(
            3,
            int(
                round(
                    np.median(
                        [max(1, line.bbox[3] - line.bbox[1]) for line in lines]
                    )
                    * 0.16
                )
            ),
        )
        cleaned = cv2.inpaint(cleaned, inpaint_mask, radius, cv2.INPAINT_TELEA)

    row_records: list[
        tuple[
            OCRLine,
            RegionAnalysis,
            np.ndarray,
            tuple[int, int, int, int],
            str,
            bool,
            bool,
            bool,
        ]
    ] = []
    for line, analysis in zip(lines, analyses):
        ink_mask = _text_ink_mask(source, analysis, line, width, height)
        boxes = _text_row_boxes(ink_mask, line, width, height)
        if (
            len(boxes) > 1
            and len([row for row in line.text.splitlines() if row.strip()]) <= 1
        ):
            line_x1, line_y1, line_x2, line_y2 = _clip_bbox(
                line.bbox,
                width,
                height,
            )
            estimated_rows = int(
                round(
                    math.sqrt(
                        _text_character_units(" ".join(line.text.split()))
                        * 0.58
                        * max(1, line_y2 - line_y1)
                        / max(1, line_x2 - line_x1)
                    )
                )
            )
            if estimated_rows <= 1:
                boxes = [max(boxes, key=lambda box: box[2] - box[0])]
        boxes = _complete_long_text_row_boxes(
            line,
            boxes,
            width,
            height,
        )
        row_texts = _split_tokens_for_rows(
            line.text,
            [max(1.0, float(box[2] - box[0])) for box in boxes],
        )
        if len(row_texts) != len(boxes):
            boxes = [line.bbox]
            row_texts = [line.text]
        italic_votes = [
            _infer_italic(ink_mask, box, row_text)
            for box, row_text in zip(boxes, row_texts)
        ]
        block_italic = sum(italic_votes) >= max(1, math.ceil(len(boxes) * 0.5))
        bold_votes = [
            _infer_bold(ink_mask, box, row_text)
            for box, row_text in zip(boxes, row_texts)
        ]
        block_bold = sum(bold_votes) >= max(1, math.ceil(len(boxes) * 0.5))
        if line.label == "text" and len(line.text.strip()) >= 200:
            block_italic = False
            block_bold = False
        for box, row_text in zip(boxes, row_texts):
            detached_bullet = (
                False
                if line.label == "text" and len(line.text.strip()) >= 200
                else _row_has_detached_bullet(ink_mask, box)
            )
            has_bullet = bool(
                detached_bullet
                or re.match(r"^[▪■□•·●○\-*]", row_text)
            )
            if detached_bullet and not re.match(r"^[▪■□•·●○\-*]", row_text):
                row_text = f"• {row_text}"
            row_records.append(
                (
                    line,
                    analysis,
                    ink_mask,
                    box,
                    row_text,
                    has_bullet,
                    block_italic,
                    block_bold,
                )
            )

    row_heights = [
        max(1, box[3] - box[1])
        for _, _, _, box, _, _, _, _ in row_records
    ]
    median_height = float(np.median(row_heights)) if row_heights else 1.0
    anchor_tolerance = width * 0.025
    left_anchor_counts = {
        id(line): sum(
            abs(other.bbox[0] - line.bbox[0]) <= anchor_tolerance
            for other in lines
        )
        for line in lines
    }
    specs: list[dict[str, Any]] = []
    spec_lines: list[OCRLine] = []
    for index, (
        line,
        analysis,
        ink_mask,
        row_box,
        display_text,
        has_bullet,
        inferred_italic,
        inferred_bold,
    ) in enumerate(
        row_records,
        start=1,
    ):
        x1, y1, x2, y2 = _clip_bbox(row_box, width, height)
        box_height = max(1, y2 - y1)
        box_width = max(1, x2 - x1)
        line_x1, line_y1, line_x2, line_y2 = _clip_bbox(
            line.bbox,
            width,
            height,
        )
        is_title = (
            line.label in {"doc_title", "paragraph_title"}
            or (
                line_y1 < height * 0.26
                and line_x2 - line_x1 > width * 0.22
                and box_height >= median_height * 1.22
            )
        )
        style = line.style or {}
        bold = bool(
            style.get(
                "bold",
                is_title
                or inferred_bold,
            )
        )
        italic = bool(style.get("italic", inferred_italic))
        typeface = _typeface_for_text(display_text, font_name, style)
        font_size_px = _font_size_for_box(
            display_text,
            row_box,
            typeface=typeface,
            bold=bold,
            italic=italic,
        )
        alignment_hint = str(style.get("align", "")).strip().casefold()
        centered = not has_bullet and (
            alignment_hint == "center"
            or (
                not alignment_hint
                and left_anchor_counts.get(id(line), 1) < 3
                and abs(((line_x1 + line_x2) * 0.5) - width * 0.5)
                < width * 0.075
                and line_x1 > width * 0.06
                and line_x2 < width * 0.94
            )
        )
        pad_x = max(1.0, box_height * 0.045)
        pad_y = max(1.0, box_height * 0.10)
        text_x = max(safe_x, x1 - pad_x)
        text_y = max(safe_y, y1 - pad_y)
        text_width = min(float(width) - safe_x - text_x, box_width + pad_x * 2.0)
        text_height = min(
            float(height) - safe_y - text_y,
            box_height + pad_y * 2.0,
        )
        local_color = _hex_from_bgr(
            _row_foreground_color(
                source,
                ink_mask,
                row_box,
                analysis.foreground_bgr,
            )
        )
        source_color = _valid_hex_color(style.get("color")) or local_color
        source_color = _readable_text_color(source_color, analysis.background_bgr)
        align = str(style.get("align", "center" if centered else "left"))
        if align not in {"left", "center", "right", "justify"}:
            align = "center" if centered else "left"
        if has_bullet and "align" not in style:
            align = "left"
        specs.append(
            {
                "name": f"text-{index:03d}",
                "text": display_text,
                "x": round(text_x, 2),
                "y": round(text_y, 2),
                "w": round(max(2.0, text_width), 2),
                "h": round(max(2.0, text_height), 2),
                "fontSize": round(font_size_px, 2),
                "fontSizePt": round(font_size_px * 72.0 / 96.0, 2),
                "color": source_color,
                "bold": bold,
                "italic": italic,
                "align": align,
                "valign": "middle",
                "typeface": typeface,
                "rotation": round(_text_rotation(line), 2),
                "confidence": round(line.confidence, 4),
                "analysisLabel": line.label,
                "needsReview": bool(
                    style.get("uncertain", False)
                    or style.get("editable") is False
                ),
            }
        )
        spec_lines.append(
            OCRLine(
                np.asarray(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                ),
                display_text,
                line.confidence,
                line.label,
                line.style,
            )
        )
    _constrain_same_row_text_specs(spec_lines, specs, width, height, 1.0)
    return cleaned, full_mask, specs


def _constrain_same_row_text_specs(
    lines: list[OCRLine],
    specs: list[dict[str, Any]],
    width: int,
    height: int,
    target_scale: float,
) -> None:
    del height, target_scale
    for index, (line, spec) in enumerate(zip(lines, specs)):
        x1, y1, x2, y2 = line.bbox
        line_height = max(1, y2 - y1)
        peers: list[tuple[OCRLine, dict[str, Any]]] = []
        for other_index, (other_line, other_spec) in enumerate(zip(lines, specs)):
            if other_index == index:
                continue
            ox1, oy1, ox2, oy2 = other_line.bbox
            vertical_overlap = max(0, min(y2, oy2) - max(y1, oy1))
            if vertical_overlap / max(1, min(y2 - y1, oy2 - oy1)) < 0.55:
                continue
            if x2 <= ox1 or ox2 <= x1:
                peers.append((other_line, other_spec))
        if not peers:
            continue

        left_boundary = 0.0
        right_boundary = float(width)
        for other_line, _ in peers:
            ox1, _, ox2, _ = other_line.bbox
            if ox2 <= x1:
                left_boundary = max(left_boundary, (ox2 + x1) * 0.5)
            elif x2 <= ox1:
                right_boundary = min(right_boundary, (x2 + ox1) * 0.5)

        original_left = max(0.0, x1 - line_height * 0.08)
        text_x = max(float(spec["x"]), left_boundary, original_left)
        available_width = max(2.0, right_boundary - text_x)
        original_width = max(2.0, float(spec["w"]))
        spec["x"] = round(text_x, 2)
        spec["w"] = round(min(original_width, available_width), 2)
        spec["align"] = "left"

        if float(spec["w"]) < original_width:
            scale = float(spec["w"]) / original_width
            spec["fontSize"] = round(max(8.0, float(spec["fontSize"]) * scale), 2)
            spec["fontSizePt"] = round(
                max(6.0, float(spec["fontSizePt"]) * scale),
                2,
            )


def _ring_pixels(
    image: np.ndarray, bbox: tuple[int, int, int, int], thickness: int
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
    outer = _clip_bbox(
        (x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness),
        width,
        height,
    )
    ox1, oy1, ox2, oy2 = outer
    roi = image[oy1:oy2, ox1:ox2]
    mask = np.ones(roi.shape[:2], dtype=bool)
    mask[max(0, y1 - oy1) : max(0, y2 - oy1), max(0, x1 - ox1) : max(0, x2 - ox1)] = False
    return roi[mask]


def _detect_native_rectangles(image: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = image.shape[:2]
    slide_area = float(width * height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 155)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approximation = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        x, y, box_width, box_height = cv2.boundingRect(approximation)
        area = float(box_width * box_height)
        if (
            box_width < 18
            or box_height < 7
            or area < slide_area * 0.00045
            or area > slide_area * 0.20
        ):
            continue
        rectangularity = abs(cv2.contourArea(approximation)) / max(area, 1.0)
        if rectangularity < 0.87:
            continue
        points = approximation.reshape(4, 2)
        if np.max(np.minimum(abs(points[:, 0] - x), abs(points[:, 0] - (x + box_width)))) > max(4, box_width * 0.04):
            continue
        inset = max(2, min(box_width, box_height) // 10)
        inner = image[y + inset : y + box_height - inset, x + inset : x + box_width - inset]
        if inner.size == 0 or _robust_spread(inner.reshape(-1, 3)) > 10.5:
            continue
        ring = _ring_pixels(image, (x, y, x + box_width, y + box_height), max(3, inset))
        if len(ring) < 20 or _robust_spread(ring) > 17.0:
            continue
        fill = np.median(inner.reshape(-1, 3), axis=0).astype(np.uint8)
        background = np.median(ring, axis=0).astype(np.uint8)
        fill_lab = cv2.cvtColor(fill.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(background.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
        if float(np.linalg.norm(fill_lab - bg_lab)) < 15.0:
            continue
        border_pixels = image[y : y + box_height, x : x + box_width]
        border_mask = np.zeros((box_height, box_width), dtype=bool)
        border_size = max(1, min(box_width, box_height) // 18)
        border_mask[:border_size, :] = True
        border_mask[-border_size:, :] = True
        border_mask[:, :border_size] = True
        border_mask[:, -border_size:] = True
        border = np.median(border_pixels[border_mask], axis=0).astype(np.uint8)
        candidates.append(
            {
                "bbox": (x, y, x + box_width, y + box_height),
                "area": area,
                "fill_bgr": fill,
                "border_bgr": border,
                "background_bgr": background,
            }
        )

    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
        if any(_bbox_iou(candidate["bbox"], other["bbox"]) > 0.84 for other in kept):
            continue
        kept.append(candidate)
        if len(kept) >= MAX_NATIVE_SHAPES:
            break

    cleaned = image.copy()
    specs: list[dict[str, Any]] = []
    for index, candidate in enumerate(sorted(kept, key=lambda item: item["area"], reverse=True), start=1):
        x1, y1, x2, y2 = candidate["bbox"]
        cleaned[y1:y2, x1:x2] = candidate["background_bgr"]
        fill = candidate["fill_bgr"]
        border = candidate["border_bgr"]
        border_contrast = float(np.linalg.norm(border.astype(np.float32) - fill.astype(np.float32)))
        specs.append(
            {
                "name": f"shape-{index:03d}",
                "kind": "rect",
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "fill": _hex_from_bgr(fill),
                "line": _hex_from_bgr(border) if border_contrast > 14 else "#00000000",
                "lineWidth": 1.0 if border_contrast > 14 else 0.0,
            }
        )
    return cleaned, specs


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    horizontal_gap = max(8, width // 120)
    vertical_gap = max(8, height // 90)
    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        output: list[tuple[int, int, int, int]] = []
        while merged:
            current = merged.pop(0)
            cx1, cy1, cx2, cy2 = current
            found = None
            for index, other in enumerate(merged):
                ox1, oy1, ox2, oy2 = other
                overlap_x = min(cx2, ox2) - max(cx1, ox1)
                overlap_y = min(cy2, oy2) - max(cy1, oy1)
                gap_x = max(0, max(cx1, ox1) - min(cx2, ox2))
                gap_y = max(0, max(cy1, oy1) - min(cy2, oy2))
                should_merge = (
                    _bbox_iou(current, other) > 0
                    or (overlap_y > 0 and gap_x <= horizontal_gap)
                    or (overlap_x > 0 and gap_y <= vertical_gap)
                )
                if not should_merge:
                    continue
                union = (min(cx1, ox1), min(cy1, oy1), max(cx2, ox2), max(cy2, oy2))
                union_area = (union[2] - union[0]) * (union[3] - union[1])
                if union_area <= width * height * 0.42:
                    found = index
                    current = union
                    break
            if found is not None:
                merged.pop(found)
                merged.insert(0, current)
                changed = True
            else:
                output.append(current)
        merged = output
    return merged


def _border_background(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    band = max(4, min(width, height) // 80)
    pixels = np.concatenate(
        [
            image[:band, :].reshape(-1, 3),
            image[-band:, :].reshape(-1, 3),
            image[:, :band].reshape(-1, 3),
            image[:, -band:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(pixels, axis=0).astype(np.uint8), _robust_spread(pixels)


def _complex_visual_regions(
    lines: list[OCRLine], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """Find chart/diagram regions that are safer to preserve as one visual layer.

    OCR text in plots is usually made of scattered short labels, numbers and
    vertical axis captions. Rebuilding those labels independently causes more
    damage than keeping the chart selectable as one image object.
    """

    candidates: list[OCRLine] = []
    for line in lines:
        x1, y1, x2, y2 = line.bbox
        center_y = (y1 + y2) * 0.5
        if height * 0.13 <= center_y <= height * 0.94:
            candidates.append(line)
    if len(candidates) < 8:
        return []

    short_fraction = float(
        np.mean([len(line.text.strip()) <= 16 for line in candidates])
    )
    numeric_fraction = float(
        np.mean(
            [
                sum(character.isdigit() for character in line.text)
                >= max(1, int(sum(character.isalnum() for character in line.text) * 0.62))
                for line in candidates
            ]
        )
    )
    vertical_fraction = float(
        np.mean(
            [
                line.bbox[3] - line.bbox[1]
                > (line.bbox[2] - line.bbox[0]) * 1.5
                for line in candidates
            ]
        )
    )
    centers_x = np.array(
        [(line.bbox[0] + line.bbox[2]) * 0.5 for line in candidates], dtype=np.float32
    )
    horizontal_scatter = float(np.std(centers_x) / max(1, width))
    ascii_letters = sum(
        character.isascii() and character.isalpha()
        for line in candidates
        for character in line.text
    )
    cjk_letters = sum(
        "\u4e00" <= character <= "\u9fff"
        for line in candidates
        for character in line.text
    )
    dense_english = (
        len(candidates) >= 18
        and ascii_letters / max(1, ascii_letters + cjk_letters) >= 0.92
        and float(np.median([len(line.text) for line in candidates])) >= 28.0
    )
    chart_like = (
        (short_fraction >= 0.52 and horizontal_scatter >= 0.14)
        or numeric_fraction >= 0.24
        or vertical_fraction >= 0.08
        or dense_english
    )
    if not chart_like:
        return []

    x1 = min(line.bbox[0] for line in candidates)
    y1 = min(line.bbox[1] for line in candidates)
    x2 = max(line.bbox[2] for line in candidates)
    y2 = max(line.bbox[3] for line in candidates)
    pad_x = max(10, int(round(width * 0.018)))
    pad_y = max(8, int(round(height * 0.020)))
    box = _clip_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    box_area = box_width * box_height
    if (
        box_width < width * 0.38
        or box_height < height * 0.25
        or box_area < width * height * 0.13
        or box_area > width * height * 0.88
    ):
        return []
    return [box]


def _extract_visual_assets(
    image: np.ndarray,
    asset_dir: Path,
    *,
    crop_source: np.ndarray | None = None,
    forced_boxes: list[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = image.shape[:2]
    crop_source = image if crop_source is None else crop_source
    forced_boxes = list(forced_boxes or [])
    background, spread = _border_background(image)
    boxes: list[tuple[int, int, int, int]] = []
    slide_area = width * height
    if spread <= 21.0:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(background.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        distance = np.linalg.norm(lab - bg_lab, axis=2)
        foreground = (distance > 23.0).astype(np.uint8) * 255
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(5, width // 180), max(5, height // 130))
            ),
        )
        foreground = cv2.dilate(foreground, np.ones((5, 5), np.uint8))
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        for component in range(1, component_count):
            x, y, box_width, box_height, pixel_area = stats[component]
            bbox_area = box_width * box_height
            if (
                pixel_area < slide_area * 0.0018
                or bbox_area < slide_area * 0.004
                or bbox_area > slide_area * 0.42
                or box_width < width * 0.045
                or box_height < height * 0.045
                or box_width > width * 0.92
                or box_height > height * 0.92
            ):
                continue
            pad = max(4, min(width, height) // 180)
            boxes.append(
                _clip_bbox(
                    (x - pad, y - pad, x + box_width + pad, y + box_height + pad),
                    width,
                    height,
                )
            )

    boxes = _merge_boxes(boxes, width, height)
    safe_boxes: list[tuple[int, int, int, int]] = list(forced_boxes)
    for box in boxes:
        if any(_bbox_iou(box, forced) > 0.02 for forced in forced_boxes):
            continue
        ring = _ring_pixels(image, box, max(5, min(width, height) // 140))
        if len(ring) < 30 or _robust_spread(ring) > 19.0:
            continue
        safe_boxes.append(box)
    safe_boxes = sorted(
        safe_boxes,
        key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        reverse=True,
    )[:MAX_VISUAL_ASSETS]
    safe_boxes.sort(key=lambda box: (box[1], box[0]))

    asset_dir.mkdir(parents=True, exist_ok=True)
    cleaned = image.copy()
    specs: list[dict[str, Any]] = []
    for index, (x1, y1, x2, y2) in enumerate(safe_boxes, start=1):
        crop = crop_source[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        asset_path = asset_dir / f"visual-{index:03d}.png"
        if not cv2.imwrite(str(asset_path), crop):
            raise OSError(f"Could not write visual asset: {asset_path}")
        ring = _ring_pixels(image, (x1, y1, x2, y2), max(5, min(width, height) // 140))
        local_background = np.median(ring, axis=0).astype(np.uint8) if len(ring) else background
        cleaned[y1:y2, x1:x2] = local_background
        specs.append(
            {
                "name": f"visual-{index:03d}",
                "file": str(asset_path.resolve()),
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
            }
        )
    return cleaned, specs


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def _slide_debug_overlay(image: np.ndarray, lines: list[OCRLine]) -> np.ndarray:
    overlay = image.copy()
    for line in lines:
        cv2.polylines(overlay, [np.rint(line.polygon).astype(np.int32)], True, (0, 0, 255), 2)
    return overlay


def _line_is_covered_by_asset(line: OCRLine, asset: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = line.bbox
    ax1 = float(asset["x"])
    ay1 = float(asset["y"])
    ax2 = ax1 + float(asset["w"])
    ay2 = ay1 + float(asset["h"])
    intersection_width = max(0.0, min(x2, ax2) - max(x1, ax1))
    intersection_height = max(0.0, min(y2, ay2) - max(y1, ay1))
    line_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    return intersection_width * intersection_height / line_area >= 0.58


def _text_spec_is_covered_by_asset(
    spec: dict[str, Any],
    asset: dict[str, Any],
) -> bool:
    x1 = float(spec["x"])
    y1 = float(spec["y"])
    x2 = x1 + float(spec["w"])
    y2 = y1 + float(spec["h"])
    ax1 = float(asset["x"])
    ay1 = float(asset["y"])
    ax2 = ax1 + float(asset["w"])
    ay2 = ay1 + float(asset["h"])
    intersection_width = max(0.0, min(x2, ax2) - max(x1, ax1))
    intersection_height = max(0.0, min(y2, ay2) - max(y1, ay1))
    area = max(1.0, (x2 - x1) * (y2 - y1))
    return intersection_width * intersection_height / area >= 0.58


def _dense_review_text_ids(
    review_lines: list[OCRLine],
    width: int,
    height: int,
) -> set[int]:
    """Promote coherent long-form OCR when image patches would fragment a page."""

    candidates = [
        line
        for line in review_lines
        if line.label == "text"
        and line.confidence >= 0.50
        and len(line.text.strip()) >= 200
    ]
    if len(candidates) < 2:
        return set()

    coverage = np.zeros((height, width), dtype=np.uint8)
    for line in candidates:
        x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
        coverage[y1:y2, x1:x2] = 1
    if float(np.mean(coverage)) < 0.20:
        return set()
    return {id(line) for line in candidates}


def _dense_text_panel_bbox(
    lines: list[OCRLine],
    promoted_line_ids: set[int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not promoted_line_ids:
        return None
    promoted = [line for line in lines if id(line) in promoted_line_ids]
    anchor_x1 = min(line.bbox[0] for line in promoted)
    anchor_x2 = max(line.bbox[2] for line in promoted)
    panel_lines = []
    for line in lines:
        if line.label not in {"text", "paragraph_title"}:
            continue
        x1, _, x2, _ = line.bbox
        overlap = max(0, min(anchor_x2, x2) - max(anchor_x1, x1))
        if overlap / max(1, x2 - x1) >= 0.60:
            panel_lines.append(line)
    if not panel_lines:
        return None
    pad_x = max(8, int(round(width * 0.008)))
    pad_y = max(6, int(round(height * 0.010)))
    return _clip_bbox(
        (
            min(line.bbox[0] for line in panel_lines) - pad_x,
            min(line.bbox[1] for line in panel_lines) - pad_y,
            max(line.bbox[2] for line in panel_lines) + pad_x,
            max(line.bbox[3] for line in panel_lines) + pad_y,
        ),
        width,
        height,
    )


def _reconstruct_slide(
    image_path: Path,
    slide_number: int,
    slide_dir: Path,
    mode: str,
    ocr_engine: (
        PaddleOCREngine
        | AutoOCREngine
        | PPStructureV3Engine
        | PaddleOCRVLEngine
        | OpenAIVisionEngine
        | None
    ),
    min_text_confidence: float,
    font_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read slide image: {image_path}")
    text_analysis_image = image.copy()
    height, width = image.shape[:2]
    slide_dir.mkdir(parents=True, exist_ok=True)

    if mode == "image":
        slide_spec = {
            "number": slide_number,
            "sourceWidth": width,
            "sourceHeight": height,
            "background": str(image_path.resolve()),
            "images": [],
            "shapes": [],
            "texts": [],
        }
        return slide_spec, {"slide": slide_number, "ocrTextCount": 0, "shapeCount": 0, "assetCount": 0}

    if ocr_engine is None:
        raise RuntimeError("OCR engine is required for editable modes")
    ocr_input = (
        image
        if isinstance(
            ocr_engine,
            (PPStructureV3Engine, PaddleOCRVLEngine, OpenAIVisionEngine),
        )
        else prepare_ocr_image(image).image
    )
    lines = ocr_engine.recognize(ocr_input, min_text_confidence)
    layout_regions = list(getattr(ocr_engine, "last_layout_regions", []))
    image = descreen_presentation_image(
        image,
        excluded_boxes=[
            *(line.bbox for line in lines),
            *(
                _clip_bbox(region["bbox"], width, height)
                for region in layout_regions
                if float(region.get("score", 0.0)) >= 0.30
            ),
            *(
                _clip_bbox(tuple(region["bbox"]), width, height)
                for region in list(getattr(ocr_engine, "last_occlusions", []))
            ),
        ],
    )
    model_forced_occlusions = [
        _clip_bbox(tuple(region["bbox"]), width, height)
        for region in list(getattr(ocr_engine, "last_occlusions", []))
        if float(region.get("confidence", 0.0)) >= 0.55
    ]
    edge_capture_occlusions = []
    for region in layout_regions:
        if region.get("label") != "image":
            continue
        score = float(region.get("score", 0.0))
        bbox = _clip_bbox(region["bbox"], width, height)
        x1, _, x2, y2 = bbox
        weak_edge_image = (
            0.20 <= score < 0.40
            and (x1 <= 1 or x2 >= width - 1)
            and x2 - x1 <= width * 0.14
            and y2 >= height * 0.65
        )
        if weak_edge_image:
            horizontal_pad = int(round(width * 0.06))
            vertical_pad = int(round(height * 0.06))
            if x1 <= 1:
                bbox = (
                    0,
                    max(0, bbox[1] - vertical_pad),
                    min(width, bbox[2] + horizontal_pad),
                    min(height, bbox[3] + vertical_pad),
                )
            else:
                bbox = (
                    max(0, bbox[0] - horizontal_pad),
                    max(0, bbox[1] - vertical_pad),
                    width,
                    min(height, bbox[3] + vertical_pad),
                )
            edge_capture_occlusions.append(bbox)
    merged_edge_occlusions: list[tuple[int, int, int, int]] = []
    for bbox in sorted(edge_capture_occlusions, key=lambda item: (item[0], item[1])):
        if merged_edge_occlusions:
            previous = merged_edge_occlusions[-1]
            same_edge = (
                previous[0] == bbox[0] == 0
                or previous[2] == bbox[2] == width
            )
            if same_edge and bbox[1] <= previous[3] + height * 0.12:
                merged_edge_occlusions[-1] = (
                    min(previous[0], bbox[0]),
                    min(previous[1], bbox[1]),
                    max(previous[2], bbox[2]),
                    max(previous[3], bbox[3]),
                )
                continue
        merged_edge_occlusions.append(bbox)
    edge_capture_occlusions = merged_edge_occlusions
    forced_occlusions = list(
        dict.fromkeys([*model_forced_occlusions, *edge_capture_occlusions])
    )
    protected_boxes = [line.bbox for line in lines]
    for region in layout_regions:
        score = float(region.get("score", 0.0))
        if region.get("label") not in VISUAL_LAYOUT_LABELS or score < 0.40:
            continue
        bbox = _clip_bbox(region["bbox"], width, height)
        x1, y1, x2, y2 = bbox
        overlaps_forced = any(
            _bbox_overlap_fraction(forced_box, bbox) >= 0.30
            for forced_box in forced_occlusions
        )
        if not overlaps_forced:
            protected_boxes.append(bbox)
    occlusion_repair = repair_border_occlusions(
        image,
        protected_boxes=protected_boxes,
        forced_boxes=model_forced_occlusions,
        solid_forced_boxes=edge_capture_occlusions,
    )
    working_image = occlusion_repair.image

    editable_lines = [
        line
        for line in lines
        if line.confidence >= min_text_confidence
        and not bool((line.style or {}).get("uncertain", False))
        and (line.style or {}).get("editable") is not False
    ]
    editable_line_ids = {id(line) for line in editable_lines}
    initial_review_lines = [line for line in lines if id(line) not in editable_line_ids]
    promoted_line_ids = _dense_review_text_ids(
        initial_review_lines,
        width,
        height,
    )
    if promoted_line_ids:
        editable_lines = [
            line
            for line in lines
            if id(line) in editable_line_ids or id(line) in promoted_line_ids
        ]
        editable_line_ids = {id(line) for line in editable_lines}
    review_lines = [line for line in lines if id(line) not in editable_line_ids]
    dense_panel_bbox = _dense_text_panel_bbox(
        lines,
        promoted_line_ids,
        width,
        height,
    )
    cleaning_lines = lines if mode == "editable" else editable_lines
    cleaned, text_mask, text_specs = _clean_text_and_build_specs(
        working_image,
        cleaning_lines,
        font_name,
        analysis_image=text_analysis_image,
    )
    if mode == "editable":
        text_specs = [
            spec
            for spec in text_specs
            if (
                float(spec.get("confidence", 0.0)) >= min_text_confidence
                and not bool(spec.get("needsReview", False))
            )
            or (
                dense_panel_bbox is not None
                and spec.get("analysisLabel") == "text"
                and float(spec.get("confidence", 0.0)) >= 0.50
            )
        ]
    shape_specs: list[dict[str, Any]] = []
    image_specs: list[dict[str, Any]] = []
    background = cleaned
    if mode == "editable":
        without_shapes, shape_specs = _detect_native_rectangles(cleaned)
        structure_boxes = [
            _clip_bbox(region["bbox"], width, height)
            for region in layout_regions
            if region.get("label") in VISUAL_LAYOUT_LABELS
            and float(region.get("score", 0.0)) >= 0.30
        ]
        complex_regions = _merge_boxes(
            [*_complex_visual_regions(lines, width, height), *structure_boxes],
            width,
            height,
        )
        background, image_specs = _extract_visual_assets(
            without_shapes,
            slide_dir / "assets",
            crop_source=working_image,
            forced_boxes=[*complex_regions, *(line.bbox for line in review_lines)],
        )
        background = normalize_nearly_solid_background(background)
        if dense_panel_bbox is not None:
            panel_x1, panel_y1, panel_x2, panel_y2 = dense_panel_bbox
            panel_fill = np.median(background.reshape(-1, 3), axis=0).astype(np.uint8)
            shape_specs.append(
                {
                    "name": f"quality-panel-{slide_number:02d}",
                    "kind": "rect",
                    "x": panel_x1,
                    "y": panel_y1,
                    "w": panel_x2 - panel_x1,
                    "h": panel_y2 - panel_y1,
                    "fill": _hex_from_bgr(panel_fill),
                    "line": "#00000000",
                    "lineWidth": 0.0,
                }
            )
        if image_specs:
            text_specs = [
                spec
                for spec in text_specs
                if not any(
                    _text_spec_is_covered_by_asset(spec, asset)
                    for asset in image_specs
                )
            ]

    background_path = slide_dir / "background.png"
    mask_path = slide_dir / "text-mask.png"
    occlusion_mask_path = slide_dir / "occlusion-mask.png"
    repaired_path = slide_dir / "capture-restored.png"
    overlay_path = slide_dir / "ocr-overlay.png"
    _write_image(background_path, background)
    _write_image(mask_path, text_mask)
    _write_image(occlusion_mask_path, occlusion_repair.mask)
    _write_image(repaired_path, working_image)
    _write_image(overlay_path, _slide_debug_overlay(image, lines))

    ocr_json = {
        "slide": slide_number,
        "source": str(image_path.resolve()),
        "layoutRegions": layout_regions,
        "visionModel": getattr(ocr_engine, "model_name", None),
        "visionBlocks": list(getattr(ocr_engine, "last_blocks", [])),
        "backgroundAnalysis": dict(
            getattr(ocr_engine, "last_background_analysis", {})
        ),
        "occlusions": [
            {
                "bbox": list(bbox),
                "source": "model"
                if any(
                    _bbox_overlap_fraction(bbox, forced_box) >= 0.30
                    for forced_box in model_forced_occlusions
                )
                else "low-confidence-edge-visual"
                if any(
                    _bbox_overlap_fraction(bbox, forced_box) >= 0.30
                    for forced_box in edge_capture_occlusions
                )
                else "conservative-border-detector",
            }
            for bbox in occlusion_repair.regions
        ],
        "lines": [
            {
                "text": line.text,
                "confidence": line.confidence,
                "label": line.label,
                "polygon": line.polygon.tolist(),
                "bbox": list(line.bbox),
                "style": line.style,
                "editable": id(line) in editable_line_ids,
                "needsReview": id(line) not in editable_line_ids,
            }
            for line in lines
        ],
    }
    (slide_dir / "ocr.json").write_text(
        json.dumps(ocr_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    slide_spec = {
        "number": slide_number,
        "sourceWidth": width,
        "sourceHeight": height,
        "background": str(background_path.resolve()),
        "images": image_specs,
        "shapes": shape_specs,
        "texts": text_specs,
    }
    summary = {
        "slide": slide_number,
        "source": str(image_path.resolve()),
        "ocrTextCount": len(text_specs),
        "shapeCount": len(shape_specs),
        "assetCount": len(image_specs),
        "ocrLanguage": getattr(ocr_engine, "last_language", None),
        "analysisEngine": type(ocr_engine).__name__,
        "visionModel": getattr(ocr_engine, "model_name", None),
        "layoutRegionCount": len(layout_regions),
        "needsReviewTextCount": len(review_lines),
        "promotedReviewTextCount": len(promoted_line_ids),
        "renderStrategy": "dense-text-rebuild"
        if dense_panel_bbox is not None
        else "editable",
        "occlusionRepairCount": len(occlusion_repair.regions),
        "meanTextConfidence": round(float(np.mean([line.confidence for line in lines])), 4)
        if lines
        else None,
    }
    return slide_spec, summary


def _find_node() -> str:
    executable = shutil.which("node")
    if executable:
        return executable
    bundled = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
        / "bin"
        / "node.exe"
    )
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError("Node.js 18+ is required to write the editable PPTX")


def _run_pptx_builder(
    manifest_path: Path,
    output_path: Path,
    work_dir: Path,
) -> None:
    builder = Path(__file__).with_name("pptx_from_manifest.mjs")
    if not builder.is_file():
        raise FileNotFoundError(f"PPTX builder is missing: {builder}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.pptx")
    command = [
        _find_node(),
        str(builder),
        "--manifest",
        str(manifest_path),
        "--out",
        str(temporary),
        "--workspace",
        str(work_dir / "artifact-workspace"),
        "--preview-dir",
        str(work_dir / "preview"),
        "--layout-dir",
        str(work_dir / "layout"),
        "--backend",
        "auto",
    ]
    builder_environment = os.environ.copy()
    result = subprocess.run(
        command,
        cwd=str(builder.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=builder_environment,
        check=False,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown builder error"
        raise RuntimeError(f"PPTX builder failed: {detail}")
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"PPTX builder did not create a valid file: {temporary}")
    inspect_sidecar = Path(f"{temporary}.inspect.ndjson")
    if inspect_sidecar.is_file():
        inspect_target = work_dir / "artifact-workspace" / "pptx-inspect.ndjson"
        inspect_target.parent.mkdir(parents=True, exist_ok=True)
        inspect_sidecar.replace(inspect_target)
    temporary.replace(output_path)


def build_editable_pptx(
    image_paths: list[Path],
    output_path: Path,
    *,
    mode: str = "editable",
    analysis_engine: str = "vlm",
    ocr_language: str = "auto",
    ocr_device: str = "auto",
    min_text_confidence: float = 0.70,
    font_name: str = "Microsoft YaHei",
    vision_model: str = DEFAULT_OPENAI_VISION_MODEL,
    work_dir: Path | None = None,
) -> Path:
    if mode not in {"editable", "hybrid", "image"}:
        raise ValueError(f"Unsupported PPT mode: {mode}")
    if analysis_engine not in {"vlm", "openai", "ocr", "structure"}:
        raise ValueError(f"Unsupported analysis engine: {analysis_engine}")
    if not image_paths:
        raise ValueError("No slide images were provided")

    image_paths = [Path(path).expanduser().resolve() for path in image_paths]
    output_path = Path(output_path).expanduser().resolve()
    reconstruction_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir
        else DEFAULT_RECONSTRUCTION_DIR
    )
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    if mode == "image":
        ocr_engine: (
            PaddleOCREngine
            | AutoOCREngine
            | PPStructureV3Engine
            | PaddleOCRVLEngine
            | OpenAIVisionEngine
            | None
        ) = None
    elif analysis_engine == "vlm":
        ocr_engine = PaddleOCRVLEngine(ocr_device)
    elif analysis_engine == "openai":
        ocr_engine = OpenAIVisionEngine(vision_model)
    elif analysis_engine == "structure":
        ocr_engine = PPStructureV3Engine(ocr_language, ocr_device)
    elif ocr_language == "auto":
        ocr_engine = AutoOCREngine(ocr_device)
    else:
        ocr_engine = PaddleOCREngine(ocr_language, ocr_device)

    slides: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for slide_number, image_path in enumerate(sorted(image_paths, key=_natural_key), start=1):
        print(f"  [PPT {slide_number}/{len(image_paths)}] reconstructing {image_path.name}")
        slide_spec, summary = _reconstruct_slide(
            Path(image_path),
            slide_number,
            reconstruction_dir / f"slide-{slide_number:02d}",
            mode,
            ocr_engine,
            min_text_confidence,
            font_name,
        )
        slides.append(slide_spec)
        summaries.append(summary)

    manifest = {
        "schemaVersion": 1,
        "mode": mode,
        "analysisEngine": analysis_engine,
        "visionModel": getattr(ocr_engine, "model_name", None),
        "slideSize": {"width": SLIDE_SIZE[0], "height": SLIDE_SIZE[1]},
        "slides": slides,
    }
    manifest_path = reconstruction_dir / "deck-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reconstruction_dir / "reconstruction-summary.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "analysisEngine": analysis_engine,
                "visionModel": getattr(ocr_engine, "model_name", None),
                "slides": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _run_pptx_builder(manifest_path, output_path, reconstruction_dir)
    return output_path
