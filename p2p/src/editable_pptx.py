from __future__ import annotations

import html
import json
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from image_enhancement import prepare_ocr_image


SLIDE_SIZE = (1280, 720)
MAX_NATIVE_SHAPES = 80
MAX_VISUAL_ASSETS = 24
PADDLEOCR_VL_PIPELINE_VERSION = "v1.6"
PADDLEOCR_VL_MODEL_NAME = "PaddleOCR-VL-1.6-0.9B"
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
                "PP-StructureV3 requires PaddleOCR 3.x; install p2p/requirements-vision.txt"
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
            requirements_path = Path(__file__).resolve().parents[1] / "requirements-vision.txt"
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
                "PaddleOCR-VL requires PaddleOCR 3.x; install p2p/requirements-vision.txt"
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
            requirements_path = Path(__file__).resolve().parents[1] / "requirements-vision.txt"
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
    uniform = _robust_spread(background_pixels) < 21.0

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
    dilation_size = max(1, int(round(line_height * 0.065)))
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
    return RegionAnalysis(
        mask=global_mask,
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


def _replace_text_polygon_with_plane(
    source: np.ndarray,
    target: np.ndarray,
    line: OCRLine,
) -> bool:
    """Replace a text polygon with a smooth local background model."""

    height, width = source.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
    line_height = max(1, y2 - y1)
    padding = max(8, int(round(line_height * 0.38)))
    rx1, ry1, rx2, ry2 = _clip_bbox(
        (x1 - padding, y1 - padding, x2 + padding, y2 + padding), width, height
    )
    roi = source[ry1:ry2, rx1:rx2].astype(np.float32)
    if roi.size == 0:
        return False
    local_polygon = np.rint(line.polygon - np.array([rx1, ry1])).astype(np.int32)
    polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [local_polygon], 255)
    polygon_mask = cv2.dilate(polygon_mask, np.ones((3, 3), np.uint8))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1)
    )
    ring_mask = cv2.subtract(cv2.dilate(polygon_mask, ring_kernel), polygon_mask)
    ys, xs = np.nonzero(ring_mask)
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
            xs_fit * xs_fit,
            ys_fit * ys_fit,
            xs_fit * ys_fit,
            np.ones_like(xs_fit),
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, samples[keep], rcond=None)

    grid_y, grid_x = np.indices(roi.shape[:2], dtype=np.float32)
    all_design = np.stack(
        (
            grid_x / max(1, roi.shape[1] - 1),
            grid_y / max(1, roi.shape[0] - 1),
            (grid_x / max(1, roi.shape[1] - 1)) ** 2,
            (grid_y / max(1, roi.shape[0] - 1)) ** 2,
            (grid_x / max(1, roi.shape[1] - 1))
            * (grid_y / max(1, roi.shape[0] - 1)),
            np.ones_like(grid_x),
        ),
        axis=-1,
    )
    plane = np.clip(all_design @ coefficients, 0, 255)
    alpha = cv2.GaussianBlur(
        polygon_mask.astype(np.float32) / 255.0,
        (0, 0),
        max(1.5, line_height * 0.10),
    )
    alpha = alpha[:, :, None]
    target_roi = target[ry1:ry2, rx1:rx2].astype(np.float32)
    target[ry1:ry2, rx1:rx2] = np.clip(
        target_roi * (1.0 - alpha) + plane * alpha, 0, 255
    ).astype(np.uint8)
    return True


def _clean_text_and_build_specs(
    image: np.ndarray,
    lines: list[OCRLine],
    font_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    height, width = image.shape[:2]
    safe_x = max(8.0, width * 0.012)
    safe_y = max(6.0, height * 0.012)
    analyses = [_analyze_text_region(image, line) for line in lines]
    cleaned = image.copy()
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for analysis in analyses:
        full_mask = cv2.bitwise_or(full_mask, analysis.mask)
    for line, analysis in zip(lines, analyses):
        if analysis.background_uniform and _replace_text_polygon_with_plane(image, cleaned, line):
            continue
        inpaint_mask = cv2.bitwise_or(inpaint_mask, analysis.mask)
    if inpaint_mask.any():
        radius = max(3, int(round(np.median([max(1, line.bbox[3] - line.bbox[1]) for line in lines]) * 0.11)))
        cleaned = cv2.inpaint(cleaned, inpaint_mask, radius, cv2.INPAINT_TELEA)

    target_scale = SLIDE_SIZE[1] / height
    line_heights = [max(1, line.bbox[3] - line.bbox[1]) for line in lines]
    median_height = float(np.median(line_heights)) if line_heights else 1.0
    specs: list[dict[str, Any]] = []
    for index, (line, analysis) in enumerate(zip(lines, analyses), start=1):
        x1, y1, x2, y2 = _clip_bbox(line.bbox, width, height)
        box_height = max(1, y2 - y1)
        box_width = max(1, x2 - x1)
        is_title = (
            line.label in {"doc_title", "paragraph_title"}
            or (
                y1 < height * 0.26
                and box_width > width * 0.22
                and box_height >= median_height * 1.22
            )
        )
        display_text = line.text
        text_rows = [row for row in display_text.splitlines() if row.strip()] or [display_text]
        if len(text_rows) == 1:
            estimated_line_height = min(box_height, median_height * 1.15)
            estimated_natural_width = (
                _text_character_units(display_text)
                * estimated_line_height
                * (0.95 if is_title else 0.72)
            )
            width_rows = max(
                1,
                math.ceil(estimated_natural_width / max(1.0, box_width * 1.03)),
            )
            height_rows = max(
                1,
                round(box_height / max(1.0, estimated_line_height * 0.88)),
            )
            inferred_rows = min(8, width_rows, height_rows)
            display_text = _wrap_text_for_rows(display_text, inferred_rows)
            text_rows = [
                row for row in display_text.splitlines() if row.strip()
            ] or [display_text]
        row_count = len(text_rows)
        height_font_size = float(
            np.clip(box_height * target_scale * 0.70 / row_count, 6.0, 52.0)
        )
        centered = (
            abs(((x1 + x2) * 0.5) - width * 0.5) < width * 0.075
            and x1 > width * 0.06
            and x2 < width * 0.94
        )
        extra_width = max(box_height * 0.45, box_width * 0.10)
        character_units = max(
            _text_character_units(row)
            for row in text_rows
        )
        if centered:
            center = (x1 + x2) * 0.5
            available_source_width = max(
                2.0,
                min(center - safe_x, width - safe_x - center) * 2.0,
            )
        else:
            source_x = max(safe_x, x1 - box_height * 0.08)
            available_source_width = max(2.0, width - safe_x - source_x)
        width_font_size = (
            available_source_width * target_scale
            / max(1.0, character_units * (96.0 / 72.0))
            * 0.97
        )
        font_size = float(np.clip(min(height_font_size, width_font_size), 6.0, 52.0))
        estimated_source_width = (
            character_units
            * font_size
            * (96.0 / 72.0)
            / max(target_scale, 1e-6)
            * 1.04
        )
        desired_width = max(box_width + extra_width, estimated_source_width)
        if centered:
            center = (x1 + x2) * 0.5
            text_x = max(safe_x, center - desired_width * 0.5)
            text_width = min(float(width) - safe_x - text_x, desired_width)
        else:
            text_x = max(safe_x, x1 - box_height * 0.08)
            text_width = min(float(width) - safe_x - text_x, desired_width)
        text_y = max(safe_y, y1 - box_height * 0.18)
        text_height = min(float(height) - safe_y - text_y, box_height * 1.46)
        specs.append(
            {
                "name": f"text-{index:03d}",
                "text": display_text,
                "x": round(text_x, 2),
                "y": round(text_y, 2),
                "w": round(max(2.0, text_width), 2),
                "h": round(max(2.0, text_height), 2),
                "fontSize": round(font_size, 2),
                "fontSizePt": round(font_size, 2),
                "color": _hex_from_bgr(analysis.foreground_bgr),
                "bold": bool(
                    is_title
                    or box_height >= median_height * 1.45
                    or (centered and y1 < height * 0.38 and box_height >= median_height * 1.06)
                ),
                "align": "center" if centered else "left",
                "valign": "middle",
                "typeface": font_name,
                "rotation": round(_text_rotation(line), 2),
                "confidence": round(line.confidence, 4),
                "analysisLabel": line.label,
            }
        )
    _constrain_same_row_text_specs(lines, specs, width, height, target_scale)
    return cleaned, full_mask, specs


def _constrain_same_row_text_specs(
    lines: list[OCRLine],
    specs: list[dict[str, Any]],
    width: int,
    height: int,
    target_scale: float,
) -> None:
    del height
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
        spec["x"] = round(text_x, 2)
        spec["w"] = round(min(float(spec["w"]), available_width), 2)
        spec["align"] = "left"

        character_units = max(
            (
                _text_character_units(row)
                for row in str(spec["text"]).splitlines()
                if row.strip()
            ),
            default=1.0,
        )
        width_font_size = (
            float(spec["w"])
            * target_scale
            / max(1.0, character_units * (96.0 / 72.0))
            * 0.96
        )
        font_size = float(np.clip(min(float(spec["fontSize"]), width_font_size), 6.0, 52.0))
        spec["fontSize"] = round(font_size, 2)
        spec["fontSizePt"] = round(font_size, 2)


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
        | None
    ),
    min_text_confidence: float,
    font_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read slide image: {image_path}")
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
        if isinstance(ocr_engine, (PPStructureV3Engine, PaddleOCRVLEngine))
        else prepare_ocr_image(image).image
    )
    lines = ocr_engine.recognize(ocr_input, min_text_confidence)
    layout_regions = list(getattr(ocr_engine, "last_layout_regions", []))
    cleaned, text_mask, text_specs = _clean_text_and_build_specs(image, lines, font_name)
    shape_specs: list[dict[str, Any]] = []
    image_specs: list[dict[str, Any]] = []
    background = cleaned
    if mode == "editable":
        without_shapes, shape_specs = _detect_native_rectangles(cleaned)
        structure_boxes = [
            _clip_bbox(region["bbox"], width, height)
            for region in layout_regions
            if region.get("label") in VISUAL_LAYOUT_LABELS
            and float(region.get("score", 0.0)) >= 0.45
        ]
        complex_regions = _merge_boxes(
            [*_complex_visual_regions(lines, width, height), *structure_boxes],
            width,
            height,
        )
        background, image_specs = _extract_visual_assets(
            without_shapes,
            slide_dir / "assets",
            crop_source=image,
            forced_boxes=complex_regions,
        )
        if image_specs:
            text_specs = [
                spec
                for line, spec in zip(lines, text_specs)
                if not any(_line_is_covered_by_asset(line, asset) for asset in image_specs)
            ]

    background_path = slide_dir / "background.png"
    mask_path = slide_dir / "text-mask.png"
    overlay_path = slide_dir / "ocr-overlay.png"
    _write_image(background_path, background)
    _write_image(mask_path, text_mask)
    _write_image(overlay_path, _slide_debug_overlay(image, lines))

    ocr_json = {
        "slide": slide_number,
        "source": str(image_path.resolve()),
        "layoutRegions": layout_regions,
        "visionModel": getattr(ocr_engine, "model_name", None),
        "visionBlocks": list(getattr(ocr_engine, "last_blocks", [])),
        "lines": [
            {
                "text": line.text,
                "confidence": line.confidence,
                "label": line.label,
                "polygon": line.polygon.tolist(),
                "bbox": list(line.bbox),
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
    temporary.replace(output_path)


def build_editable_pptx(
    image_paths: list[Path],
    output_path: Path,
    *,
    mode: str = "editable",
    analysis_engine: str = "vlm",
    ocr_language: str = "auto",
    ocr_device: str = "auto",
    min_text_confidence: float = 0.78,
    font_name: str = "Microsoft YaHei",
    work_dir: Path | None = None,
) -> Path:
    if mode not in {"editable", "hybrid", "image"}:
        raise ValueError(f"Unsupported PPT mode: {mode}")
    if analysis_engine not in {"vlm", "ocr", "structure"}:
        raise ValueError(f"Unsupported analysis engine: {analysis_engine}")
    if not image_paths:
        raise ValueError("No slide images were provided")

    image_paths = [Path(path).expanduser().resolve() for path in image_paths]
    output_path = Path(output_path).expanduser().resolve()
    reconstruction_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir
        else output_path.parent / "intermediate" / "reconstruction"
    )
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    if mode == "image":
        ocr_engine: (
            PaddleOCREngine
            | AutoOCREngine
            | PPStructureV3Engine
            | PaddleOCRVLEngine
            | None
        ) = None
    elif analysis_engine == "vlm":
        ocr_engine = PaddleOCRVLEngine(ocr_device)
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
