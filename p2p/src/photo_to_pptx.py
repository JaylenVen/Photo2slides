from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from editable_pptx import build_editable_pptx
from image_enhancement import (
    enhance_slide_image,
    prepare_ocr_image,
    resolve_rectified_size,
)

if TYPE_CHECKING:
    from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "slide-seg.pt"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "work" / "latest"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "photo2slide-vlm.pptx"

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MASK_COLOR = np.array([0, 80, 0], dtype=np.uint8)
MASK_ALPHA = 0.45
_DLL_DIRECTORY_HANDLES: list[Any] = []


def configure_windows_torch_dll_search_path() -> None:
    """Prefer the DLLs bundled with torch over conflicting Anaconda copies."""
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
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if not path_entries or path_entries[0].casefold() != torch_lib_text.casefold():
        os.environ["PATH"] = os.pathsep.join(
            [torch_lib_text, *(entry for entry in path_entries if entry)]
        )
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(torch_lib_text))


def natural_sort_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def get_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")

    image_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"输入文件夹中没有支持的图片：{input_dir}")

    stems = [path.stem.lower() for path in image_paths]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        raise ValueError("输入图片存在同名文件，过程图会互相覆盖：" + ", ".join(duplicate_stems))
    return image_paths


def resize_mask_to_image(mask: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8)


def select_largest_mask(result) -> np.ndarray | None:
    if result.masks is None:
        return None
    masks = result.masks.data.cpu().numpy()
    if len(masks) == 0:
        return None
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    return masks[int(np.argmax(areas))]


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("四角坐标必须为 (4, 2)")
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError("幻灯片四角存在重复坐标")

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)

    first_edge = ordered[1] - ordered[0]
    last_edge = ordered[3] - ordered[0]
    cross_product = first_edge[0] * last_edge[1] - first_edge[1] * last_edge[0]
    if cross_product < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def fit_slide_quadrilateral(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("分割掩膜为空")

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        raise ValueError("分割掩膜没有有效面积")

    candidates: list[np.ndarray] = []
    for curve in (contour, cv2.convexHull(contour)):
        perimeter = cv2.arcLength(curve, True)
        if perimeter <= 0:
            continue
        for ratio in np.linspace(0.002, 0.08, 80):
            approximation = cv2.approxPolyDP(curve, ratio * perimeter, True)
            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                candidates.append(approximation.reshape(4, 2).astype(np.float32))

    candidates.append(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32))
    original_area = float(binary_mask.sum())
    best_score = -1.0
    best_mask: np.ndarray | None = None
    best_points: np.ndarray | None = None

    for points in candidates:
        candidate_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(candidate_mask, [np.rint(points).astype(np.int32)], 1)
        candidate_area = float(candidate_mask.sum())
        if candidate_area <= 0:
            continue
        intersection = float(np.logical_and(binary_mask, candidate_mask).sum())
        union = float(np.logical_or(binary_mask, candidate_mask).sum())
        if union <= 0:
            continue
        overlap = intersection / union
        area_delta = abs(candidate_area - original_area) / original_area
        score = overlap - 0.02 * area_delta
        if score > best_score:
            best_score = score
            best_mask = candidate_mask
            best_points = points

    if best_mask is None or best_points is None:
        raise ValueError("无法把分割掩膜拟合为四边形")
    return best_mask, order_points(best_points)


def extract_slide_region(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    extracted = np.zeros_like(image)
    extracted[mask.astype(bool)] = image[mask.astype(bool)]
    return extracted


def make_debug_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[mask.astype(bool)] = MASK_COLOR
    return cv2.addWeighted(overlay, MASK_ALPHA, image, 1 - MASK_ALPHA, 0)


def warp_slide_to_rectangle(
    image: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"无法写入图片：{path}")


def release_torch_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _vlm_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault(
        "PADDLE_PDX_CACHE_HOME",
        str((PROJECT_ROOT / "data" / "cache" / "paddlex").resolve()),
    )
    environment.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    environment.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    return environment


def _build_pptx_with_vlm_subprocess(
    image_paths: list[Path],
    output_path: Path,
    *,
    ppt_mode: str,
    ocr_device: str,
    ocr_confidence: float,
    font_name: str,
    work_dir: Path | None,
) -> Path:
    output_path = output_path.expanduser().resolve()
    reconstruction_dir = (
        work_dir.expanduser().resolve()
        if work_dir is not None
        else output_path.parent / "intermediate" / "reconstruction"
    )
    worker = Path(__file__).with_name("vlm_reconstruct.py")
    command = [
        sys.executable,
        str(worker),
        "--output",
        str(output_path),
        "--work-dir",
        str(reconstruction_dir),
        "--ppt-mode",
        ppt_mode,
        "--analysis-device",
        ocr_device,
        "--ocr-conf",
        str(ocr_confidence),
        "--font-name",
        font_name,
    ]
    for image_path in image_paths:
        command.extend(["--image", str(image_path.expanduser().resolve())])
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT.parent),
        env=_vlm_subprocess_environment(),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"PaddleOCR-VL 子进程退出码：{result.returncode}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("PaddleOCR-VL 子进程未生成有效 PPTX")
    return output_path


def process_image(
    model: Any,
    image_path: Path,
    work_dir: Path,
    device: str | None,
    imgsz: int,
    confidence: float,
    width: int,
    height: int,
) -> Path | None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    predict_args = {
        "source": str(image_path),
        "imgsz": imgsz,
        "conf": confidence,
        "retina_masks": True,
        "verbose": False,
    }
    if device is not None:
        predict_args["device"] = device

    results = model.predict(**predict_args)
    if not results:
        return None
    mask = select_largest_mask(results[0])
    if mask is None:
        return None

    mask = resize_mask_to_image(mask, image.shape)
    quadrilateral_mask, corners = fit_slide_quadrilateral(mask)
    output_width, output_height = resolve_rectified_size(
        corners,
        requested_width=width,
        requested_height=height,
    )
    extracted = extract_slide_region(image, quadrilateral_mask)
    debug = make_debug_image(image, quadrilateral_mask)
    rectified = warp_slide_to_rectangle(image, corners, output_width, output_height)
    enhanced = enhance_slide_image(rectified)
    ocr_preview = prepare_ocr_image(enhanced).image

    extract_path = work_dir / "extracted" / f"{image_path.stem}.png"
    debug_path = work_dir / "masks" / f"{image_path.stem}.png"
    rectified_path = work_dir / "rectified" / f"{image_path.stem}.png"
    enhanced_path = work_dir / "enhanced" / f"{image_path.stem}.png"
    ocr_path = work_dir / "ocr" / f"{image_path.stem}.png"
    write_image(extract_path, extracted)
    write_image(debug_path, debug)
    write_image(rectified_path, rectified)
    write_image(enhanced_path, enhanced)
    write_image(ocr_path, ocr_preview)
    return enhanced_path


def build_pptx(
    image_paths: list[Path],
    output_path: Path,
    *,
    ppt_mode: str = "editable",
    analysis_engine: str = "vlm",
    ocr_language: str = "auto",
    ocr_device: str = "auto",
    ocr_confidence: float = 0.78,
    font_name: str = "Microsoft YaHei",
    work_dir: Path | None = None,
) -> Path:
    sorted_paths = sorted(image_paths, key=natural_sort_key)
    if analysis_engine == "vlm" and ppt_mode != "image":
        return _build_pptx_with_vlm_subprocess(
            sorted_paths,
            output_path,
            ppt_mode=ppt_mode,
            ocr_device=ocr_device,
            ocr_confidence=ocr_confidence,
            font_name=font_name,
            work_dir=work_dir,
        )
    return build_editable_pptx(
        sorted_paths,
        output_path,
        mode=ppt_mode,
        analysis_engine=analysis_engine,
        ocr_language=ocr_language,
        ocr_device=ocr_device,
        min_text_confidence=ocr_confidence,
        font_name=font_name,
        work_dir=work_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从会议现场照片中提取幻灯片，并重建为可编辑 PPTX。"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--work-dir",
        "--debug-dir",
        dest="work_dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="过程图根目录；--debug-dir 作为旧参数名仍可使用。",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--device",
        default="auto",
        help="YOLO 推理设备，例如 0、cpu；默认 auto 由 Ultralytics 自动选择。",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="校正图宽度；0 表示按照片有效像素自动选择。",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="校正图高度；0 表示按照片有效像素自动选择。",
    )
    parser.add_argument(
        "--ppt-mode",
        choices=("editable", "hybrid", "image"),
        default="editable",
        help=(
            "editable=原生文字/形状并拆分复杂视觉；hybrid=清理后的背景图+原生文字；"
            "image=旧版整页图片。"
        ),
    )
    parser.add_argument(
        "--analysis-engine",
        choices=("vlm", "ocr", "structure"),
        default="vlm",
        help=(
            "vlm=使用 PaddleOCR-VL 1.6 直接理解整页（默认）；"
            "ocr/structure 仅保留为本地诊断后备。"
        ),
    )
    parser.add_argument("--ocr-lang", choices=("auto", "ch", "en"), default="auto")
    parser.add_argument(
        "--analysis-device",
        "--ocr-device",
        dest="ocr_device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="页面理解设备；auto 优先使用可用 GPU。",
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.78,
        help="仅用于 ocr/structure 后备引擎的文字置信度阈值。",
    )
    parser.add_argument("--font-name", default="Microsoft YaHei")
    args = parser.parse_args(argv)

    if not 0 <= args.conf <= 1:
        parser.error("--conf 必须在 0 到 1 之间")
    if not 0 <= args.ocr_conf <= 1:
        parser.error("--ocr-conf 必须在 0 到 1 之间")
    if args.imgsz <= 0:
        parser.error("--imgsz 必须为正整数")
    if args.width < 0 or args.height < 0:
        parser.error("--width 和 --height 不能为负数")
    if (args.width == 0) != (args.height == 0):
        parser.error("--width 和 --height 必须同时为 0（自动）或同时指定")
    if args.width and args.width * 9 != args.height * 16:
        parser.error("--width 和 --height 必须保持 16:9")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.model.is_file():
        print(f"[错误] 模型文件不存在：{args.model}")
        return 1

    try:
        image_paths = get_image_files(args.input_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}")
        return 1

    device = None if args.device.lower() == "auto" else args.device
    print(f"加载模型：{args.model}")
    print(f"输入图片：{len(image_paths)} 张")
    print(f"推理设备：{args.device}")
    print(f"PPT 模式：{args.ppt_mode}")
    print(f"页面分析：{args.analysis_engine}")
    if args.analysis_engine == "vlm":
        print("视觉模型：PaddleOCR-VL 1.6（本地，不使用 DeepSeek）")
    resolution = "自动（1280 至 2560 宽）" if args.width == 0 else f"{args.width}x{args.height}"
    print(f"校正分辨率：{resolution}")
    try:
        configure_windows_torch_dll_search_path()
        from ultralytics import YOLO

        model = YOLO(str(args.model))
    except Exception as exc:
        print(f"[错误] 加载模型失败：{exc}")
        return 1

    successful: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for index, image_path in enumerate(image_paths, start=1):
        try:
            stretch_path = process_image(
                model=model,
                image_path=image_path,
                work_dir=args.work_dir,
                device=device,
                imgsz=args.imgsz,
                confidence=args.conf,
                width=args.width,
                height=args.height,
            )
            if stretch_path is None:
                failed.append((image_path, "未检测到幻灯片"))
                print(f"[{index}/{len(image_paths)}] 跳过 {image_path.name}：未检测到幻灯片")
            else:
                successful.append(stretch_path)
                print(f"[{index}/{len(image_paths)}] 完成 {image_path.name}")
        except Exception as exc:
            failed.append((image_path, str(exc)))
            print(f"[{index}/{len(image_paths)}] 跳过 {image_path.name}：{exc}")

    if not successful:
        print("[错误] 本次没有成功提取任何幻灯片，未生成 PPTX。")
        return 1

    del model
    release_torch_gpu_memory()
    try:
        output_path = build_pptx(
            successful,
            args.output,
            ppt_mode=args.ppt_mode,
            analysis_engine=args.analysis_engine,
            ocr_language=args.ocr_lang,
            ocr_device=args.ocr_device,
            ocr_confidence=args.ocr_conf,
            font_name=args.font_name,
            work_dir=args.work_dir / "reconstruction",
        )
    except Exception as exc:
        print(f"[错误] 生成 PPTX 失败：{exc}")
        return 1

    print(f"PPTX：{output_path}")
    print(f"成功：{len(successful)} 张；跳过：{len(failed)} 张")
    for image_path, reason in failed:
        print(f"  - {image_path.name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
