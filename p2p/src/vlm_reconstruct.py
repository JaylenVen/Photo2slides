from __future__ import annotations

import argparse
from pathlib import Path

from editable_pptx import build_editable_pptx


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在独立进程中使用 PaddleOCR-VL 重建可编辑 PPTX。"
    )
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--ppt-mode",
        choices=("editable", "hybrid"),
        default="editable",
    )
    parser.add_argument(
        "--analysis-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
    )
    parser.add_argument("--ocr-conf", type=float, default=0.78)
    parser.add_argument("--font-name", default="Microsoft YaHei")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = build_editable_pptx(
        args.image,
        args.output,
        mode=args.ppt_mode,
        analysis_engine="vlm",
        ocr_device=args.analysis_device,
        min_text_confidence=args.ocr_conf,
        font_name=args.font_name,
        work_dir=args.work_dir,
    )
    print(f"VLM PPTX：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
