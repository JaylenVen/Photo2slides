from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from photo_to_pptx import parse_args  # noqa: E402


class PhotoToPptxCliTests(unittest.TestCase):
    def test_vlm_is_the_default_without_deepseek_options(self) -> None:
        args = parse_args([])

        self.assertEqual(args.analysis_engine, "vlm")
        self.assertEqual(args.ocr_device, "auto")
        self.assertFalse(hasattr(args, "deepseek_refine"))
        self.assertEqual(
            args.model,
            Path(__file__).resolve().parents[1] / "models" / "slide-seg.pt",
        )
        self.assertEqual(args.output.name, "photo2slide-vlm.pptx")
        self.assertEqual(args.work_dir.name, "latest")


if __name__ == "__main__":
    unittest.main()
