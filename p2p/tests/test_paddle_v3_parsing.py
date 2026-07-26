from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from editable_pptx import (  # noqa: E402
    OCRLine,
    PPStructureV3Engine,
    _clean_text_and_build_specs,
    _layout_regions_from_v3_output,
    _normalize_vlm_text,
    _ocr_lines_from_v3_output,
    _wrap_text_for_rows,
    _vlm_blocks_from_output,
)


class PaddleV3ParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = [
            {
                "res": {
                    "overall_ocr_res": {
                        "rec_texts": ["Slide title", "ignored"],
                        "rec_scores": [0.96, 0.4],
                        "rec_polys": np.asarray(
                            [
                                [[10, 20], [210, 20], [210, 60], [10, 60]],
                                [[20, 80], [120, 80], [120, 100], [20, 100]],
                            ],
                            dtype=np.int16,
                        ),
                    },
                    "layout_det_res": {
                        "boxes": [
                            {
                                "label": "image",
                                "score": 0.984,
                                "coordinate": [300.2, 100.7, 800.8, 500.1],
                            },
                            {
                                "label": "text",
                                "score": 0.95,
                                "coordinate": [10, 20, 210, 60],
                            },
                        ]
                    },
                }
            }
        ]

    def test_reads_ocr_lines_from_pp_structure_result(self) -> None:
        lines = _ocr_lines_from_v3_output(
            self.output,
            language="en",
            min_confidence=0.78,
        )
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "Slide title")
        self.assertEqual(lines[0].bbox, (10, 20, 210, 60))

    def test_reads_layout_regions_with_integer_bounds(self) -> None:
        regions = _layout_regions_from_v3_output(self.output)
        self.assertEqual(regions[0]["label"], "image")
        self.assertEqual(regions[0]["bbox"], [300, 100, 801, 501])
        self.assertEqual(len(regions), 2)

    def test_structure_dependency_error_has_actionable_install_command(self) -> None:
        class BrokenPPStructureV3:
            def __init__(self, **kwargs: object) -> None:
                raise RuntimeError(
                    "A dependency error occurred during pipeline creation."
                )

        fake_paddleocr = types.SimpleNamespace(PPStructureV3=BrokenPPStructureV3)
        with (
            patch("editable_pptx._package_major_version", return_value=3),
            patch("editable_pptx._configure_windows_torch_dll_search_path"),
            patch.dict(sys.modules, {"paddleocr": fake_paddleocr}),
        ):
            with self.assertRaisesRegex(RuntimeError, "requirements-vision.txt"):
                PPStructureV3Engine("auto", "cpu")

    def test_reads_multiline_text_and_visual_regions_from_vlm_result(self) -> None:
        output = [
            {
                "res": {
                    "layout_det_res": {
                        "boxes": [
                            {
                                "label": "doc_title",
                                "score": 0.99,
                                "coordinate": [80, 40, 920, 120],
                            },
                            {
                                "label": "text",
                                "score": 0.94,
                                "coordinate": [80, 170, 600, 320],
                            },
                            {
                                "label": "image",
                                "score": 0.98,
                                "coordinate": [680, 170, 1180, 620],
                            },
                        ]
                    },
                    "parsing_res_list": [
                        {
                            "block_label": "doc_title",
                            "block_content": "# Enacting &amp; ambient care",
                            "block_bbox": [80, 40, 920, 120],
                        },
                        {
                            "block_label": "text",
                            "block_content": "- First point<br>Second line",
                            "block_bbox": [80, 170, 600, 320],
                        },
                        {
                            "block_label": "image",
                            "block_content": "",
                            "block_bbox": [680, 170, 1180, 620],
                        },
                    ],
                }
            }
        ]

        lines, regions, blocks = _vlm_blocks_from_output(output)

        self.assertEqual([line.label for line in lines], ["doc_title", "text"])
        self.assertEqual(lines[0].text, "Enacting & ambient care")
        self.assertEqual(lines[1].text, "• First point\nSecond line")
        self.assertAlmostEqual(lines[1].confidence, 0.94)
        self.assertEqual(len(regions), 3)
        self.assertEqual(len(blocks), 3)

    def test_prefers_serialized_data_for_dict_subclass_results(self) -> None:
        class PaddleXResult(dict):
            @property
            def json(self) -> dict[str, object]:
                return {
                    "res": {
                        "layout_det_res": {
                            "boxes": [
                                {
                                    "label": "text",
                                    "score": 0.91,
                                    "coordinate": [10, 20, 210, 60],
                                }
                            ]
                        },
                        "parsing_res_list": [
                            {
                                "block_label": "text",
                                "block_content": "Serialized content",
                                "block_bbox": [10, 20, 210, 60],
                            }
                        ],
                    }
                }

        lines, regions, blocks = _vlm_blocks_from_output([PaddleXResult()])

        self.assertEqual([line.text for line in lines], ["Serialized content"])
        self.assertEqual(len(regions), 1)
        self.assertEqual(len(blocks), 1)

    def test_wraps_vlm_text_without_changing_content(self) -> None:
        chinese = "环境式关怀的生成：智能音箱与中国家庭代际照护的协商"
        english = (
            "Smart speakers and the negotiation of intergenerational care "
            "in Chinese families"
        )

        chinese_rows = _wrap_text_for_rows(chinese, 2).splitlines()
        english_rows = _wrap_text_for_rows(english, 2).splitlines()

        self.assertEqual(len(chinese_rows), 2)
        self.assertEqual("".join(chinese_rows), chinese)
        self.assertEqual(len(english_rows), 2)
        self.assertEqual(" ".join(english_rows), english)

    def test_converts_inline_latex_to_plain_editable_text(self) -> None:
        self.assertEqual(
            _normalize_vlm_text(r"Grid: $25\,m \times 25\,m$"),
            "Grid: 25 m × 25 m",
        )
        self.assertEqual(_normalize_vlm_text(r"2 $ ^{nd} $ edition"), "2nd edition")

    def test_text_specs_keep_a_safe_slide_margin(self) -> None:
        image = np.full((900, 1600, 3), 245, dtype=np.uint8)
        line = OCRLine(
            polygon=np.asarray(
                [[0, 0], [1599, 0], [1599, 80], [0, 80]],
                dtype=np.float32,
            ),
            text="A title that originally touches the slide edge",
            confidence=0.98,
            label="doc_title",
        )

        _, _, specs = _clean_text_and_build_specs(image, [line], "Arial")
        spec = specs[0]

        self.assertGreaterEqual(spec["x"], 1600 * 0.012)
        self.assertGreaterEqual(spec["y"], 900 * 0.012)
        self.assertLessEqual(spec["x"] + spec["w"], 1600 * (1 - 0.012))
        self.assertLessEqual(spec["y"] + spec["h"], 900 * (1 - 0.012))


if __name__ == "__main__":
    unittest.main()
