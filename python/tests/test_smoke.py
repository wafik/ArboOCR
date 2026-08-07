# Smoke tests for arboocr Python bindings (no real models required for most cases).
import os
import sys
import unittest
from pathlib import Path

# Allow running without install: repo/python on path
_REPO_PYTHON = Path(__file__).resolve().parents[1]
if str(_REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(_REPO_PYTHON))

try:
    from arboocr import (EngineConfig, resolve_model_paths, to_json, PagePrediction,
                         LinePrediction, Point2f, WordBox)
    import arboocr
    _IMPORT_OK = True
    _IMPORT_ERR = ""
except ImportError as e:
    _IMPORT_OK = False
    _IMPORT_ERR = str(e)


@unittest.skipUnless(_IMPORT_OK, f"extension not built: {_IMPORT_ERR}")
class TestArboocrSmoke(unittest.TestCase):
    def test_config_defaults(self):
        cfg = EngineConfig()
        self.assertEqual(cfg.ocr_version, "PP-OCRv6")
        self.assertEqual(cfg.model_type, "small")
        self.assertEqual(cfg.rec_batch_num, 6)
        self.assertTrue(cfg.use_fp16)
        self.assertFalse(cfg.use_clahe)
        self.assertFalse(cfg.split_overmerged)
        self.assertAlmostEqual(cfg.minimum_confidence, 0.5)
        self.assertEqual(cfg.models_dir, "models")
        self.assertEqual(cfg.det_model_path, "")
        self.assertEqual(cfg.rec_model_path, "")

    def test_resolve_model_paths_defaults(self):
        cfg = EngineConfig()
        cfg.models_dir = "models"
        cfg.model_type = "medium"
        paths = resolve_model_paths(cfg)
        self.assertTrue(paths["det"].replace("\\", "/").endswith("PP-OCRv6_det.onnx"))
        self.assertTrue(paths["rec"].replace("\\", "/").endswith("PP-OCRv6_rec_medium.onnx"))
        self.assertTrue(paths["dict"].replace("\\", "/").endswith("PP-OCRv6_rec_medium_dict.txt"))

    def test_resolve_model_paths_override(self):
        cfg = EngineConfig()
        cfg.rec_model_path = "custom/rec.onnx"
        paths = resolve_model_paths(cfg)
        self.assertEqual(paths["rec"], "custom/rec.onnx")

    def test_to_json_empty_page(self):
        page = PagePrediction()
        page.image = "none.jpg"
        js = to_json(page)
        self.assertIn("none.jpg", js)
        self.assertIn("lines", js)

    def test_line_prediction_fields(self):
        line = LinePrediction()
        line.text = "hi"
        line.score = 0.9
        line.det_score = 0.8
        line.polygon = [Point2f(1.0, 2.0), Point2f(3.0, 4.0)]
        self.assertEqual(line.text, "hi")
        self.assertAlmostEqual(line.score, 0.9)
        self.assertAlmostEqual(line.det_score, 0.8)
        self.assertEqual(len(line.polygon), 2)

    def test_word_boxes_default_off_and_settable(self):
        cfg = EngineConfig()
        self.assertFalse(cfg.return_word_boxes)
        cfg.return_word_boxes = True
        self.assertTrue(cfg.return_word_boxes)

    def test_word_box_fields_and_line_words(self):
        w = WordBox()
        w.text = "hi"
        w.score = 0.9
        w.polygon = [Point2f(0.0, 0.0), Point2f(4.0, 0.0),
                     Point2f(4.0, 2.0), Point2f(0.0, 2.0)]
        line = LinePrediction()
        self.assertEqual(len(line.words), 0)  # empty unless return_word_boxes
        line.words = [w]
        self.assertEqual(len(line.words), 1)
        self.assertEqual(line.words[0].text, "hi")
        self.assertEqual(len(line.words[0].polygon), 4)
        self.assertIn("hi", repr(line.words[0]))

    def test_words_serialize_into_json(self):
        w = WordBox()
        w.text = "hi"
        w.score = 0.9
        w.polygon = [Point2f(0.0, 0.0), Point2f(4.0, 0.0),
                     Point2f(4.0, 2.0), Point2f(0.0, 2.0)]
        line = LinePrediction()
        line.text = "hi"
        line.words = [w]
        self.assertIn("words", to_json(line))
        # Absent when empty, so the shape wrappers parse is unchanged.
        self.assertNotIn("words", to_json(LinePrediction()))


if __name__ == "__main__":
    unittest.main()
