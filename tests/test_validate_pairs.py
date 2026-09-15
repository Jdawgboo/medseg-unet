from __future__ import annotations
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from medseg_unet.cli import main
from medseg_unet.pipeline import list_case_ids, validate_data_dir


def _make_layout(root: Path, images: list[str], masks: list[str]) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    for case_id in images:
        (root / "images" / f"{case_id}.pt").touch()
    for case_id in masks:
        (root / "masks" / f"{case_id}.pt").touch()
    return root


class ValidatePairsTests(unittest.TestCase):
    def test_valid_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_layout(Path(tmp), ["case-001", "case-002"], ["case-001", "case-002"])
            self.assertEqual(validate_data_dir(tmp), ["case-001", "case-002"])
            self.assertEqual(list_case_ids(tmp), ({"case-001", "case-002"}, {"case-001", "case-002"}))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["validate-pairs", "--data", tmp])
            self.assertEqual(code, 0)
            self.assertIn("Validated 2 paired cases", buf.getvalue())

    def test_image_only_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_layout(Path(tmp), ["case-001", "image-only"], ["case-001"])
            with self.assertRaises(ValueError):
                validate_data_dir(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["validate-pairs", "--data", tmp])
            self.assertNotEqual(code, 0)
            output = buf.getvalue()
            self.assertIn("image-only", output)
            self.assertIn("images/<case_id>.pt", output)

    def test_mask_only_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_layout(Path(tmp), ["case-001"], ["case-001", "mask-only"])
            with self.assertRaises(ValueError):
                validate_data_dir(tmp)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(["validate-pairs", "--data", tmp])
            self.assertNotEqual(code, 0)
            output = buf.getvalue()
            self.assertIn("mask-only", output)
            self.assertIn("masks/<case_id>.pt", output)


if __name__ == "__main__":
    unittest.main()
