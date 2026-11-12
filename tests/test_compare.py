"""scripts/compare.py per-image semantic metrics on a synthetic 2x2 case (no gold needed)."""

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare import semantic  # noqa: E402


class CompareSemantic(unittest.TestCase):
    def test_per_image_iou_confusion_and_void(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            task = tmp / "task"; (task / "images").mkdir(parents=True)
            (task / "classes.json").write_text(json.dumps({"trainid": {"road": 0, "sky": 10}}))
            gt = tmp / "gt"; gt.mkdir()
            Image.fromarray(np.array([[0, 0], [10, 255]], np.uint8)).save(gt / "a.png")
            # prediction: road, sky / sky, sky  -> road IoU 1/2, sky IoU 1/3, one pixel on void
            pred = np.array([[[128, 64, 128], [70, 130, 180]], [[70, 130, 180], [70, 130, 180]]], np.uint8)
            buf = io.BytesIO(); Image.fromarray(pred).save(buf, "PNG")
            sub = tmp / "sub.zip"
            with zipfile.ZipFile(sub, "w") as z:
                z.writestr("labelmap.txt", "road:128,64,128::\nsky:70,130,180::\n")
                z.writestr("SegmentationClass/a.png", buf.getvalue())

            overall, rows = semantic(task, gt, sub, None)
            r = rows[0]
            self.assertAlmostEqual(r["per_class_iou"]["road"], 0.5)
            self.assertAlmostEqual(r["per_class_iou"]["sky"], 1 / 3)
            self.assertEqual(r["missed_as"]["road"], {"sky": 1})
            self.assertEqual(r["pixels_on_void"], 1)
            self.assertAlmostEqual(overall["value"], (0.5 + 1 / 3) / 2)


if __name__ == "__main__":
    unittest.main()
