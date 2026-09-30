"""Small synthetic COCO cases; no model weights or external datasets required."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from score_predictions import evaluate_predictions


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.annotations = self.folder / 'annotations.json'
        self.predictions = self.folder / 'predictions.json'
        # Non-contiguous category IDs and missing `info` match common dataset exports.
        ground_truth = {
            'images': [{'id': i, 'width': 100, 'height': 100} for i in (1, 2)],
            'categories': [{'id': 3, 'name': 'person'}],
            'annotations': [
                {'id': i, 'image_id': i, 'category_id': 3, 'bbox': [10, 10, 20, 20],
                 'area': 400, 'iscrowd': 0} for i in (1, 2)
            ],
        }
        self.annotations.write_text(json.dumps(ground_truth), encoding='utf-8')
        self.detections = [
            {'image_id': i, 'category_id': 3, 'bbox': [10, 10, 20, 20], 'score': 0.9}
            for i in (1, 2)
        ]

    def score(self, detections):
        self.predictions.write_text(json.dumps(detections), encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()):
            return evaluate_predictions(self.annotations, self.predictions)

    def test_perfect_detections_and_input_hash(self):
        report = self.score(self.detections)
        self.assertAlmostEqual(report['metrics']['AP'], 1.0)
        self.assertEqual(report['metrics']['AP_medium'], -1.0)
        self.assertEqual(report['evaluation']['category_ids'], [3])
        self.assertEqual(report['inputs']['annotations']['sha256'],
                         hashlib.sha256(self.annotations.read_bytes()).hexdigest())

    def test_unpredicted_images_still_count(self):
        report = self.score(self.detections[:1])
        self.assertEqual(report['evaluation']['image_count'], 2)
        self.assertAlmostEqual(report['metrics']['AR100'], 0.5)
        self.assertGreater(report['metrics']['AP'], 0.5)
        self.assertLess(report['metrics']['AP'], 0.51)

    def test_empty_predictions(self):
        report = self.score([])
        self.assertEqual(report['metrics']['AP'], 0.0)
        self.assertEqual(report['metrics']['AR100'], 0.0)
        self.assertEqual(report['evaluation']['prediction_count'], 0)

    def test_unknown_ids_and_nonfinite_values_are_rejected(self):
        for field, value in [('image_id', 99), ('category_id', 99), ('score', float('nan'))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.score([{**self.detections[0], field: value}])

    def test_cli_json_output_and_input_overwrite_guard(self):
        self.score(self.detections)
        command = [sys.executable, str(ROOT / 'tools' / 'score_predictions.py'),
                   '--annotations', str(self.annotations), '--predictions', str(self.predictions)]
        output = self.folder / 'metrics.json'
        process = subprocess.run(command + ['--output', str(output)], capture_output=True,
                                 text=True, encoding='utf-8', check=True)
        self.assertEqual(json.loads(process.stdout), json.loads(output.read_text(encoding='utf-8')))
        before = self.predictions.read_bytes()
        rejected = subprocess.run(command + ['--output', str(self.predictions)], capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(self.predictions.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
