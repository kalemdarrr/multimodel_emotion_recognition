from __future__ import annotations

import unittest

import numpy as np

from multimodal_emotion.evaluation.metrics import classification_metrics, confusion_records


class MetricsTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        probabilities = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.6, 0.4],
            ]
        )
        metrics = classification_metrics(y_true, y_pred, ["neutral", "joy"], probabilities)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertIn("macro_f1", metrics)
        self.assertIn("brier_score", metrics)

    def test_confusion_records(self) -> None:
        records = confusion_records([0, 1, 1], [0, 1, 0], ["neutral", "joy"])
        self.assertEqual(len(records), 4)
        self.assertEqual(sum(record["count"] for record in records), 3)


if __name__ == "__main__":
    unittest.main()
