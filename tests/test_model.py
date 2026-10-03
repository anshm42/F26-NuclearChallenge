import unittest

import numpy as np

from leak_detection.model import choose_threshold, classification_metrics


class ThresholdTests(unittest.TestCase):
    def test_selects_high_precision_threshold_at_required_recall(self) -> None:
        targets = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        threshold = choose_threshold(targets, probabilities, minimum_recall=1.0)
        self.assertEqual(threshold, 0.6)

        metrics = classification_metrics(targets, probabilities, threshold)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_requires_a_positive_validation_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one leak"):
            choose_threshold(np.array([0, 0]), np.array([0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
