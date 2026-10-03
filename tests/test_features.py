from pathlib import Path
import tempfile
import unittest

import pandas as pd

from leak_detection.features import DIRECT_LEAK_COLUMNS, FeatureConfig, extract_run_features


class FeatureExtractionTests(unittest.TestCase):
    def test_extracts_early_window_and_excludes_direct_leak_signals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "LOCA" / "1.csv"
            path.parent.mkdir()
            pd.DataFrame(
                {
                    "TIME": [0.0, 10.0, 20.0],
                    "PRESSURE": [100.0, 90.0, 0.0],
                    "WLR": [0.0, 5.0, 10.0],
                }
            ).to_csv(path, index=False)

            features = extract_run_features(path, FeatureConfig(max_time_seconds=10.0))

        self.assertEqual(features["PRESSURE__last"], 90.0)
        self.assertEqual(features["PRESSURE__delta"], -10.0)
        self.assertEqual(features["PRESSURE__slope"], -1.0)
        self.assertFalse(any(name.startswith("WLR__") for name in features.index))
        self.assertIn("WLR", DIRECT_LEAK_COLUMNS)

    def test_rejects_missing_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Normal" / "1.csv"
            path.parent.mkdir()
            pd.DataFrame({"P": [1.0]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "TIME"):
                extract_run_features(path, FeatureConfig())


if __name__ == "__main__":
    unittest.main()
