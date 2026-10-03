import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from leak_detection.cli import run_evaluate, run_train


class CliIntegrationTests(unittest.TestCase):
    @staticmethod
    def write_run(root: Path, scenario: str, run_number: int, offset: float) -> None:
        path = root / scenario / f"{run_number}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "TIME": [0.0, 10.0, 20.0],
                "P": [100.0, 100.0 - offset, 100.0 - (2 * offset)],
                "TAVG": [300.0, 300.0 + offset, 300.0 + (2 * offset)],
                "WLR": [0.0, offset, 2 * offset],
            }
        ).to_csv(path, index=False)

    def test_train_then_evaluate_without_test_data_in_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            validation = root / "validation"
            test = root / "test"
            for run_number in range(1, 5):
                self.write_run(train, "Normal", run_number, 0.05 * run_number)
                self.write_run(train, "LOCA", run_number, 2.0 + run_number)
            for run_number in range(1, 3):
                self.write_run(validation, "Normal", run_number, 0.1 * run_number)
                self.write_run(validation, "LOCA", run_number, 3.0 + run_number)
                self.write_run(test, "Normal", run_number, 0.15 * run_number)
                self.write_run(test, "LOCA", run_number, 4.0 + run_number)

            artifact = root / "artifacts" / "model.joblib"
            run_train(
                Namespace(
                    train_dir=str(train),
                    validation_dir=str(validation),
                    output=str(artifact),
                    positive_classes=("LOCA",),
                    exclude_columns=(),
                    max_time_seconds=20.0,
                    minimum_recall=1.0,
                    random_state=42,
                )
            )
            self.assertTrue(artifact.exists())

            report_directory = root / "reports"
            run_evaluate(
                Namespace(
                    model=str(artifact),
                    test_dir=str(test),
                    output_dir=str(report_directory),
                )
            )
            self.assertTrue((report_directory / "test_metrics.json").exists())
            self.assertTrue((report_directory / "test_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
