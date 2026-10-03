import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from leak_detection.cli import (
    run_evaluate,
    run_evaluate_types,
    run_predict,
    run_train,
    run_train_types,
)


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

    def test_train_then_evaluate_leak_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            validation = root / "validation"
            test = root / "test"
            leak_classes = ("FLB", "LOCA", "SGATR")
            for run_number in range(1, 7):
                self.write_run(train, "Normal", run_number, 0.05 * run_number)
            for run_number in range(1, 3):
                self.write_run(validation, "Normal", run_number, 0.1 * run_number)
                self.write_run(test, "Normal", run_number, 0.15 * run_number)

            for class_index, scenario in enumerate(leak_classes, start=1):
                for run_number in range(1, 7):
                    self.write_run(
                        train, scenario, run_number, class_index * 2.0 + run_number
                    )
                for run_number in range(1, 3):
                    self.write_run(
                        validation, scenario, run_number, class_index * 3.0 + run_number
                    )
                    self.write_run(
                        test, scenario, run_number, class_index * 4.0 + run_number
                    )

            artifact = root / "artifacts" / "type_model.joblib"
            run_train_types(
                Namespace(
                    train_dir=str(train),
                    validation_dir=str(validation),
                    output=str(artifact),
                    leak_classes=leak_classes,
                    exclude_columns=(),
                    max_time_seconds=20.0,
                    random_state=42,
                )
            )
            self.assertTrue(artifact.exists())

            report_directory = root / "type_reports"
            run_evaluate_types(
                Namespace(
                    model=str(artifact),
                    test_dir=str(test),
                    output_dir=str(report_directory),
                )
            )
            self.assertTrue(
                (report_directory / "leak_type_test_metrics.json").exists()
            )
            self.assertTrue(
                (report_directory / "leak_type_test_predictions.csv").exists()
            )

            binary_artifact = root / "artifacts" / "binary_model.joblib"
            run_train(
                Namespace(
                    train_dir=str(train),
                    validation_dir=str(validation),
                    output=str(binary_artifact),
                    positive_classes=leak_classes,
                    exclude_columns=(),
                    max_time_seconds=20.0,
                    minimum_recall=1.0,
                    random_state=42,
                )
            )
            prediction_path = root / "prediction.json"
            run_predict(
                Namespace(
                    binary_model=str(binary_artifact),
                    type_model=str(artifact),
                    input_csv=str(test / "LOCA" / "1.csv"),
                    output=str(prediction_path),
                )
            )
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            self.assertIn("leak_probability", prediction)
            self.assertEqual(set(prediction["leak_type_probabilities"]), set(leak_classes))

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
