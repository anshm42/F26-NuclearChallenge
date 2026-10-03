import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SPLITTER = Path(__file__).parents[1] / "nuclear_dataset_splitter.py"


class DataSplitTests(unittest.TestCase):
    def test_creates_configurable_model_ready_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "ML_Dataset"
            for scenario in ("Normal", "LOCA"):
                scenario_directory = source / scenario
                scenario_directory.mkdir(parents=True)
                for run_number in range(20):
                    (scenario_directory / f"{run_number}.csv").write_text(
                        "TIME,P\n0,100\n", encoding="utf-8"
                    )
            rare_directory = source / "Rare"
            rare_directory.mkdir()
            (rare_directory / "1.csv").write_text("TIME,P\n0,100\n", encoding="utf-8")

            command = [
                sys.executable,
                str(SPLITTER),
                "--input-folder",
                str(source),
                "--output-folder",
                str(output),
                "--training-ratio",
                "0.20",
                "--validation-ratio",
                "0.30",
                "--testing-ratio",
                "0.50",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)

            self.assertEqual(len(list((output / "Training").rglob("*.csv"))), 9)
            self.assertEqual(len(list((output / "Validation").rglob("*.csv"))), 12)
            self.assertEqual(len(list((output / "Testing").rglob("*.csv"))), 20)
            self.assertTrue((output / "Training" / "Rare" / "1.csv").exists())
            self.assertTrue((output / "dataset_manifest.csv").exists())
            self.assertTrue((output / "split_report.txt").exists())

            failed = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Use --overwrite", failed.stderr)


if __name__ == "__main__":
    unittest.main()
