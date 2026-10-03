"""
Nuclear Reactor Simulation Dataset Splitter

Purpose
-------
Splits complete reactor-simulation CSV files into:
    Training / Validation / Testing

Default split:
    10% Training
    45% Validation
    45% Testing

Important:
- Splits complete CSV simulations, not individual time-series rows.
- Preserves scenario folders as metadata.
- Uses stratified sampling within scenarios for scenarios with >1 file.
- Keeps single-file scenarios together according to SINGLE_FILE_POLICY.
- Produces a manifest and a human-readable report.
- Does NOT invent leak/cause/severity labels.
"""

import argparse
import csv
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="Split complete NPPAD simulation runs.")
parser.add_argument("--input-folder", default="Operation_csv_data")
parser.add_argument("--output-folder", default="ML_Dataset")
parser.add_argument("--training-ratio", type=float, default=0.10)
parser.add_argument("--validation-ratio", type=float, default=0.45)
parser.add_argument("--testing-ratio", type=float, default=0.45)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument(
    "--single-file-policy",
    choices=("training", "validation", "testing", "exclude"),
    default="training",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace an existing output folder.",
)
arguments = parser.parse_args()

TRAINING_RATIO = arguments.training_ratio
VALIDATION_RATIO = arguments.validation_ratio
TESTING_RATIO = arguments.testing_ratio
RANDOM_SEED = arguments.random_seed
INPUT_FOLDER = arguments.input_folder
OUTPUT_FOLDER = arguments.output_folder
SINGLE_FILE_POLICY = arguments.single_file_policy


# ============================================================
# 2. VALIDATE SETTINGS
# ============================================================

ratios = (
    TRAINING_RATIO,
    VALIDATION_RATIO,
    TESTING_RATIO,
)

if any(r < 0 or r > 1 for r in ratios):
    raise ValueError("All dataset ratios must be between 0 and 1.")

if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
    raise ValueError(
        "TRAINING_RATIO + VALIDATION_RATIO + TESTING_RATIO must equal 1.0."
    )

if SINGLE_FILE_POLICY not in {
    "training", "validation", "testing", "exclude"
}:
    raise ValueError(
        "SINGLE_FILE_POLICY must be training, validation, testing, or exclude."
    )


# ============================================================
# 3. FIND SCENARIO FOLDERS
# ============================================================

input_path = Path(INPUT_FOLDER)

if not input_path.exists():
    raise FileNotFoundError(
        f"Input folder not found: {input_path.resolve()}"
    )

scenarios = {}

for folder in sorted(input_path.iterdir()):
    if folder.is_dir():
        files = sorted(folder.glob("*.csv"))
        if files:
            scenarios[folder.name] = files

if not scenarios:
    raise ValueError(
        "No scenario folders containing CSV files were found."
    )


# ============================================================
# 4. PREPARE OUTPUT
# ============================================================

output_path = Path(OUTPUT_FOLDER)

if output_path.exists():
    if not arguments.overwrite:
        raise FileExistsError(
            f"Output folder already exists: {output_path.resolve()}. "
            "Use --overwrite to replace it."
        )
    shutil.rmtree(output_path)

for dataset in ("Training", "Validation", "Testing"):
    for scenario in scenarios:
        (output_path / dataset / scenario).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 5. SPLIT COMPLETE SIMULATIONS
# ============================================================

rng = random.Random(RANDOM_SEED)

manifest_rows = []
scenario_summary = defaultdict(Counter)

for scenario, files in scenarios.items():

    files = files.copy()
    rng.shuffle(files)

    n = len(files)

    # --------------------------------------------------------
    # Special case: only one simulation exists.
    # --------------------------------------------------------
    if n == 1:

        if SINGLE_FILE_POLICY == "exclude":
            continue

        dataset = SINGLE_FILE_POLICY.capitalize()
        source = files[0]

        destination = (
            output_path / dataset / scenario / source.name
        )

        shutil.copy2(source, destination)

        manifest_rows.append({
            "file": str(source.relative_to(input_path)),
            "dataset": dataset,
            "scenario": scenario,
            "single_file_exception": "yes",
        })

        scenario_summary[scenario][dataset] += 1
        continue

    # --------------------------------------------------------
    # Normal case: stratify within the scenario.
    #
    # We use largest-remainder allocation so the integer file
    # counts are as close as possible to the requested ratios.
    # --------------------------------------------------------

    desired = {
        "Training": n * TRAINING_RATIO,
        "Validation": n * VALIDATION_RATIO,
        "Testing": n * TESTING_RATIO,
    }

    counts = {
        name: int(math.floor(value))
        for name, value in desired.items()
    }

    remaining = n - sum(counts.values())

    # Give leftover simulations to the datasets with the
    # largest fractional remainders.
    remainders = sorted(
        desired,
        key=lambda name: desired[name] - counts[name],
        reverse=True,
    )

    for name in remainders[:remaining]:
        counts[name] += 1

    # With small groups, a mathematically exact percentage may
    # produce zero files in one dataset. We do not force a file
    # into every dataset because doing so could distort the
    # requested ratios.
    start = 0

    for dataset in ("Training", "Validation", "Testing"):

        end = start + counts[dataset]

        for source in files[start:end]:

            destination = (
                output_path
                / dataset
                / scenario
                / source.name
            )

            shutil.copy2(source, destination)

            manifest_rows.append({
                "file": str(source.relative_to(input_path)),
                "dataset": dataset,
                "scenario": scenario,
                "single_file_exception": "no",
            })

            scenario_summary[scenario][dataset] += 1

        start = end


# ============================================================
# 6. WRITE MANIFEST
# ============================================================

manifest_path = output_path / "dataset_manifest.csv"

with manifest_path.open("w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "dataset",
            "scenario",
            "single_file_exception",
        ],
    )

    writer.writeheader()
    writer.writerows(manifest_rows)


# ============================================================
# 7. WRITE SUMMARY REPORT
# ============================================================

report_path = output_path / "split_report.txt"

dataset_totals = Counter(
    row["dataset"] for row in manifest_rows
)

with report_path.open("w", encoding="utf-8") as f:

    f.write("NUCLEAR REACTOR DATASET SPLIT REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Random seed: {RANDOM_SEED}\n")
    f.write(
        f"Requested split: "
        f"{TRAINING_RATIO:.0%} Training / "
        f"{VALIDATION_RATIO:.0%} Validation / "
        f"{TESTING_RATIO:.0%} Testing\n\n"
    )

    f.write(
        f"Total simulations assigned: "
        f"{len(manifest_rows)}\n"
    )
    f.write(f"Training:   {dataset_totals['Training']}\n")
    f.write(f"Validation: {dataset_totals['Validation']}\n")
    f.write(f"Testing:    {dataset_totals['Testing']}\n\n")

    f.write("SCENARIO DISTRIBUTION\n")
    f.write("-" * 60 + "\n")
    f.write(
        f"{'Scenario':<12}"
        f"{'Total':>8}"
        f"{'Train':>10}"
        f"{'Valid':>10}"
        f"{'Test':>10}\n"
    )

    for scenario in sorted(scenario_summary):

        c = scenario_summary[scenario]
        total = (
            c["Training"]
            + c["Validation"]
            + c["Testing"]
        )

        f.write(
            f"{scenario:<12}"
            f"{total:>8}"
            f"{c['Training']:>10}"
            f"{c['Validation']:>10}"
            f"{c['Testing']:>10}\n"
        )

    f.write("\nIMPORTANT NOTES\n")
    f.write("-" * 60 + "\n")
    f.write(
        "1. Complete CSV simulations are kept in exactly one "
        "dataset to avoid time-series leakage.\n"
    )
    f.write(
        "2. Scenario names are preserved as metadata and are "
        "not used as model inputs by this splitter.\n"
    )
    f.write(
        "3. Single-simulation scenarios are handled according "
        "to SINGLE_FILE_POLICY.\n"
    )
    f.write(
        "4. Leak probability, cause, and severity are NOT "
        "invented by this script. Those labels must be established "
        "from the source data before model training.\n"
    )


# ============================================================
# 8. FINAL OUTPUT
# ============================================================

print("Dataset split complete.")
print(f"Output:   {output_path.resolve()}")
print(f"Manifest: {manifest_path.resolve()}")
print(f"Report:   {report_path.resolve()}")
print()
print(f"Training:   {dataset_totals['Training']}")
print(f"Validation: {dataset_totals['Validation']}")
print(f"Testing:    {dataset_totals['Testing']}")
