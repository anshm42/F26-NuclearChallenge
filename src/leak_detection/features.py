"""Convert one plant simulation CSV into leakage-safe tabular features."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# These fields directly measure a break, tube leak, or accumulated release. Using
# them would turn detection into target leakage rather than early inference.
DIRECT_LEAK_COLUMNS = frozenset(
    {
        "WLR",   # RCS leak flow
        "WTRA",  # steam-generator A tube leak flow
        "WTRB",  # steam-generator B tube leak flow
        "WBK",   # total break flow entering reactor building
        "RBLK",  # accumulated reactor-building leakage
        "SGLK",  # accumulated steam-generator leakage
        "MBK",   # integrated break flow
        "EBK",   # integrated break energy
        "WFLB",  # feedwater-line break flow
    }
)

# NPPAD scenarios whose defining event is a coolant, steam, feedwater, or
# letdown-line break/rupture. Override this list from the CLI if the project uses
# a narrower operational definition of "leak".
DEFAULT_LEAK_CLASSES = frozenset(
    {"LOCA", "LOCAC", "SGATR", "SGBTR", "FLB", "LLB", "SLBIC", "SLBOC"}
)

SUMMARY_STATS = ("last", "mean", "std", "min", "max", "delta", "slope")


@dataclass(frozen=True)
class FeatureConfig:
    """Settings that must be identical during training and evaluation."""

    max_time_seconds: float = 120.0
    time_column: str = "TIME"
    excluded_columns: tuple[str, ...] = tuple(sorted(DIRECT_LEAK_COLUMNS))


def discover_runs(split_directory: str | Path) -> list[Path]:
    """Return all simulation CSVs below a split directory."""

    root = Path(split_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {root}")

    runs = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if not runs:
        raise ValueError(f"No CSV simulation files found below: {root}")
    return runs


def scenario_from_path(path: Path) -> str:
    """Infer the NPPAD scenario from the CSV's immediate parent directory."""

    return path.parent.name.upper()


def extract_run_features(path: str | Path, config: FeatureConfig) -> pd.Series:
    """Summarize an early time window from one simulation.

    The model receives distribution and trend features rather than individual
    timestamps. This keeps each complete simulation as one independent sample.
    """

    path = Path(path)
    frame = pd.read_csv(path, low_memory=False)
    if config.time_column not in frame.columns:
        raise ValueError(f"{path} has no {config.time_column!r} column")

    time = pd.Series(
        pd.to_numeric(frame[config.time_column], errors="coerce"),
        index=frame.index,
        dtype=float,
    )
    valid_time = time.notna()
    frame = frame.loc[valid_time].copy()
    frame[config.time_column] = time.loc[valid_time]
    frame = frame.sort_values(config.time_column)
    frame = frame.loc[frame[config.time_column] <= config.max_time_seconds]
    if frame.empty:
        raise ValueError(
            f"{path} has no observations at or before {config.max_time_seconds}s"
        )

    excluded = {config.time_column, *config.excluded_columns}
    sensor_columns = [column for column in frame.columns if column not in excluded]
    sensors = frame[sensor_columns].apply(pd.to_numeric, errors="coerce")
    sensors = sensors.replace([np.inf, -np.inf], np.nan)
    if sensors.shape[1] == 0:
        raise ValueError(f"{path} has no usable sensor columns")

    times = frame[config.time_column].to_numpy(dtype=float)
    centered_time = times - np.mean(times)
    time_denominator = float(np.dot(centered_time, centered_time))

    result: dict[str, float] = {}
    for column in sensors.columns:
        values = sensors[column]
        valid = values.notna().to_numpy()
        valid_values = values.to_numpy(dtype=float)[valid]
        if valid_values.size == 0:
            for statistic in SUMMARY_STATS:
                result[f"{column}__{statistic}"] = np.nan
            continue

        first = float(valid_values[0])
        last = float(valid_values[-1])
        result[f"{column}__last"] = last
        result[f"{column}__mean"] = float(np.mean(valid_values))
        result[f"{column}__std"] = (
            float(np.std(valid_values, ddof=1)) if valid_values.size > 1 else 0.0
        )
        result[f"{column}__min"] = float(np.min(valid_values))
        result[f"{column}__max"] = float(np.max(valid_values))
        result[f"{column}__delta"] = last - first

        valid_times = times[valid]
        centered_valid_time = valid_times - np.mean(valid_times)
        denominator = float(np.dot(centered_valid_time, centered_valid_time))
        if valid_values.size < 2 or denominator == 0.0 or time_denominator == 0.0:
            result[f"{column}__slope"] = 0.0
        else:
            centered_values = valid_values - np.mean(valid_values)
            result[f"{column}__slope"] = float(
                np.dot(centered_valid_time, centered_values) / denominator
            )

    return pd.Series(result, dtype=float)


def load_split(
    split_directory: str | Path,
    positive_classes: Iterable[str],
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load a split without allowing rows from a run to become separate samples."""

    positive = {name.upper() for name in positive_classes}
    paths = discover_runs(split_directory)
    feature_rows: list[pd.Series] = []
    labels: list[int] = []
    metadata: list[dict[str, str]] = []

    for path in paths:
        scenario = scenario_from_path(path)
        feature_rows.append(extract_run_features(path, config))
        labels.append(int(scenario in positive))
        metadata.append({"path": str(path), "scenario": scenario})

    features = pd.DataFrame(feature_rows).reset_index(drop=True)
    targets = pd.Series(labels, name="is_leak", dtype=int)
    run_metadata = pd.DataFrame(metadata)
    return features, targets, run_metadata
