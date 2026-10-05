# F26 Nuclear Challenge — Leak Probability Model

This project trains a calibrated XGBoost classifier to estimate the probability that one NPPAD simulation represents a leak/break event.

> This is a research prototype built from simulated data. It is not a validated nuclear safety system and must not be used as the sole basis for operational decisions.

## Data split contract

Split **whole simulation CSV files**, never individual timestamp rows. Keeping rows from one run in different splits would leak the same trajectory into training and evaluation.

Expected layout:

```text
data/
├── train/
│   ├── LOCA/*.csv
│   ├── Normal/*.csv
│   └── ...
├── validation/
│   └── <scenario>/*.csv
└── test/
    └── <scenario>/*.csv
```

The model infers the scenario label from each CSV's immediate parent folder. Keep corresponding operation/dose files or related variants from the same simulation in the same split. This first model is intended for `Operation_csv_data`; dose data has a different sensor schema and should be modeled separately or deliberately joined by run ID.

Default positive leak/break scenarios are:

```text
LOCA, LOCAC, SGATR, SGBTR, FLB, LLB, SLBIC, SLBOC
```

Change `--positive-classes` if your team adopts a narrower definition.

## Leakage controls

Each CSV becomes one sample. The first 1,500 seconds are summarized with last value, mean, standard deviation, minimum, maximum, change, and linear trend for each sensor.

Direct answer fields are excluded by default:

```text
WLR, WTRA, WTRB, WBK, RBLK, SGLK, MBK, EBK, WFLB
```

These variables directly encode simulated break/leak flow or accumulated leakage and would produce misleadingly optimistic results. Add other unavailable or post-alarm sensors with `--exclude-columns`.

## Setup

Use Python 3.10 or newer. On macOS, install XGBoost's OpenMP runtime first:

```bash
brew install libomp
```

Then create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Train and select threshold

Training reads only the training and validation directories. XGBoost is probability-calibrated using folds within the training data. Validation selects the highest-precision alert threshold that reaches the requested minimum recall.

```bash
leak-model train \
  --train-dir data/train \
  --validation-dir data/validation \
  --output artifacts/leak_model.joblib \
  --minimum-recall 0.95 \
  --max-time-seconds 1500
```

This writes the model artifact plus validation metrics and per-run predictions. Do not tune the model after inspecting final test results.

## Final held-out evaluation

Run this separately, once the feature window, leak definition, model, and threshold are frozen:

```bash
leak-model evaluate \
  --model artifacts/leak_model.joblib \
  --test-dir data/test \
  --output-dir reports/final_test
```

Reported metrics include ROC AUC, average precision, Brier score (probability calibration), precision, recall, and the full confusion matrix counts. For leak detection, pay particular attention to false negatives and recall rather than accuracy alone.

## Tests

```bash
python -m unittest discover -s tests
```
