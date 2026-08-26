"""Run the preprocessing pipeline end to end and write the exploration artefacts.

Run it with:

    uv run python scripts/run_preprocessing.py

It loads the canonical table, validates it against the contract, fits the pipeline,
transforms, and writes three files to `data/processed/`: the feature matrix, the target,
and the fitted pipeline.

===========================================================================
READ THIS BEFORE USING `features.parquet` FOR ANYTHING
===========================================================================

**This script fits the pipeline on the whole dataset, and that is correct here and
leakage anywhere else.**

The pipeline learns four statistics from the data it is fitted on: the payment-ratio
clipping thresholds, the imputation medians, the scaler's median and interquartile range,
and the one-hot vocabulary. Fitting on all 30,000 rows means every one of those numbers
was computed with every row in view, this row included.

That is fine for **exploration**: a notebook that plots the distribution of a scaled
feature, or checks how many columns the one-hot produced, is describing the dataset it
has, and there is no held-out set whose independence could be violated.

It is **leakage** the moment this matrix is used to estimate how a model would perform on
data it has not seen. A model trained and evaluated on `features.parquet` would be
evaluated on rows whose scaling was informed by their own values, and the resulting metric
would be optimistic by an amount nobody can bound afterwards.

**The correct use for training is the pipeline artefact, not the matrix.** Put
`preprocessor.joblib` - or a fresh `build_preprocessor()` - inside the cross-validation as
a step, so that its `fit` runs once per fold on that fold's training part only. Section
6.5 of `docs/METHODOLOGY.md` explains why this has to be enforced by the tool rather than
remembered: `Pipeline` makes it impossible to scale before the split, and a document
asking for it does not.

The exit code is the verdict of the contract check: 0 when the table honours it, 1 when a
blocking finding needs a human decision. A script that reports a problem and returns
success teaches everyone to stop reading its output.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from credit_copilot.config import settings
from credit_copilot.data import schema
from credit_copilot.data.loader import load_dataset
from credit_copilot.data.preprocessor import (
    CATEGORICAL_COLUMNS,
    INDICATOR_COLUMNS,
    NUMERIC_COLUMNS,
    build_preprocessor,
    learned_parameters,
)
from credit_copilot.data.validator import validate_dataframe

FEATURES_FILENAME = "features.parquet"
"""The transformed matrix: model-ready columns, and no target."""

TARGET_FILENAME = "target.parquet"
"""The target, written apart so that a careless read of the matrix cannot pick it up."""

PIPELINE_FILENAME = "preprocessor.joblib"
"""The fitted pipeline: the artefact the notebook, the training script and the API share."""


def _write_artefacts(
    features: pd.DataFrame,
    target: pd.Series,
    pipeline: object,
    destination: Path,
) -> dict[str, Path]:
    """Write the three artefacts to `data/processed/`.

    The target goes to its own file rather than into the matrix. Keeping them apart is the
    same reasoning that removes `ID` at load time: a column that must not be a feature is
    safest when it is not in the table the feature code reads.

    Args:
        features: The transformed matrix.
        target: The label, sharing the index of `features`.
        pipeline: The fitted pipeline.
        destination: Directory to write into. Created if absent.

    Returns:
        Artefact name -> path written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": destination / FEATURES_FILENAME,
        "target": destination / TARGET_FILENAME,
        "pipeline": destination / PIPELINE_FILENAME,
    }
    features.to_parquet(paths["features"], index=True)
    target.to_frame().to_parquet(paths["target"], index=True)
    joblib.dump(pipeline, paths["pipeline"])
    return paths


def _print_group(title: str, names: list[str], per_line: int = 4) -> None:
    """Print a named group of output columns, wrapped.

    Args:
        title: Heading for the group.
        names: Column names to print.
        per_line: How many names to put on one line.
    """
    print(f"\n{title} ({len(names)})")
    for start in range(0, len(names), per_line):
        print("    " + "  ".join(f"{name:<32}" for name in names[start : start + per_line]))


def main() -> int:
    """Load, validate, fit, transform, write and report.

    Returns:
        0 if the data honours the contract, 1 if any blocking finding was produced.
    """
    frame = load_dataset()
    print(f"Loaded: {frame.shape[0]:,} rows x {frame.shape[1]} canonical columns")

    result = validate_dataframe(frame)
    print(result.report())
    if not result.is_valid:
        print("\nBlocking findings above. Nothing was written.")
        return 1

    print("\n" + "=" * 75)
    print("FITTING ON THE FULL DATASET - an exploration artefact, not a training set.")
    print("Read the module docstring of this script before using features.parquet.")
    print("=" * 75)

    target = frame[schema.TARGET_COLUMN]
    pipeline = build_preprocessor()
    features = pipeline.fit_transform(frame)

    names = list(pipeline.get_feature_names_out())
    if names != list(features.columns):
        print("\nThe declared feature names do not match the produced columns.")
        return 1

    paths = _write_artefacts(features, target, pipeline, settings.processed_data_dir)

    print(f"\nMatrix shape : {features.shape[0]:,} rows x {features.shape[1]} columns")
    print(f"Features     : {len(names)}")
    print(f"Missing values in the matrix: {int(features.isna().sum().sum())}")

    learned = learned_parameters(pipeline)
    print("\nPayment-ratio caps learned in fit:")
    for column, bound in learned["clip"].items():
        print(f"    {column:<20} <= {bound:.4f}")

    constant = [name for name in names if features[name].nunique(dropna=False) == 1]
    print(f"\nConstant columns in the matrix: {len(constant)}")
    for name in constant:
        print(f"    {name} (single value: {features[name].iloc[0]})")

    encoder = pipeline.named_steps["columns"].named_transformers_["categorical"]
    print("\nOne-hot expansion, source column -> levels found in fit:")
    for column, levels in zip(CATEGORICAL_COLUMNS, encoder.categories_, strict=True):
        print(f"    {column:<15} {len(levels):>2} -> {sorted(int(level) for level in levels)}")

    groups = {
        "categorical (one-hot)": [n for n in names if n not in NUMERIC_COLUMNS + INDICATOR_COLUMNS],
        "numeric (imputed + robust-scaled)": [n for n in names if n in NUMERIC_COLUMNS],
        "indicators (passthrough)": [n for n in names if n in INDICATOR_COLUMNS],
    }
    for title, group in groups.items():
        _print_group(title, group)

    print("\nWritten:")
    for label, path in paths.items():
        print(f"    {label:<9} {path}")

    print(
        "\nReminder: to train or evaluate, put the pipeline inside the cross-validation "
        "so it is fitted once per fold. Do not train on features.parquet."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
