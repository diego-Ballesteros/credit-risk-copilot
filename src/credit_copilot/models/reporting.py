"""Read measurements back out of MLflow, so a narrative never recomputes them.

**Why reading is the right operation and recomputing is the wrong one.** Every number this
project reports was produced by a script, under a stated protocol, and recorded with the run
that produced it. A notebook that recomputed those numbers to draw them would be running a
second implementation of the same measurement, and the day the two disagree the notebook
would be the one nobody checks. Worse, it would be *slower* and *less* trustworthy at the
same time: slower because it refits models that are already fitted, less trustworthy because
the figure would no longer be evidence of what was recorded - only of what the notebook
happened to compute at render time.

So the contract of this module is narrow: **fetch what was recorded, change nothing**. There
is no metric arithmetic here beyond arranging recorded values into a table. `metrics.py` owns
what a metric means and `evaluation.py` owns how it is measured; this module owns neither and
only knows how to find them again.

**What it does not do.** It never fits, never scores, never touches the dataset. A caller
that needs a number nobody recorded has found a gap in the scripts, and the fix is to record
it there rather than to compute it here.
"""

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mlflow
import pandas as pd

from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    ExperimentContext,
    ensure_experiment,
)

RUN_TYPE_TAG: Final[str] = LEAKAGE_CHECK_TAG
"""Tag that says what kind of measurement a run holds. Named once, here and in `tracking`."""

FOLD_METRICS_FILE: Final[str] = "fold_metrics.csv"
"""Per-fold table attached to every cross-validated run by `evaluate_and_log`."""


class RecordedRunNotFoundError(LookupError):
    """No recorded run matches the identification asked for."""


@dataclass(frozen=True)
class RecordedRun:
    """One measurement as MLflow stored it.

    Attributes:
        run_id: MLflow identifier, and the thing a report cites so a number can be traced.
        name: Run name, unique only within a run type.
        run_type: Value of the run-type tag: which experiment the run belongs to.
        metrics: Every metric recorded on the run.
        params: Every parameter recorded on the run.
        tags: Every tag, including the run type.
    """

    run_id: str
    name: str
    run_type: str
    metrics: Mapping[str, float]
    params: Mapping[str, str]
    tags: Mapping[str, str]

    def metric(self, name: str) -> float:
        """Read one recorded metric.

        Args:
            name: Metric key as it was logged.

        Returns:
            The recorded value.

        Raises:
            KeyError: If the run never recorded that metric, which is a different and more
                useful failure than silently returning a default.
        """
        if name not in self.metrics:
            raise KeyError(
                f"Run {self.name!r} ({self.run_id}) recorded no metric {name!r}. "
                f"It has: {sorted(self.metrics)}"
            )
        return float(self.metrics[name])


def load_recorded_runs(context: ExperimentContext | None = None) -> list[RecordedRun]:
    """Fetch every active run of the experiment.

    Args:
        context: Experiment to read. Defaults to the project's experiment, configuring
            MLflow as a side effect.

    Returns:
        Every run, in no guaranteed order. Deleted runs are not returned.
    """
    resolved = ensure_experiment() if context is None else context
    runs = mlflow.search_runs(experiment_ids=[resolved.experiment_id], output_format="list")
    return [
        RecordedRun(
            run_id=run.info.run_id,
            name=run.info.run_name or "",
            run_type=run.data.tags.get(RUN_TYPE_TAG, ""),
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            tags=dict(run.data.tags),
        )
        for run in runs
    ]


def find_run(runs: Sequence[RecordedRun], run_type: str, name: str) -> RecordedRun:
    """Locate one run by its type and name.

    Both are needed because a name is unique only inside a type: `logistic-l2-balanced`
    exists as a `baseline` and again as a `model-comparison`, deliberately, and they are two
    measurements of the same configuration taken in different turns.

    Args:
        runs: Runs to search, from `load_recorded_runs`.
        run_type: Value of the run-type tag.
        name: Run name.

    Returns:
        The matching run.

    Raises:
        RecordedRunNotFoundError: If nothing matches, or if more than one does.
    """
    matches = [run for run in runs if run.run_type == run_type and run.name == name]
    if not matches:
        available = sorted({(r.run_type, r.name) for r in runs if r.run_type == run_type})
        raise RecordedRunNotFoundError(
            f"No run of type {run_type!r} named {name!r}. Available in that type: {available}"
        )
    if len(matches) > 1:
        raise RecordedRunNotFoundError(
            f"{len(matches)} runs of type {run_type!r} are named {name!r}: "
            f"{[m.run_id for m in matches]}. A report cannot cite an ambiguous measurement."
        )
    return matches[0]


def read_run_artefact(run_id: str, filename: str) -> pd.DataFrame:
    """Download one CSV artefact of a run and read it.

    Args:
        run_id: Run that owns the artefact.
        filename: Artefact path within the run.

    Returns:
        The artefact as a table.
    """
    with tempfile.TemporaryDirectory() as staging:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=filename, dst_path=staging
        )
        return pd.read_csv(Path(local))


def fold_metrics(run: RecordedRun) -> pd.DataFrame:
    """Read the per-fold table a cross-validated run attached to itself.

    The per-fold values are what make a mean auditable, and they are the only way to see the
    case a mean hides: four folds agreeing and one disagreeing loudly.

    Args:
        run: The run.

    Returns:
        The per-fold table, indexed by fold, with the `mean` and `std` rows the script
        appended still present.
    """
    table = read_run_artefact(run.run_id, FOLD_METRICS_FILE)
    return table.set_index(table.columns[0])


def comparison_table(
    selection: Mapping[str, RecordedRun],
    metric_names: Sequence[str],
    decimals: int = 4,
) -> pd.DataFrame:
    """Arrange several recorded runs into one `mean ± std` table.

    Formatting only. Every value comes from `<metric>_mean` and `<metric>_std` exactly as the
    run recorded them; nothing is averaged or recomputed here.

    Args:
        selection: Column label -> run. Insertion order is column order.
        metric_names: Metrics to show, in row order.
        decimals: Digits after the point.

    Returns:
        A table of formatted strings, metrics down the rows and models across the columns.
    """
    return pd.DataFrame(
        {
            label: [
                f"{run.metric(f'{name}_mean'):.{decimals}f} "
                f"± {run.metric(f'{name}_std'):.{decimals}f}"
                for name in metric_names
            ]
            for label, run in selection.items()
        },
        index=list(metric_names),
    )


def metric_frame(
    selection: Mapping[str, RecordedRun],
    metric_names: Sequence[str],
) -> pd.DataFrame:
    """Arrange recorded means and standard deviations as numbers, for plotting.

    The numeric counterpart of `comparison_table`: same values, unformatted, so a chart can
    use them without parsing strings back out of a display table.

    Args:
        selection: Row label -> run.
        metric_names: Metrics to read.

    Returns:
        One row per label, with a `<metric>` and a `<metric>_std` column for each metric.
    """
    rows = {}
    for label, run in selection.items():
        row: dict[str, float] = {}
        for name in metric_names:
            row[name] = run.metric(f"{name}_mean")
            row[f"{name}_std"] = run.metric(f"{name}_std")
        rows[label] = row
    return pd.DataFrame(rows).T
