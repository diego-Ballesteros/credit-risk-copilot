"""Measure whether the model treats demographic groups differently. It measures; it does not fix.

Run it with::

    uv run python scripts/run_fairness_analysis.py

**Why this exists.** The production model consumes `SEX`, `EDUCATION`, `MARRIAGE` and `AGE`,
and until now nobody had measured whether it refuses some groups more than others. SHAP said
the demographic columns carry 4.5% of the total attribution, and that is **not** evidence of
absence: a small weight applied to a large population can still be unequal treatment. This
script produces the numbers that were missing.

**It quantifies and does not mitigate.** Reweighting, thresholding per group, or dropping a
variable are each decisions with alternatives and costs, and they belong to whoever reads
this. What is here is the measurement they would need.

---

THE TWO DEFINITIONS, WRITTEN OUT, BECAUSE THE WORDS ARE USED FOR DIFFERENT THINGS
---------------------------------------------------------------------------------

Notation: `A` is the group, `Y = 1` means the client really defaulted, and `Yhat = 1` means
the model scores the client at or above the operating threshold, i.e. **recommends refusing
them**. Refusal is the adverse outcome here.

**Demographic parity** (also statistical parity, and the basis of the "four-fifths rule"):

    P(Yhat = 1 | A = a)  equal for every a

Reported two ways: the **difference** `max_a - min_a`, and the **ratio** `min_a / max_a`,
which is the disparate-impact ratio that US guidance conventionally flags below 0.8. It
asks only whether groups are refused at the same rate, and deliberately **ignores whether
they default at the same rate**. That is its strength and its weakness: a group that really
does default more will fail demographic parity even from a perfect model.

**Equal opportunity** (Hardt, Price and Srebro, 2016):

    P(Yhat = 1 | Y = 1, A = a)  equal for every a

That is the **true positive rate** per group: among the clients who really did default, the
share the model caught. Unlike demographic parity it conditions on the truth, so a group
with a genuinely higher default rate does not fail it automatically.

**The false positive rate is reported beside it, and for this problem it is the one that
carries the human harm**:

    P(Yhat = 1 | Y = 0, A = a)

Among clients who **would have paid**, the share wrongly refused. Equal opportunity and
equal false-positive rate together are *equalised odds*. A reader who wants one number for
"is somebody being treated unfairly" should look at the false-positive rate gap: it counts
people who did nothing wrong and were refused anyway.

---

WHY THE BASE RATE IS IN EVERY TABLE
-----------------------------------

A difference in refusal rates has two possible causes and they call for opposite responses.
If a group defaults more often, refusing it more often is the model doing its job. If a
group defaults at the same rate and is refused more often, that is unequal treatment. The
tables carry the observed default rate next to the refusal rate so the two cannot be
confused, and the error-rate columns are what separate them.

**Every rate carries its group size.** A rate computed on 54 rows has a standard error of
about six percentage points and cannot support any conclusion; the summary statistics are
therefore computed only over groups above a stated floor, and the excluded groups are named
rather than dropped silently.

Exit code 0 when the measurement completed and was recorded, 1 when the tracking server
could not be configured.
"""

import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import (
    PRODUCTION_CALIBRATION_METHOD,
    PRODUCTION_OPERATING_THRESHOLD,
    build_calibrated_forest,
)
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    PRACTICAL_SIGNIFICANCE_THRESHOLD,
    cross_val_probabilities,
    cross_validate_estimator,
    split_features_and_target,
)
from credit_copilot.models.feature_groups import ALL_SOURCE_COLUMNS, SelectFeatureGroup
from credit_copilot.models.metrics import DECISION_METRIC
from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

FAIRNESS_TAG_VALUE: Final[str] = "fairness-analysis"
"""Marks the run that holds the fairness measurement."""

PROTECTED_COLUMNS: Final[tuple[str, ...]] = ("SEX", "EDUCATION", "MARRIAGE", "AGE")
"""The attributes this analysis treats as protected.

`SEX` and `AGE` are protected characteristics in essentially every jurisdiction.
`EDUCATION` and `MARRIAGE` are not universally so, and are included because they are
demographic attributes of the person rather than of their conduct, and because a credit
decision that varies with marital status is worth being able to answer for. `LIMIT_BAL` is
demographic in `feature_groups` but is **not** here: a granted credit limit is a property of
the account, not of the person.
"""

MIN_GROUP_SIZE: Final[int] = 500
"""Smallest group whose rates enter the summary statistics.

Chosen from the arithmetic rather than by habit. The standard error of a proportion near
this dataset's prevalence is `sqrt(0.22 * 0.78 / n)`: about **1.9 percentage points at
n = 500**, 4.2 at n = 100 and 5.7 at n = 54. Below 500 the noise on a single group's rate is
the same size as the disparities being looked for, so those groups are printed - the rule is
that every rate travels with its size - and excluded from the max-minus-min statistics,
with their names listed.
"""

AGE_BAND_EDGES: Final[tuple[int, ...]] = (20, 29, 39, 49, 120)
"""Upper edges of the age bands, exclusive of the first.

Decade bands are the convention in credit scoring and they are not tuned to this result.
The top band is left open at 50+ rather than split further **because of a measurement**:
there are 2,341 clients aged 50-59 and only **339** aged 60 or over, and a band of 339 would
sit below `MIN_GROUP_SIZE` and could not support a conclusion. Merging is the honest
alternative to reporting a rate nobody should read.
"""

AGE_BAND_LABELS: Final[tuple[str, ...]] = ("21-29", "30-39", "40-49", "50+")
"""Names of the age bands, in order."""

FOUR_FIFTHS: Final[float] = 0.8
"""Disparate-impact ratio below which US guidance conventionally flags a selection rate.

Carried as a reference point and not as a rule of this project: it is a screening
convention from employment law, not a legal standard for credit in any jurisdiction this
project operates in, and it is reported so the reader can see where the numbers fall
against a yardstick people recognise.
"""

RULE: Final[str] = "=" * 104


def age_band(ages: pd.Series) -> pd.Series:
    """Bucket ages into the reporting bands.

    Args:
        ages: Client ages in years.

    Returns:
        The band label for each client, as an ordered categorical.
    """
    return pd.cut(ages, bins=list(AGE_BAND_EDGES), labels=list(AGE_BAND_LABELS))


def protected_groups(features: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Build the group label of every client, for each protected attribute.

    `EDUCATION` and `MARRIAGE` are reported on their **raw** codes rather than on the
    collapsed ones the pipeline uses. The collapse of ADR-0004 folds three undocumented
    education codes onto level 4, and folding them here would hide exactly the small groups
    this analysis is supposed to size.

    Args:
        features: The canonical predictors.

    Returns:
        Attribute name -> the group each row belongs to.
    """
    return {
        "SEX": features["SEX"].astype(str),
        "EDUCATION": features["EDUCATION"].astype(str),
        "MARRIAGE": features["MARRIAGE"].astype(str),
        "AGE": age_band(features["AGE"]).astype(str),
    }


def group_table(
    target: pd.Series,
    probabilities: pd.Series,
    groups: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Measure refusal, base rate and both error rates for every group of one attribute.

    Args:
        target: True labels, 1 meaning the client defaulted.
        probabilities: Out-of-fold probability of default.
        groups: Group label per row.
        threshold: Probability at or above which the model refuses.

    Returns:
        One row per group: size, refusal rate, observed default rate, false positive rate,
        false negative rate and true positive rate.
    """
    refused = probabilities.to_numpy() >= threshold
    defaulted = target.to_numpy() == 1

    rows = []
    for label in sorted(groups.unique()):
        member = (groups == label).to_numpy()
        n_member = int(member.sum())
        paid = member & ~defaulted
        bad = member & defaulted
        rows.append(
            {
                "group": label,
                "n": n_member,
                "share_of_book": n_member / len(target),
                "refusal_rate": float(refused[member].mean()),
                "base_default_rate": float(defaulted[member].mean()),
                "false_positive_rate": float(refused[paid].mean()) if paid.any() else np.nan,
                "true_positive_rate": float(refused[bad].mean()) if bad.any() else np.nan,
                "false_negative_rate": float(1 - refused[bad].mean()) if bad.any() else np.nan,
                "supports_conclusion": n_member >= MIN_GROUP_SIZE,
            }
        )
    return pd.DataFrame(rows)


def disparity(table: pd.DataFrame) -> Mapping[str, float]:
    """Summarise one attribute's table into the gaps the definitions above name.

    Computed only over groups that meet `MIN_GROUP_SIZE`, because a maximum taken over a
    group of 54 rows measures the noise of that group and nothing else.

    Args:
        table: The output of `group_table`.

    Returns:
        The demographic-parity difference and ratio, the equal-opportunity (true positive
        rate) gap, the false-positive-rate gap, and the spread of the base rates.
    """
    usable = table.loc[table["supports_conclusion"]]
    if len(usable) < 2:
        return {}

    def gap(column: str) -> float:
        return float(usable[column].max() - usable[column].min())

    return {
        "demographic_parity_difference": gap("refusal_rate"),
        "demographic_parity_ratio": float(
            usable["refusal_rate"].min() / usable["refusal_rate"].max()
        ),
        "equal_opportunity_difference": gap("true_positive_rate"),
        "false_positive_rate_difference": gap("false_positive_rate"),
        "base_rate_difference": gap("base_default_rate"),
        "groups_compared": float(len(usable)),
    }


def build_blind_estimator() -> Pipeline:
    """Build the same production model, restricted to columns that are not protected.

    The restriction happens on the preprocessor's **output**, exactly as the hypothesis
    contrast does it, and for the same reason: the preprocessor addresses every source
    column by name and builds the behaviour features out of the raw blocks, so a table with
    `EDUCATION` removed would fail rather than shrink. Selecting afterwards keeps one
    identical preprocessor in both arms, so the only difference between the two models is
    which columns reach the classifier.

    Returns:
        An unfitted pipeline whose classifier never sees a protected attribute.
    """
    kept = tuple(column for column in ALL_SOURCE_COLUMNS if column not in set(PROTECTED_COLUMNS))
    return Pipeline(
        [
            ("select", SelectFeatureGroup(kept)),
            ("model", build_calibrated_forest(PRODUCTION_CALIBRATION_METHOD)),
        ]
    )


def _print_table(attribute: str, table: pd.DataFrame) -> None:
    """Print one attribute's measurement, sizes first.

    Args:
        attribute: Name of the protected attribute.
        table: The output of `group_table`.
    """
    print(f"\n{attribute}")
    print("-" * 104)
    print(
        f"{'group':<10}{'n':>8}{'% book':>9}{'refused':>10}{'defaults':>10}"
        f"{'FPR':>9}{'TPR':>9}{'FNR':>9}   note"
    )
    for _, row in table.iterrows():
        note = "" if row["supports_conclusion"] else f"n < {MIN_GROUP_SIZE}: not conclusive"
        print(
            f"{row['group']:<10}{row['n']:>8,}{row['share_of_book']:>9.1%}"
            f"{row['refusal_rate']:>10.4f}{row['base_default_rate']:>10.4f}"
            f"{row['false_positive_rate']:>9.4f}{row['true_positive_rate']:>9.4f}"
            f"{row['false_negative_rate']:>9.4f}   {note}"
        )


def _print_disparity(attribute: str, summary: Mapping[str, float]) -> None:
    """Print the gaps for one attribute, with the four-fifths reference beside the ratio.

    Args:
        attribute: Name of the protected attribute.
        summary: The output of `disparity`.
    """
    if not summary:
        print(f"  {attribute}: fewer than two groups meet the size floor; no gap computed.")
        return
    flag = "" if summary["demographic_parity_ratio"] >= FOUR_FIFTHS else "  <- below 0.80"
    print(
        f"  {attribute:<12}"
        f"parity diff {summary['demographic_parity_difference']:>7.4f}   "
        f"ratio {summary['demographic_parity_ratio']:>6.4f}{flag:<16}"
        f"  equal-opp gap {summary['equal_opportunity_difference']:>7.4f}   "
        f"FPR gap {summary['false_positive_rate_difference']:>7.4f}   "
        f"base-rate gap {summary['base_rate_difference']:>7.4f}"
    )


def main() -> int:
    """Measure the disparities, compare against a model blind to the protected columns.

    Returns:
        0 on success, 1 if MLflow could not be configured.
    """
    enable_unicode_console()
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("FAIRNESS - measuring disparity between demographic groups")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}")
    print(f"Prevalence      : {target.mean():.6f}")
    print(f"Model           : production forest + {PRODUCTION_CALIBRATION_METHOD} calibration")
    print(f"Threshold       : {PRODUCTION_OPERATING_THRESHOLD:.3f}  (from the 5:1 cost matrix)")
    print(
        f"Probabilities   : out-of-fold, StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), ",
        end="",
    )
    print(f"random_state={settings.random_state}")
    print(f"Protected       : {', '.join(PROTECTED_COLUMNS)}")
    print(f"Size floor      : {MIN_GROUP_SIZE:,} rows to enter a summary statistic")

    print("\nScoring every row by a model that never saw it...")
    probabilities = cross_val_probabilities(
        build_calibrated_forest(PRODUCTION_CALIBRATION_METHOD), features, target
    )
    print(f"   scored {len(probabilities):,} rows")

    groups = protected_groups(features)
    tables = {
        attribute: group_table(target, probabilities, labels, PRODUCTION_OPERATING_THRESHOLD)
        for attribute, labels in groups.items()
    }

    print("\n" + RULE)
    print("THE FULL MODEL - one table per protected attribute")
    print(RULE)
    print(
        "refused = share the model would turn away | defaults = share that really defaulted\n"
        "FPR = of those who WOULD HAVE PAID, share wrongly refused (the human harm)\n"
        "TPR = of those who DID default, share caught (equal opportunity)"
    )
    for attribute, table in tables.items():
        _print_table(attribute, table)

    summaries = {attribute: disparity(table) for attribute, table in tables.items()}
    print("\n" + RULE)
    print(f"GAPS, over groups of at least {MIN_GROUP_SIZE:,} rows")
    print(RULE)
    for attribute, summary in summaries.items():
        _print_disparity(attribute, summary)

    excluded = {
        attribute: list(table.loc[~table["supports_conclusion"], "group"])
        for attribute, table in tables.items()
    }
    dropped = {k: v for k, v in excluded.items() if v}
    print(f"\nGroups below the size floor, excluded from the gaps above: {dropped or 'none'}")
    print("They are printed in the tables with their sizes; what they cannot do is anchor a")
    print("maximum or a minimum, because at those sizes the rate is mostly noise.")

    print("\n" + RULE)
    print("WHAT IF THE MODEL COULD NOT SEE THE PROTECTED COLUMNS?")
    print(RULE)
    print("Same preprocessor, same estimator, same folds, same seed. The classifier simply")
    print(f"never receives any column derived from {', '.join(PROTECTED_COLUMNS)}.\n")

    blind = build_blind_estimator()
    blind_probabilities = cross_val_probabilities(blind, features, target)
    blind_tables = {
        attribute: group_table(target, blind_probabilities, labels, PRODUCTION_OPERATING_THRESHOLD)
        for attribute, labels in groups.items()
    }
    blind_summaries = {a: disparity(t) for a, t in blind_tables.items()}

    full_result = cross_validate_estimator(
        build_calibrated_forest(PRODUCTION_CALIBRATION_METHOD), features, target
    )
    blind_result = cross_validate_estimator(blind, features, target)

    print(f"Performance, {DECISION_METRIC}, mean ± std over {DEFAULT_N_SPLITS} folds")
    print("-" * 104)
    print(
        f"  full model        {full_result.means[DECISION_METRIC]:.4f} "
        f"± {full_result.stds[DECISION_METRIC]:.4f}   "
        f"{full_result.n_features} columns"
    )
    print(
        f"  blind model       {blind_result.means[DECISION_METRIC]:.4f} "
        f"± {blind_result.stds[DECISION_METRIC]:.4f}   "
        f"{blind_result.n_features} columns"
    )
    cost = blind_result.means[DECISION_METRIC] - full_result.means[DECISION_METRIC]
    verdict = (
        "within noise" if abs(cost) < PRACTICAL_SIGNIFICANCE_THRESHOLD else "beyond the threshold"
    )
    print(
        f"  difference        {cost:+.4f}   threshold {PRACTICAL_SIGNIFICANCE_THRESHOLD:.2f}"
        f"   -> {verdict}"
    )

    print("\nGaps, full model against blind model")
    print("-" * 104)
    print(f"{'attribute':<12}{'parity diff':>26}{'FPR gap':>26}{'equal-opp gap':>26}")
    print(f"{'':<12}{'full':>12}{'blind':>14}{'full':>12}{'blind':>14}{'full':>12}{'blind':>14}")
    for attribute in tables:
        full_summary = summaries[attribute]
        blind_summary = blind_summaries[attribute]
        if not full_summary or not blind_summary:
            continue
        print(
            f"{attribute:<12}"
            f"{full_summary['demographic_parity_difference']:>12.4f}"
            f"{blind_summary['demographic_parity_difference']:>14.4f}"
            f"{full_summary['false_positive_rate_difference']:>12.4f}"
            f"{blind_summary['false_positive_rate_difference']:>14.4f}"
            f"{full_summary['equal_opportunity_difference']:>12.4f}"
            f"{blind_summary['equal_opportunity_difference']:>14.4f}"
        )
    print(
        "\nA gap that survives removing the columns is carried by proxies: the behaviour\n"
        "features correlate with the demographic ones, and dropping the label does not drop\n"
        "the information."
    )

    with mlflow.start_run(experiment_id=context.experiment_id, run_name="fairness-analysis") as run:
        mlflow.set_tags(
            {
                LEAKAGE_CHECK_TAG: FAIRNESS_TAG_VALUE,
                "phase": "02-modeling",
                "measures_only": "true - this run quantifies disparity and mitigates nothing",
                "probabilities": "out-of-fold",
            }
        )
        mlflow.log_params(
            {
                "protected_columns": ", ".join(PROTECTED_COLUMNS),
                "operating_threshold": str(PRODUCTION_OPERATING_THRESHOLD),
                "min_group_size": str(MIN_GROUP_SIZE),
                "age_bands": ", ".join(AGE_BAND_LABELS),
                "random_state": str(settings.random_state),
            }
        )
        mlflow.log_metrics(
            {
                **{
                    f"{attribute}_{name}": value
                    for attribute, summary in summaries.items()
                    for name, value in summary.items()
                },
                **{
                    f"blind_{attribute}_{name}": value
                    for attribute, summary in blind_summaries.items()
                    for name, value in summary.items()
                },
                "full_model_pr_auc": full_result.means[DECISION_METRIC],
                "blind_model_pr_auc": blind_result.means[DECISION_METRIC],
                "blindness_cost_pr_auc": cost,
            }
        )
        with tempfile.TemporaryDirectory() as staging:
            directory = Path(staging)
            for attribute, table in tables.items():
                table.to_csv(directory / f"fairness_full_{attribute}.csv", index=False)
            for attribute, table in blind_tables.items():
                table.to_csv(directory / f"fairness_blind_{attribute}.csv", index=False)
            for path in sorted(directory.iterdir()):
                mlflow.log_artifact(str(path))
        run_id = run.info.run_id

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    print(f"    fairness-analysis      run_id {run_id}")
    print("\nThis run measures. It changes nothing about the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
