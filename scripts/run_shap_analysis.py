"""Explain the production model with SHAP, on the pipeline's own feature names.

Run it with::

    uv run python scripts/run_shap_analysis.py

**Why the feature names are checked before anything is drawn.** A `ColumnTransformer` that
loses its names labels its output positionally, and SHAP will then happily attribute an
explanation to whichever variable happens to sit at that index. The result is not a missing
explanation - it is a plausible, well-formatted, *wrong* story about why a client was
refused, and nothing in the plot looks off. This script asserts that the names the
preprocessor declares are the names the model was fitted on and the names SHAP received,
and refuses to produce a single artefact if any of the three disagree.

**What model is explained, and why that is the right one.** SHAP runs on the uncalibrated
`build_production_forest`, not on the calibrated wrapper. That is deliberate and it costs
nothing: the production calibration is a sigmoid, which is *strictly* increasing, so it
changes the value of the probability without changing the order of any two clients or the
relative contribution of any feature to that order. Explaining the forest is explaining the
ranking the deployed model serves. It is also the only option that is well defined - a
calibrated wrapper is not a tree ensemble and `TreeExplainer` cannot see into it.

**Why the model is fitted on all 30,000 rows here.** This script measures nothing. There is
no held-out metric to protect, so there is no partition to respect, and an explanation of
the deployed artefact should be an explanation of the artefact that will actually be
deployed. The same reasoning is written out at length in `register_production_model.py`.

Exit code 0 when every artefact was produced and recorded, 1 when the tracking server could
not be configured, 2 when the feature names do not line up.
"""

import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from credit_copilot.config import settings  # noqa: E402
from credit_copilot.console import enable_unicode_console  # noqa: E402
from credit_copilot.data.loader import load_dataset  # noqa: E402
from credit_copilot.data.preprocessor import build_preprocessor  # noqa: E402
from credit_copilot.models.estimators import (  # noqa: E402
    PRODUCTION_OPERATING_THRESHOLD,
    build_production_forest,
)
from credit_copilot.models.evaluation import split_features_and_target  # noqa: E402
from credit_copilot.models.feature_groups import (  # noqa: E402
    BEHAVIOURAL_SOURCE_COLUMNS,
    DEMOGRAPHIC_SOURCE_COLUMNS,
    group_columns_by_source,
)
from credit_copilot.models.tracking import (  # noqa: E402
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

SHAP_TAG_VALUE: Final[str] = "shap-analysis"
"""Marks the run that holds the explanation artefacts."""

EXPLAIN_SAMPLE_SIZE: Final[int] = 3000
"""Rows used for the global summary.

The beeswarm shows a distribution, and 3,000 rows describe it as well as 30,000 while
keeping the plot readable and the computation in seconds. The sample is drawn with the
project seed, so the figure is reproducible. Individual explanations are computed on their
own rows and are not affected by this.
"""

TOP_FEATURES_TO_REPORT: Final[int] = 10
"""How many features the report lists by mean absolute SHAP value."""

DEPENDENCE_PLOTS: Final[int] = 5
"""How many of the top features get a dependence plot."""

POSITIVE_CLASS_INDEX: Final[int] = 1
"""Column of the SHAP output that explains the probability of default."""

RULE: Final[str] = "=" * 100


class FeatureNameMismatchError(RuntimeError):
    """The names declared by the pipeline are not the names SHAP was handed."""


def assert_names_line_up(
    declared: Sequence[str],
    matrix: pd.DataFrame,
    model_names: Sequence[str],
    shap_names: Sequence[str],
) -> None:
    """Refuse to continue unless all four views of the feature names agree.

    Args:
        declared: What `get_feature_names_out` says the pipeline produces.
        matrix: The transformed matrix the model was fitted on.
        model_names: What the fitted estimator recorded in `feature_names_in_`.
        shap_names: What the SHAP explanation carries.

    Raises:
        FeatureNameMismatchError: If any of the four disagree.
    """
    views = {
        "pipeline.get_feature_names_out": list(declared),
        "transformed matrix columns": list(matrix.columns),
        "model.feature_names_in_": list(model_names),
        "shap explanation.feature_names": list(shap_names),
    }
    reference = views["pipeline.get_feature_names_out"]
    disagreeing = [name for name, value in views.items() if value != reference]
    if disagreeing:
        detail = "; ".join(f"{name}: {len(views[name])} names" for name in views)
        raise FeatureNameMismatchError(
            "The feature names do not line up across the pipeline, the model and SHAP: "
            f"{disagreeing} differ from the pipeline's own list. Sizes were {detail}. "
            "Refusing to draw anything: an explanation attributed to the wrong variable "
            "is worse than no explanation, because it is readable and confident."
        )


def rank_features(values: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Mean *absolute* value, because a feature that pushes some clients up and others down by
    the same amount matters to the model and would average to zero.

    Args:
        values: SHAP values for the positive class, one row per explained sample.
        names: Feature names, aligned to the columns of `values`.

    Returns:
        Features in descending order of importance, with their mean signed value beside it
        so the direction is not lost.
    """
    frame = pd.DataFrame(
        {
            "feature": list(names),
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    )
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def choose_cases(probabilities: pd.Series, threshold: float) -> Mapping[str, int]:
    """Pick three clients worth explaining individually.

    The three are chosen by criterion rather than at random: the model's most confident
    refusal, its most confident approval, and the client sitting closest to the operating
    threshold - which is the one where the decision is genuinely uncertain and where an
    explanation has to carry the most weight.

    Args:
        probabilities: The model's predicted probability of default for every row.
        threshold: The operating threshold.

    Returns:
        Case label -> positional index into the explained matrix.
    """
    return {
        "highest-risk": int(np.argmax(probabilities.to_numpy())),
        "lowest-risk": int(np.argmin(probabilities.to_numpy())),
        "at-the-threshold": int(np.argmin(np.abs(probabilities.to_numpy() - threshold))),
    }


def _save(figure_path: Path) -> None:
    """Write the current matplotlib figure and close it.

    Args:
        figure_path: File to write.
    """
    plt.tight_layout()
    plt.savefig(figure_path, dpi=130, bbox_inches="tight")
    plt.close("all")


def _group_share(ranked: pd.DataFrame, matrix_columns: Sequence[str]) -> Mapping[str, float]:
    """Split the total attribution between the demographic and behavioural groups.

    This is the SHAP counterpart of the hypothesis contrast in entry 003: that measured how
    much each group of columns was *worth* to the metric, this measures how much the model
    actually *uses* them. The two are different questions and agreeing is not guaranteed.

    Args:
        ranked: The output of `rank_features`.
        matrix_columns: Every column of the explained matrix.

    Returns:
        Group name -> share of the total mean absolute SHAP value.
    """
    owners = group_columns_by_source(list(matrix_columns))
    demographic = {c for source in DEMOGRAPHIC_SOURCE_COLUMNS for c in owners.get(source, [])}
    behavioural = {c for source in BEHAVIOURAL_SOURCE_COLUMNS for c in owners.get(source, [])}
    total = float(ranked["mean_abs_shap"].sum())
    return {
        "demographic": float(
            ranked.loc[ranked["feature"].isin(demographic), "mean_abs_shap"].sum() / total
        ),
        "behavioural": float(
            ranked.loc[ranked["feature"].isin(behavioural), "mean_abs_shap"].sum() / total
        ),
    }


def main() -> int:
    """Fit the final model, explain it, and record every artefact.

    Returns:
        0 on success, 1 if MLflow could not be configured, 2 if the names do not line up.
    """
    enable_unicode_console()
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("SHAP - explaining the production model")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}")
    print("Model explained : build_production_forest(), uncalibrated")
    print("                  sigmoid calibration is strictly increasing, so it moves the")
    print("                  probability without moving the order this explains")

    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(features, target)
    declared = list(preprocessor.get_feature_names_out())

    forest = build_production_forest()
    forest.fit(matrix, target)
    probabilities = pd.Series(
        forest.predict_proba(matrix)[:, POSITIVE_CLASS_INDEX], index=matrix.index
    )
    print(f"Matrix          : {matrix.shape[0]:,} rows x {matrix.shape[1]} columns")

    sample = matrix.sample(
        n=min(EXPLAIN_SAMPLE_SIZE, len(matrix)), random_state=settings.random_state
    )
    explainer = shap.TreeExplainer(forest)
    explanation = explainer(sample)

    print("\n" + RULE)
    print("FEATURE NAMES - checked before anything is drawn")
    print(RULE)
    try:
        assert_names_line_up(declared, sample, forest.feature_names_in_, explanation.feature_names)
    except FeatureNameMismatchError as error:
        print(f"STOPPING.\n{error}", file=sys.stderr)
        return 2
    print(f"  pipeline.get_feature_names_out : {len(declared)} names")
    print("  transformed matrix columns     : identical")
    print("  model.feature_names_in_        : identical")
    print("  shap explanation.feature_names : identical")
    print("  All four agree. Every attribution below belongs to the variable it names.")

    values = explanation.values[..., POSITIVE_CLASS_INDEX]
    ranked = rank_features(values, explanation.feature_names)

    print("\n" + RULE)
    print(f"THE {TOP_FEATURES_TO_REPORT} MOST IMPORTANT FEATURES, by mean |SHAP|")
    print(RULE)
    print(f"{'#':<4}{'feature':<36}{'mean |SHAP|':>14}{'mean SHAP':>14}{'  direction':<12}")
    for position, row in ranked.head(TOP_FEATURES_TO_REPORT).iterrows():
        direction = "raises risk" if row["mean_shap"] > 0 else "lowers risk"
        print(
            f"{position + 1:<4}{row['feature']:<36}{row['mean_abs_shap']:>14.6f}"
            f"{row['mean_shap']:>+14.6f}  {direction:<12}"
        )

    shares = _group_share(ranked, matrix.columns)
    print("\nAttribution split between the two groups of entry 003")
    print("-" * 100)
    print(f"  behavioural columns : {shares['behavioural']:.1%} of total mean |SHAP|")
    print(f"  demographic columns : {shares['demographic']:.1%}")
    print(
        "\nEntry 003 measured what each group is *worth* to the metric. This measures how\n"
        "much the model actually *uses* it. They are different questions."
    )

    cases = choose_cases(probabilities, PRODUCTION_OPERATING_THRESHOLD)
    print(f"\nIndividual cases (operating threshold {PRODUCTION_OPERATING_THRESHOLD:.3f})")
    print("-" * 100)
    for label, position in cases.items():
        print(
            f"  {label:<20} row {matrix.index[position]:<8} "
            f"predicted probability {probabilities.iloc[position]:.4f}   "
            f"actual: {'default' if target.iloc[position] == 1 else 'paid'}"
        )

    with mlflow.start_run(experiment_id=context.experiment_id, run_name="shap-analysis") as run:
        mlflow.set_tags(
            {
                LEAKAGE_CHECK_TAG: SHAP_TAG_VALUE,
                "phase": "02-modeling",
                "explained_model": "build_production_forest (uncalibrated)",
                "feature_names_verified": "true",
            }
        )
        mlflow.log_params(
            {
                "explain_sample_size": str(len(sample)),
                "n_features": str(matrix.shape[1]),
                "random_state": str(settings.random_state),
                "operating_threshold": str(PRODUCTION_OPERATING_THRESHOLD),
            }
        )
        mlflow.log_metrics(
            {
                "behavioural_attribution_share": shares["behavioural"],
                "demographic_attribution_share": shares["demographic"],
            }
        )

        with tempfile.TemporaryDirectory() as staging:
            directory = Path(staging)

            ranked.to_csv(directory / "shap_feature_importance.csv", index=False)

            shap.plots.beeswarm(explanation[..., POSITIVE_CLASS_INDEX], max_display=20, show=False)
            _save(directory / "shap_summary_beeswarm.png")

            shap.plots.bar(explanation[..., POSITIVE_CLASS_INDEX], max_display=20, show=False)
            _save(directory / "shap_summary_bar.png")

            for rank, feature in enumerate(ranked["feature"].head(DEPENDENCE_PLOTS), start=1):
                shap.plots.scatter(explanation[:, feature, POSITIVE_CLASS_INDEX], show=False)
                _save(directory / f"shap_dependence_{rank}_{feature}.png")

            for label, position in cases.items():
                single = explainer(matrix.iloc[[position]])
                shap.plots.waterfall(single[0, :, POSITIVE_CLASS_INDEX], max_display=15, show=False)
                _save(directory / f"shap_waterfall_{label}.png")

            for path in sorted(directory.iterdir()):
                mlflow.log_artifact(str(path))
            produced = sorted(p.name for p in directory.iterdir())

        run_id = run.info.run_id

    print("\n" + RULE)
    print("ARTEFACTS")
    print(RULE)
    for name in produced:
        print(f"    {name}")

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    print(f"    shap-analysis          run_id {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
