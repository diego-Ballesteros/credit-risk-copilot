"""Train the final model on all the data and register it in the MLflow Model Registry.

Run it with::

    uv run python scripts/register_production_model.py

===========================================================================
WHY FITTING ON ALL 30,000 ROWS IS CORRECT HERE AND LEAKAGE ANYWHERE ELSE
===========================================================================

**The rule this appears to break.** Every other script in this project fits the
preprocessor and the model strictly inside a cross-validation fold, and
`scripts/run_preprocessing.py` carries a warning that `data/processed/features.parquet` -
fitted on all 30,000 rows - must never be used to train or evaluate. This script fits on
all 30,000 rows deliberately. The two are not in tension, and the reason is worth stating
precisely rather than waving at.

**The rule is about *estimating* performance, not about *fitting* a model.** A metric is a
claim about how a model behaves on data it has not seen. That claim is only true if some
data was genuinely held out, which is why the preprocessor goes inside the fold: a scaler
that has seen a validation row has smuggled information about that row into the number the
row later produces, and the metric becomes optimistic by an amount nobody can bound.

**This script produces no metric.** It attaches metrics that were already measured, by
cross-validation, by other scripts, on models fitted only on training folds. Nothing here
is estimated from the fit performed here. The fit exists to produce the *artefact*, and for
that purpose withholding a fifth of the data would simply make the deployed model worse for
no gain: the held-out fifth would buy an estimate this script is not making.

**Where the leakage would actually be.** It would be in reporting `predict_proba` on these
same 30,000 rows as a performance figure, and this script never does that. The metrics
logged to the registry are the cross-validated ones, tagged with the runs they came from.

**Concretely, what is registered.** The whole `Pipeline` - preprocessor plus calibrated
forest - as one object, because section 6.3 of `docs/METHODOLOGY.md` requires the notebook,
the training script and the API to load *the same artefact* rather than three copies of the
same arithmetic. The signature is inferred from the 23 raw canonical columns, so a consumer
feeds the model what `loader.load_dataset` produces and not a matrix it would have to build
itself.

Exit code 0 when the model was registered, 1 when the tracking server could not be
configured, 2 when the registry refused the model.
"""

import sys
from typing import Final

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from credit_copilot.config import settings
from credit_copilot.console import enable_unicode_console
from credit_copilot.data.loader import load_dataset
from credit_copilot.data.preprocessor import build_preprocessor
from credit_copilot.models.estimators import (
    PRODUCTION_CALIBRATION_METHOD,
    PRODUCTION_FOREST_PARAMS,
    PRODUCTION_OPERATING_THRESHOLD,
    build_calibrated_forest,
)
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    MODEL_STEP,
    PREPROCESSOR_STEP,
    split_features_and_target,
)
from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    MLflowConfigurationError,
    ensure_experiment,
)

REGISTERED_MODEL_NAME: Final[str] = "credit-risk-default-probability"
"""Name the model carries in the registry. Stable: new versions attach to this name."""

REGISTRY_TAG_VALUE: Final[str] = "production-candidate"
"""Marks the run that produced the registered artefact."""

SIGNATURE_SAMPLE_ROWS: Final[int] = 5
"""Rows used to infer the signature and to serve as the logged input example.

The signature is a schema - column names and dtypes - so five rows carry as much of it as
thirty thousand, and a small example keeps the registry entry light.
"""

CROSS_VALIDATED_METRICS: Final[dict[str, float]] = {
    "cv_pr_auc_mean": 0.564230,
    "cv_pr_auc_std": 0.007962,
    "cv_roc_auc_mean": 0.786279,
    "cv_ks_mean": 0.439205,
    "cv_gini_mean": 0.572558,
    "cv_brier_mean": 0.133408,
    "cv_precision_at_top_10pct_mean": 0.706333,
    "cv_precision_at_top_5pct_mean": 0.768667,
    "cv_baseline_trivial_pr_auc": 0.221200,
    "cv_baseline_logistic_pr_auc": 0.540173,
    "operating_threshold": PRODUCTION_OPERATING_THRESHOLD,
}
"""Metrics attached to the registered model, and where each one came from.

**None of these is computed in this script**, and that is the point: they are the
cross-validated figures of entry 006 of `docs/EVALUATION.md`, measured with the preprocessor
inside each fold and the model never scoring a row it was fitted on. Recomputing them here,
on the rows this script fits on, would produce larger numbers that mean nothing.

The two `baseline` entries travel with the model on purpose. A registry entry that records
PR-AUC 0.564 and not the 0.221 floor it is read against invites the number being quoted on
its own, which the methodology treats as the failure mode it is.
"""

SKOPS_TRUSTED_TYPES: Final[list[str]] = [
    "credit_copilot.data.preprocessor.AttachPaymentBehaviourFeatures",
    "credit_copilot.data.preprocessor.CollapseEducation",
    "credit_copilot.data.preprocessor.PercentileClipper",
    "credit_copilot.features.builder.PaymentBehaviourFeatures",
    "numpy.dtype",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
]
"""Classes the artefact contains that `skops` will not deserialise without permission.

**Why this list has to exist at all.** MLflow 3 serialises scikit-learn models with `skops`
rather than pickle, and `skops` refuses to reconstruct any class it was not told to trust -
loading a pickle executes whatever the file says to execute, and that is a real way to be
attacked by a model artefact. The refusal is the feature.

**Why trusting exactly these is safe.** The first four are this project's own transformers,
defined in this repository and reviewed like any other code here. The last three are
scikit-learn internals and a numpy type that the calibrated forest is built from. Nothing
third-party and nothing dynamic is on the list.

**Why the list is spelled out rather than switched off.** `serialization_format` could be
set back to cloudpickle, which trusts everything and never raises. Naming the seven types
instead keeps the guarantee and makes the contents of the artefact legible: a change to the
pipeline that adds a new custom step will fail loudly here, and the fix is to read the new
name and decide whether it belongs.
"""

ROUND_TRIP_TOLERANCE: Final[float] = 1e-9
"""Largest prediction difference the round-trip check accepts between save and load.

**Bit equality is the wrong test here, and finding that out was a measurement.** The forest
runs with `n_jobs=-1`, so `predict_proba` accumulates the votes of 300 trees across threads.
Floating-point addition is not associative, thread scheduling varies between calls, and the
result is that **the same fitted object called twice on the same row does not return bit-
identical probabilities**. Measured on this pipeline the spread is **5e-16**, and with
`n_jobs=1` it is exactly zero, which identifies the cause beyond doubt.

That is fifteen orders of magnitude below the operating threshold of 0.160, so it cannot
move a decision about a client. This tolerance is set at 1e-9 - still far below anything
that could flip a refusal, and far above the noise - so the check tests what actually
matters: that the artefact loaded from the registry is the artefact that was saved, rather
than that two floating-point sums happened to be ordered the same way.
"""

RULE: Final[str] = "=" * 100


def build_production_pipeline() -> Pipeline:
    """Assemble the artefact that gets deployed: preprocessor plus calibrated forest.

    One object, with the same two step names every other script in the project uses, so the
    thing in the registry has the same shape as the thing the cross-validation measured.

    Returns:
        An unfitted pipeline taking the 23 raw canonical columns.
    """
    return Pipeline(
        [
            (PREPROCESSOR_STEP, build_preprocessor()),
            (MODEL_STEP, build_calibrated_forest(PRODUCTION_CALIBRATION_METHOD)),
        ]
    )


def main() -> int:
    """Fit the production pipeline on everything, log it, and register it.

    Returns:
        0 on success, 1 if MLflow could not be configured, 2 if the registry refused.
    """
    enable_unicode_console()
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 1

    print(RULE)
    print("REGISTERING THE PRODUCTION MODEL")
    print(RULE)
    print(f"Tracking server : {context.tracking_uri}")
    print(f"Experiment      : {context.name} (id {context.experiment_id})")
    print(f"Registry name   : {REGISTERED_MODEL_NAME}")

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    print(f"Rows            : {len(frame):,}  (all of them - see the module docstring)")
    print(f"Raw columns     : {features.shape[1]} predictors, target held out")
    print(f"Model           : random forest {dict(PRODUCTION_FOREST_PARAMS)}")
    print(f"                  class_weight=None, {PRODUCTION_CALIBRATION_METHOD} calibration")
    print(f"Threshold       : {PRODUCTION_OPERATING_THRESHOLD:.3f} at 5:1 FN:FP")

    print("\n" + RULE)
    print("WHAT THIS FIT IS AND IS NOT")
    print(RULE)
    print("  This fit produces the ARTEFACT. It produces no metric.")
    print(f"  The metrics attached come from {DEFAULT_N_SPLITS}-fold cross-validation, entry 006,")
    print("  where the preprocessor was fitted inside each fold and no row was ever")
    print("  scored by a model that had seen it. Scoring these 30,000 rows with this")
    print("  fitted object would produce a larger number that means nothing.")

    print("\nFitting on the full dataset...")
    pipeline = build_production_pipeline()
    pipeline.fit(features, target)
    print("   fitted.")

    example = features.head(SIGNATURE_SAMPLE_ROWS)
    predictions = pipeline.predict_proba(example)
    signature = infer_signature(example, predictions)
    print(f"\nSignature inferred from {SIGNATURE_SAMPLE_ROWS} raw rows:")
    print(f"   inputs : {len(signature.inputs.inputs)} columns")
    print(f"   outputs: {signature.outputs}")

    try:
        with mlflow.start_run(
            experiment_id=context.experiment_id, run_name="production-model"
        ) as run:
            mlflow.set_tags(
                {
                    LEAKAGE_CHECK_TAG: REGISTRY_TAG_VALUE,
                    "phase": "02-modeling",
                    "fitted_on": "all 30000 rows - artefact, not an estimate",
                    "metrics_source": "5-fold cross-validation, EVALUATION.md entry 006",
                    "calibration_method": PRODUCTION_CALIBRATION_METHOD,
                    "imbalance_strategy": "none",
                }
            )
            mlflow.log_params(
                {
                    **{f"forest__{k}": str(v) for k, v in PRODUCTION_FOREST_PARAMS.items()},
                    "forest__class_weight": "None",
                    "calibration_method": PRODUCTION_CALIBRATION_METHOD,
                    "operating_threshold": str(PRODUCTION_OPERATING_THRESHOLD),
                    "cost_ratio_fn_to_fp": "5",
                    "n_training_rows": str(len(frame)),
                    "n_raw_columns": str(features.shape[1]),
                    "random_state": str(settings.random_state),
                }
            )
            mlflow.log_metrics(CROSS_VALIDATED_METRICS)

            info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=example,
                registered_model_name=REGISTERED_MODEL_NAME,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            run_id = run.info.run_id
    except Exception as error:  # noqa: BLE001 - the registry raises many unrelated types
        print(f"The registry refused the model: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))

    print("\n" + RULE)
    print("REGISTERED")
    print(RULE)
    print(f"  name          {REGISTERED_MODEL_NAME}")
    print(f"  version       {latest.version}")
    print(f"  run           {run_id}")
    print(f"  model uri     {info.model_uri}")
    print(f"  total versions{len(versions):>3}")

    print("\nRound trip: loading the registered artefact back and scoring a raw row")
    print("-" * 100)
    loaded = mlflow.sklearn.load_model(info.model_uri)
    reloaded = loaded.predict_proba(example)[:, 1]
    original = predictions[:, 1]
    print(f"   original  {[f'{value:.6f}' for value in original]}")
    print(f"   reloaded  {[f'{value:.6f}' for value in reloaded]}")
    difference = float(abs(original - reloaded).max())
    print(f"   max |difference|: {difference:.3e}   tolerance: {ROUND_TRIP_TOLERANCE:.0e}")
    if difference > ROUND_TRIP_TOLERANCE:
        print(
            f"   The loaded artefact does not reproduce its own predictions: {difference:.3e} "
            f"exceeds {ROUND_TRIP_TOLERANCE:.0e}.",
            file=sys.stderr,
        )
        return 2
    print("   The artefact loaded from the registry is the artefact that was saved.")

    decisions = pd.Series(reloaded >= PRODUCTION_OPERATING_THRESHOLD).map(
        {True: "refuse", False: "approve"}
    )
    print(f"   decisions at threshold {PRODUCTION_OPERATING_THRESHOLD:.3f}: {list(decisions)}")

    print("\n" + RULE)
    print("MLFLOW")
    print(RULE)
    print(f"Experiment: {context.url}")
    print(f"    production-model       run_id {run_id}")
    print(
        "\nThe artefact takes the 23 raw canonical columns that loader.load_dataset returns.\n"
        "A consumer never builds the feature matrix itself - that is the guarantee of\n"
        "section 6.3 of the methodology, and it is why the whole pipeline is registered."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
