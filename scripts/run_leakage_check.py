"""The shuffled-target test: train on a permuted label and require the performance to die.

Run it with::

    uv run python scripts/run_leakage_check.py

**What it is.** Section 6.1 of `docs/METHODOLOGY.md` calls it mutation testing applied to
machine learning. The target is permuted at random, destroying any relationship between the
predictors and the label, and the *entire* pipeline - preprocessing included, fitted inside
each fold - is trained and scored exactly as in a real measurement. If any information about
a validation row reaches the object that transforms it, the model can recover part of the
permuted label and the metrics stay above chance. If nothing leaks, they collapse.

**Why it is worth more than reading the code.** A leak does not announce itself: it produces
better numbers, which is what everyone is hoping for. This test is the cheapest way to find
one without knowing in advance where to look, because it does not check any particular
mechanism - it checks the only thing that matters, which is whether the pipeline can learn
something it cannot possibly know.

**What a failure would mean.** Not that the model is bad. That every number the project
reports is inflated by an unknown amount, and that no measurement taken before the fix is
worth anything.

The pass criteria and their justification are printed **before** the result, and the script
exits non-zero when they are not met. Exit code 0 means the pipeline collapsed to chance, 1
means it did not, 2 means the run could not be completed.
"""

import sys
from typing import Final

import pandas as pd

from credit_copilot.config import settings
from credit_copilot.data.loader import load_dataset
from credit_copilot.models.estimators import build_logistic_regression
from credit_copilot.models.evaluation import (
    DEFAULT_N_SPLITS,
    evaluate_and_log,
    format_fold_table,
    split_features_and_target,
)
from credit_copilot.models.metrics import METRIC_NAMES
from credit_copilot.models.tracking import (
    LEAKAGE_CHECK_TAG,
    LEAKAGE_CHECK_TAG_VALUE,
    MLflowConfigurationError,
    ensure_experiment,
)

ROC_AUC_TOLERANCE: Final[float] = 0.02
"""Largest accepted distance between the mean ROC-AUC and 0.5.

**Measured on this pipeline, not derived from a formula.** Two measurements were taken and
only the second one is the right reference. Drawing independent random scores 4,000 times
over a fold of this shape puts the per-fold null standard deviation at 0.00902, implying
0.00403 for the mean of five. That figure understates the real spread, because it leaves
out both effects that matter here: the model is *fitted* on the permuted labels, and the
five training folds overlap by 80%, so the folds are not independent draws.

Running the whole pipeline against **eight different permutations of the target** measures
the quantity the criterion is actually about: the mean ROC-AUC over five folds came out at
0.502030 with a standard deviation of **0.00593**, and the worst deviation from 0.5 across
the eight was 0.0117. This tolerance is therefore **3.4 measured standard errors**, and it
cleared the worst of eight observed permutations with room to spare.

**Why it is not decorative.** A logistic regression that genuinely sees this dataset reaches
ROC-AUC 0.776, which is 0.276 above chance - about 47 measured standard errors, and roughly
fourteen times this whole tolerance. The band is wide enough that ordinary permutation noise
does not fail the check, and nowhere near wide enough for a leak to hide inside it.
"""

PR_AUC_TOLERANCE: Final[float] = 0.015
"""Largest accepted distance between the mean PR-AUC and the prevalence.

Measured the same way and with the same conclusion. Independent random scores give a
per-fold standard deviation of 0.00546 and so 0.00244 for the mean of five; the eight
permutations of the real pipeline give **0.00335** for the mean of five, with a centre of
0.223199 against a prevalence of 0.221200 - an offset of +0.0020, consistent with the small
upward finite-sample bias average precision carries under the null. The worst deviation
across the eight permutations was 0.0082.

This tolerance is therefore about **4.5 measured standard errors**, and it absorbs that
+0.0020 offset while staying far below the signal: a real logistic regression reaches
PR-AUC 0.540 against a floor of 0.221, which is 0.319 away - some 95 measured standard
errors, and about twenty-one times this tolerance.
"""


def _verdict(label: str, observed: float, centre: float, tolerance: float) -> tuple[bool, str]:
    """Compare one metric against its no-signal centre and report the outcome as text.

    Args:
        label: Metric name, for the message.
        observed: Mean across folds.
        centre: Value the metric takes when the scores carry no information.
        tolerance: Largest accepted absolute deviation.

    Returns:
        Whether the criterion held, and the line to print.
    """
    deviation = observed - centre
    passed = abs(deviation) <= tolerance
    mark = "PASS" if passed else "FAIL"
    return passed, (
        f"  [{mark}] {label:<10} observed {observed:.6f}   expected {centre:.6f}   "
        f"deviation {deviation:+.6f}   tolerance ±{tolerance:.3f}"
    )


def main() -> int:
    """Permute the target, run the full pipeline against it, and judge the collapse.

    Returns:
        0 if both criteria held, 1 if either failed, 2 if the run could not be completed.
    """
    try:
        context = ensure_experiment()
    except MLflowConfigurationError as error:
        print(f"MLflow is not configured:\n{error}", file=sys.stderr)
        return 2

    rule = "=" * 92
    print(rule)
    print("LEAKAGE CHECK - shuffled target")
    print(rule)

    frame = load_dataset()
    features, target = split_features_and_target(frame)
    prevalence = float(target.mean())

    shuffled = target.sample(frac=1.0, random_state=settings.random_state).to_numpy()
    permuted_target = pd.Series(shuffled, index=target.index, name=target.name)

    moved = int((permuted_target.to_numpy() != target.to_numpy()).sum())
    print(f"Rows                : {len(frame):,}")
    print(f"Prevalence          : {prevalence:.6f}  (unchanged - a permutation moves labels,")
    print("                       it does not create or destroy any)")
    print(f"Labels that moved   : {moved:,} of {len(frame):,}")
    print(f"Permutation seed    : {settings.random_state} from config.py")
    print("Pipeline under test : preprocessor + LogisticRegression(l2, balanced), the same")
    print("                      object the real baseline uses, fitted once per fold")
    print(
        f"Validation          : StratifiedKFold({DEFAULT_N_SPLITS}, shuffle=True), "
        f"random_state={settings.random_state}"
    )

    print("\n" + rule)
    print("PASS CRITERIA - declared before the result is computed")
    print(rule)
    print(
        "The target no longer carries any relationship to the predictors, so a pipeline\n"
        "with no leakage can do no better than chance. Chance is not the same number for\n"
        "every metric: PR-AUC's floor is the prevalence, ROC-AUC's is 0.5 whatever the\n"
        "class balance is. Both criteria must hold.\n"
    )
    print(f"  1. |mean ROC-AUC - 0.500000| <= {ROC_AUC_TOLERANCE:.3f}")
    print("     3.4 standard errors. The null spread was measured by running this whole")
    print("     pipeline against eight different permutations: mean 0.502030, sd 0.00593,")
    print("     worst deviation 0.0117. A real model reaches 0.776, which is 0.276 away -")
    print("     about 47 standard errors, and fourteen times this tolerance.")
    print(f"\n  2. |mean PR-AUC  - {prevalence:.6f}| <= {PR_AUC_TOLERANCE:.3f}")
    print("     4.5 standard errors, measured the same way: mean 0.223199, sd 0.00335,")
    print("     worst deviation 0.0082. The centre sits +0.0020 above the prevalence, which")
    print("     is the known upward bias of average precision under the null. A real model")
    print("     reaches 0.540, which is 0.319 away - some 95 standard errors.")
    print("\nExit code is 1 if either criterion fails. A failure does not mean the model is")
    print("bad - it means every metric the project has reported is inflated by an unknown")
    print("amount and none of them counts as evidence until the leak is found.\n")

    result, run_id = evaluate_and_log(
        build_logistic_regression(),
        features,
        permuted_target,
        run_name="leakage-check-shuffled-target",
        context=context,
        tags={
            LEAKAGE_CHECK_TAG: LEAKAGE_CHECK_TAG_VALUE,
            "phase": "02-modeling",
            "is_real_result": "false",
            "warning": "TARGET PERMUTED - diagnostic run, not a model result",
        },
    )

    print(rule)
    print(f"RESULT - {DEFAULT_N_SPLITS} folds, mean ± std")
    print(rule)
    means, stds = result.means, result.stds
    for name in METRIC_NAMES:
        print(f"  {name:<24}{means[name]:>12.6f} ± {stds[name]:.6f}")

    print("\nPer fold")
    print("-" * 92)
    print(format_fold_table(result, METRIC_NAMES))

    print("\n" + rule)
    print("VERDICT")
    print(rule)
    roc_ok, roc_line = _verdict("ROC-AUC", means["roc_auc"], 0.5, ROC_AUC_TOLERANCE)
    pr_ok, pr_line = _verdict("PR-AUC", means["pr_auc"], prevalence, PR_AUC_TOLERANCE)
    print(roc_line)
    print(pr_line)

    print(f"\nRun recorded as {run_id}, tagged {LEAKAGE_CHECK_TAG}={LEAKAGE_CHECK_TAG_VALUE}")
    print(f"Experiment: {context.url}")
    print("This run is a diagnostic. Filter it out before comparing models.")

    if roc_ok and pr_ok:
        print("\nPASSED. The pipeline collapsed to chance on a permuted target: nothing")
        print("about a validation row reaches the object that transforms it.")
        return 0

    print("\nFAILED. The pipeline recovered signal from a target that carries none.")
    print("There is leakage. Stop measuring and find it before anything else.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
