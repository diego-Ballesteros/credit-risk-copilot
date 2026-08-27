"""The metrics ADR-0002 fixes, computed in exactly one place.

**Why they live here and not in each script.** ADR-0002 fixes seven metrics and makes
PR-AUC the only one that decides a comparison. A script that recomputes "the same" KS with
its own three lines will eventually disagree with another script's three lines, and the
disagreement is invisible: both produce a plausible number in the right range. There is
one implementation, every consumer imports it, and a difference between two reported
numbers can then only come from the model or the data.

**What is deliberately not in `METRIC_NAMES`.** `accuracy_at_threshold` is exported and is
*not* one of the seven. ADR-0002 discards accuracy because a model that always predicts
"does not default" scores about 78% on this dataset while identifying nobody. Keeping the
function available makes that argument demonstrable instead of asserted - the baselines
script prints it next to PR-AUC precisely so the gap is visible - and keeping it out of
`METRIC_NAMES` makes sure nothing can quietly start deciding on it.

**Why `precision_at_top_percent` is the only one written from scratch.** The other six are
thin wrappers over scikit-learn, and wrapping rather than reimplementing is the point: the
test then checks that the right function was called with its arguments in the right order,
which is the mistake that actually happens. Precision in the top decile has no
scikit-learn equivalent, and its tie policy is a real decision - see its docstring.
"""

import math
from collections.abc import Mapping
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

DECISION_METRIC: Final[str] = "pr_auc"
"""The single metric that decides a comparison between models, per ADR-0002."""

TOP_PERCENT_PRIMARY: Final[float] = 10.0
"""Primary review capacity, as a percentage of the scored population. ADR-0002."""

TOP_PERCENT_SECONDARY: Final[float] = 5.0
"""Secondary review capacity, as a percentage of the scored population. ADR-0002."""

DEFAULT_DECISION_THRESHOLD: Final[float] = 0.5
"""Probability cut used only by `accuracy_at_threshold`, which decides nothing."""

METRIC_NAMES: Final[tuple[str, ...]] = (
    "pr_auc",
    "roc_auc",
    "ks",
    "gini",
    "brier",
    "precision_at_top_10pct",
    "precision_at_top_5pct",
)
"""The seven metrics of ADR-0002, in reporting order. The first one decides."""

ACCURACY_METRIC: Final[str] = "accuracy"
"""Name of the metric ADR-0002 discards. Reported to show why, never to decide."""

REPORTED_METRIC_NAMES: Final[tuple[str, ...]] = (*METRIC_NAMES, ACCURACY_METRIC)
"""Everything `compute_metrics` returns: the seven that count and the one that does not."""

LOWER_IS_BETTER: Final[frozenset[str]] = frozenset({"brier"})
"""Metrics whose better direction is downward. Every other reported metric goes up."""


class MetricInputError(ValueError):
    """The arrays handed to a metric cannot support the number it is asked for."""


def _validated_pair(
    y_true: ArrayLike,
    y_score: ArrayLike,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Coerce and check a (label, score) pair before any metric touches it.

    Both classes are required to be present, for every metric and not only for the ones
    that mathematically need it. ROC-AUC, KS and Gini are undefined on a single-class
    array, and the seven metrics are always computed together, so a shared precondition is
    honest about what the caller actually has to provide. A fold with one class cannot
    happen under stratified splitting on this dataset; if it ever does, the right outcome
    is a loud failure rather than a `nan` travelling into a reported mean.

    Args:
        y_true: Ground-truth labels, expected to hold only 0 and 1.
        y_score: Predicted score or probability for the positive class.

    Returns:
        The labels as integers and the scores as floats, same length.

    Raises:
        MetricInputError: If the lengths differ, the arrays are empty, a label is not 0
            or 1, only one class is present, or a score is not finite.
    """
    labels = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()

    if labels.shape[0] != scores.shape[0]:
        raise MetricInputError(
            f"y_true has {labels.shape[0]} rows and y_score has {scores.shape[0]}."
        )
    if labels.shape[0] == 0:
        raise MetricInputError("No rows to score.")

    present = set(np.unique(labels).tolist())
    if not present <= {0, 1}:
        raise MetricInputError(
            f"y_true must hold only 0 and 1; found {sorted(present)}. The target of this "
            "project is binary and a third value means the labels were mangled upstream."
        )
    if len(present) < 2:
        raise MetricInputError(
            f"y_true holds only the class {present.pop()}. Ranking metrics are undefined "
            "on a single class, and returning nan would hide that inside a mean."
        )
    if not np.isfinite(scores).all():
        raise MetricInputError(
            "y_score holds nan or inf; a metric computed on it would be silent nonsense."
        )

    return labels.astype(np.int_), scores


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve. The decision metric of the project.

    Measures how well the scores rank defaulters above non-defaulters, judged by precision
    - whose denominator is only the rows flagged positive. That is why ADR-0002 prefers it
    to ROC-AUC here: with 22% positives the false-positive rate has a huge denominator that
    absorbs mistakes on the minority class without moving the curve.

    How to read it: the floor is the positive-class prevalence, not 0.5. A value equal to
    the prevalence means the ranking carries no information at all, so this number is
    meaningless unless the prevalence is printed beside it.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_score: Predicted probability of the positive class.

    Returns:
        Average precision in [0, 1].
    """
    labels, scores = _validated_pair(y_true, y_score)
    return float(average_precision_score(labels, scores))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve. Context only; it never decides a comparison.

    The probability that a randomly chosen defaulter is scored above a randomly chosen
    non-defaulter.

    How to read it: the floor is 0.5 regardless of the class balance, which is exactly what
    makes it a comfortable and optimistic number on an imbalanced problem. ADR-0002 keeps
    it because it is the lingua franca of the domain, and refuses to decide on it.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_score: Predicted probability of the positive class.

    Returns:
        ROC-AUC in [0, 1].
    """
    labels, scores = _validated_pair(y_true, y_score)
    return float(roc_auc_score(labels, scores))


def ks_statistic(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Kolmogorov-Smirnov statistic: the widest gap between the two score distributions.

    Computed as the maximum of `TPR - FPR` over every threshold on the ROC curve, which is
    the same quantity as the maximum distance between the cumulative distribution of the
    scores of defaulters and that of non-defaulters.

    How to read it: 0 means the two populations are scored identically and the model
    separates nothing. It is the standard separation measure in credit risk, and unlike
    PR-AUC it describes a single best threshold rather than the whole curve.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_score: Predicted probability of the positive class.

    Returns:
        KS in [0, 1].
    """
    labels, scores = _validated_pair(y_true, y_score)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    return float(np.max(true_positive_rate - false_positive_rate))


def gini(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Gini coefficient: `2 * ROC-AUC - 1`. Context only.

    Carries exactly the information ROC-AUC carries, rescaled so that 0 is the coin flip
    and 1 is perfect ranking. It is reported because the credit-risk domain reads Gini
    fluently, not because it adds anything ROC-AUC does not already say.

    How to read it: a Gini of 0 is a model that orders nothing. Because it is an affine
    transform of ROC-AUC, it inherits the same optimistic bias on imbalanced data.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_score: Predicted probability of the positive class.

    Returns:
        Gini in [-1, 1].
    """
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error of the predicted probabilities. Calibration, not ranking.

    The only reported metric that reads the *value* of the probability rather than the
    order it induces. A model can rank perfectly and still be badly calibrated, and without
    calibration there is no expected loss and therefore no pricing.

    How to read it: **lower is better**, unlike every other metric here. Its floor is not 0
    in practice: a perfectly calibrated model on an irreducibly uncertain problem still
    pays for the residual noise. It has to be read against the baseline's Brier on the same
    data, never on its own.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_prob: Predicted probability of the positive class, in [0, 1].

    Returns:
        Brier score in [0, 1].
    """
    labels, probabilities = _validated_pair(y_true, y_prob)
    return float(np.mean((probabilities - labels) ** 2))


def precision_at_top_percent(y_true: ArrayLike, y_score: ArrayLike, percent: float) -> float:
    """Share of true defaulters among the worst-scored `percent`% of the population.

    ADR-0002 expresses the cut as a percentage of the scored population rather than an
    absolute `k`, because a percentage is a statement about review capacity - how many
    files the team can open out of everything that arrives - and transfers between sample
    sizes, while "the worst 500" means something different on 5,000 clients than on 50,000.

    The reviewed group holds `ceil(n * percent / 100)` rows, so it is never empty.

    **How ties are resolved, and why it matters.** The decile cut frequently lands in the
    middle of a block of rows that share one score. The trivial baseline is the extreme
    case: it gives *every* row the same probability, so the cut falls entirely inside one
    tie group, and any rule that simply takes the first `k` rows after sorting reports
    whatever the row order happened to be. That number would be an artefact of the index,
    and it would move if the same rows arrived shuffled.

    The rule used here is the **expected precision under uniform random tie-breaking**.
    Rows scored strictly above the cut are counted whole; the remaining slots are filled
    from the tie group at that group's own positive rate::

        expected positives = positives strictly above
                             + positives in the tie group * (free slots / tie group size)

    Three properties follow, and each one is why the metric is worth reporting at all. It
    is **deterministic** - no seed, no run-to-run drift. It is **independent of row order**
    - shuffling the input cannot change it. And it is **exactly the naive top-k precision
    whenever no tie straddles the cut**, which is the ordinary case for a real model, so
    the policy costs nothing when it is not needed. In the degenerate case where one score
    covers the whole population it returns the prevalence, which is the honest answer: a
    constant ranking concentrates nothing.

    How to read it: the floor is the prevalence. A value equal to the prevalence means the
    top decile is no worse than a decile drawn at random.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_score: Predicted probability of the positive class.
        percent: Size of the reviewed group as a percentage of the population, in (0, 100].

    Returns:
        Expected precision in the top group, in [0, 1].

    Raises:
        MetricInputError: If `percent` is outside (0, 100].
    """
    if not 0.0 < percent <= 100.0:
        raise MetricInputError(f"percent must be in (0, 100]; got {percent}.")

    labels, scores = _validated_pair(y_true, y_score)
    n_rows = labels.shape[0]
    group_size = min(n_rows, math.ceil(n_rows * percent / 100.0))

    cut_score = float(np.sort(scores)[::-1][group_size - 1])
    strictly_above = scores > cut_score
    tied_at_cut = scores == cut_score

    free_slots = group_size - int(strictly_above.sum())
    tie_group_size = int(tied_at_cut.sum())
    expected_positives = float(labels[strictly_above].sum()) + float(labels[tied_at_cut].sum()) * (
        free_slots / tie_group_size
    )
    return float(expected_positives / group_size)


def accuracy_at_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> float:
    """Share of correct labels once the probabilities are cut at `threshold`.

    **Deliberately outside `METRIC_NAMES`.** ADR-0002 discards accuracy as a decision
    metric: with ~22% positives, a model that always answers "does not default" reaches
    ~78% while identifying not one of the cases that matter. It is computed and reported so
    that the reader can watch the trivial baseline win on this number and lose on every
    other one, which is a shorter argument than the paragraph explaining it.

    How to read it: never on its own, and never against another model. Read it only next to
    the same model's PR-AUC.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_score: Predicted probability of the positive class.
        threshold: Probability at or above which a row is labelled positive.

    Returns:
        Accuracy in [0, 1].
    """
    labels, scores = _validated_pair(y_true, y_score)
    return float(np.mean((scores >= threshold).astype(np.int_) == labels))


def compute_metrics(y_true: ArrayLike, y_score: ArrayLike) -> Mapping[str, float]:
    """Compute every reported metric at once, so that no caller can report a subset.

    Returning all of them together is the mechanism behind the methodology's rule that a
    metric never travels alone. A caller that wanted only ROC-AUC would have to discard the
    rest explicitly, and that is visible in a diff.

    Args:
        y_true: Ground-truth labels, 0 or 1.
        y_score: Predicted probability of the positive class.

    Returns:
        Metric name -> value, keyed by `REPORTED_METRIC_NAMES` and in that order: the seven
        of ADR-0002 followed by accuracy, which decides nothing.
    """
    labels, scores = _validated_pair(y_true, y_score)
    return {
        "pr_auc": pr_auc(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "ks": ks_statistic(labels, scores),
        "gini": gini(labels, scores),
        "brier": brier_score(labels, scores),
        "precision_at_top_10pct": precision_at_top_percent(labels, scores, TOP_PERCENT_PRIMARY),
        "precision_at_top_5pct": precision_at_top_percent(labels, scores, TOP_PERCENT_SECONDARY),
        ACCURACY_METRIC: accuracy_at_threshold(labels, scores),
    }
