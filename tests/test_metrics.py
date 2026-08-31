"""Tests of the seven metrics of ADR-0002.

Every expected value in this file is worked out by hand in the test's own comment, from
arrays small enough to count on paper. That is the whole design: a test that compares
`pr_auc(y, s)` against `average_precision_score(y, s)` restates the implementation and
passes even when the wrong function is called or the arguments are swapped. A test that
compares it against a number derived from the definition does not.

The tie cases in `precision_at_top_percent` get the most attention, because that is the one
place where a defensible-looking implementation can produce a number that depends on the
order the rows happened to arrive in.
"""

import numpy as np
import pytest

from credit_copilot.models.metrics import (
    ACCURACY_METRIC,
    LOWER_IS_BETTER,
    METRIC_NAMES,
    REPORTED_METRIC_NAMES,
    MetricInputError,
    accuracy_at_threshold,
    brier_score,
    compute_metrics,
    gini,
    ks_statistic,
    pr_auc,
    precision_at_top_percent,
    roc_auc,
)

# ---------------------------------------------------------------------------
# A ranking with no ties, small enough to evaluate on paper.
#
#   score   0.90  0.80  0.70  0.60  0.50  0.40  0.30  0.20
#   label      1     1     0     1     0     0     1     0
#
# Four positives, four negatives.
# ---------------------------------------------------------------------------
SIMPLE_LABELS = np.array([1, 1, 0, 1, 0, 0, 1, 0])
SIMPLE_SCORES = np.array([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20])


def test_roc_auc_equals_the_share_of_correctly_ordered_pairs() -> None:
    # ROC-AUC is the share of (positive, negative) pairs the score orders correctly.
    # 4 positives x 4 negatives = 16 pairs. Negatives sit at ranks 3, 5, 6, 8.
    #   positive at rank 1 beats all 4 negatives      -> 4
    #   positive at rank 2 beats all 4 negatives      -> 4
    #   positive at rank 4 beats the negatives at 5,6,8 -> 3
    #   positive at rank 7 beats the negative at 8    -> 1
    # 4 + 4 + 3 + 1 = 12 correctly ordered pairs out of 16 -> 0.75
    assert roc_auc(SIMPLE_LABELS, SIMPLE_SCORES) == pytest.approx(12 / 16)


def test_gini_is_the_affine_rescaling_of_roc_auc() -> None:
    # Gini = 2 * ROC-AUC - 1 = 2 * 0.75 - 1 = 0.5
    assert gini(SIMPLE_LABELS, SIMPLE_SCORES) == pytest.approx(0.5)


def test_pr_auc_equals_the_recall_weighted_sum_of_precisions() -> None:
    # Average precision sums the precision at each threshold where a positive appears,
    # weighted by the recall it adds. Walking down the ranking:
    #   rank 1 (label 1): 1 of 1 flagged is positive  -> precision 1/1, recall step 1/4
    #   rank 2 (label 1): 2 of 2                      -> precision 2/2, recall step 1/4
    #   rank 4 (label 1): 3 of 4                      -> precision 3/4, recall step 1/4
    #   rank 7 (label 1): 4 of 7                      -> precision 4/7, recall step 1/4
    # AP = (1/4)(1 + 1 + 3/4 + 4/7) = (1/4)(3.321428...) = 0.830357...
    expected = 0.25 * (1.0 + 1.0 + 0.75 + 4 / 7)
    assert pr_auc(SIMPLE_LABELS, SIMPLE_SCORES) == pytest.approx(expected)


def test_ks_is_the_widest_gap_between_the_two_cumulative_distributions() -> None:
    # KS = max over thresholds of (TPR - FPR). Sweeping the ranking top-down, with
    # 4 positives and 4 negatives:
    #   after rank 1: TPR 1/4, FPR 0/4 -> 0.25
    #   after rank 2: TPR 2/4, FPR 0/4 -> 0.50   <- widest
    #   after rank 3: TPR 2/4, FPR 1/4 -> 0.25
    #   after rank 4: TPR 3/4, FPR 1/4 -> 0.50   <- tied widest
    #   after rank 5: TPR 3/4, FPR 2/4 -> 0.25
    #   after rank 6: TPR 3/4, FPR 3/4 -> 0.00
    #   after rank 7: TPR 4/4, FPR 3/4 -> 0.25
    #   after rank 8: TPR 4/4, FPR 4/4 -> 0.00
    assert ks_statistic(SIMPLE_LABELS, SIMPLE_SCORES) == pytest.approx(0.5)


def test_brier_is_the_mean_squared_distance_to_the_label() -> None:
    # Squared errors, in order:
    #   (0.90-1)^2 = 0.01   (0.80-1)^2 = 0.04   (0.70-0)^2 = 0.49   (0.60-1)^2 = 0.16
    #   (0.50-0)^2 = 0.25   (0.40-0)^2 = 0.16   (0.30-1)^2 = 0.49   (0.20-0)^2 = 0.04
    # sum = 1.64 over 8 rows -> 0.205
    assert brier_score(SIMPLE_LABELS, SIMPLE_SCORES) == pytest.approx(0.205)


def test_brier_is_the_only_reported_metric_that_improves_downwards() -> None:
    assert set(LOWER_IS_BETTER) == {"brier"}


def test_accuracy_cuts_at_the_threshold_and_counts_agreements() -> None:
    # At 0.5, the predicted labels are 1,1,1,1,1,0,0,0 against 1,1,0,1,0,0,1,0.
    # Agreements at positions 1,2,4,6,8 -> 5 of 8 = 0.625
    assert accuracy_at_threshold(SIMPLE_LABELS, SIMPLE_SCORES) == pytest.approx(5 / 8)


def test_accuracy_treats_the_threshold_itself_as_positive() -> None:
    # Labels 1,0 with scores exactly 0.5 and 0.5: both predicted positive, so one hit.
    assert accuracy_at_threshold([1, 0], [0.5, 0.5], threshold=0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# precision@top-k%: the no-tie case
# ---------------------------------------------------------------------------


def test_precision_at_top_percent_with_no_ties_is_plain_top_k_precision() -> None:
    # n = 8, percent = 50 -> group size ceil(4.0) = 4. The top 4 scores are
    # 0.90, 0.80, 0.70, 0.60 with labels 1, 1, 0, 1 -> 3 positives of 4 = 0.75
    assert precision_at_top_percent(SIMPLE_LABELS, SIMPLE_SCORES, 50.0) == pytest.approx(0.75)


def test_the_group_size_is_rounded_up_so_it_is_never_empty() -> None:
    # n = 8, percent = 5 -> 0.4 rows, rounded up to 1. The single worst-scored row is
    # the one at 0.90, whose label is 1 -> precision 1.0
    assert precision_at_top_percent(SIMPLE_LABELS, SIMPLE_SCORES, 5.0) == pytest.approx(1.0)


def test_the_whole_population_scores_the_prevalence() -> None:
    # percent = 100 selects everybody, so precision is the prevalence: 4 of 8.
    assert precision_at_top_percent(SIMPLE_LABELS, SIMPLE_SCORES, 100.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# precision@top-k%: the tie cases, which are the reason the policy is documented
#
#   score   0.90  0.50  0.50  0.50  0.50  0.10  0.10  0.10  0.10  0.10
#   label      1     1     0     0     1     0     0     0     0     0
#
# n = 10, percent = 30 -> group size 3. One row sits strictly above the cut of 0.50,
# leaving 2 free slots to be filled from a tie group of 4 rows holding 2 positives.
#   expected positives = 1 + 2 * (2/4) = 2.0
#   precision          = 2.0 / 3 = 0.666666...
# ---------------------------------------------------------------------------
TIED_SCORES = np.array([0.90, 0.50, 0.50, 0.50, 0.50, 0.10, 0.10, 0.10, 0.10, 0.10])
TIED_LABELS = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])


def test_a_tie_straddling_the_cut_is_resolved_by_expected_value() -> None:
    assert precision_at_top_percent(TIED_LABELS, TIED_SCORES, 30.0) == pytest.approx(2.0 / 3.0)


def test_the_tie_policy_does_not_depend_on_the_order_the_rows_arrive_in() -> None:
    # This is the property the policy exists for. A rule that takes the first k rows after
    # sorting would return 3/3 or 2/3 or 1/3 here depending only on how the four tied rows
    # happen to be laid out, and nothing in the data would have changed.
    expected = precision_at_top_percent(TIED_LABELS, TIED_SCORES, 30.0)
    generator = np.random.default_rng(0)
    for _ in range(25):
        order = generator.permutation(len(TIED_LABELS))
        assert precision_at_top_percent(
            TIED_LABELS[order], TIED_SCORES[order], 30.0
        ) == pytest.approx(expected)


def test_a_constant_score_returns_the_prevalence() -> None:
    # The degenerate case the trivial baseline actually produces: one score for everybody,
    # so the cut lies entirely inside a single tie group of all 10 rows holding 3 positives.
    #   expected positives = 0 + 3 * (3/10) = 0.9 over a group of 3 -> 0.3, the prevalence.
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    scores = np.full(10, 0.42)
    assert precision_at_top_percent(labels, scores, 30.0) == pytest.approx(0.3)
    assert precision_at_top_percent(labels, scores, 10.0) == pytest.approx(0.3)
    assert precision_at_top_percent(labels, scores, 50.0) == pytest.approx(0.3)


def test_a_tie_that_ends_exactly_at_the_cut_is_the_plain_top_k_case() -> None:
    # score  0.9 0.9 0.9 0.2 0.2 ... : with group size 3 the tie group is exactly the
    # group, no slot is left over, and the answer is the plain count: labels 1,1,0 -> 2/3.
    labels = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    scores = np.array([0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    assert precision_at_top_percent(labels, scores, 30.0) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# compute_metrics: the contract of the bundle
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_every_reported_name_in_order() -> None:
    computed = compute_metrics(SIMPLE_LABELS, SIMPLE_SCORES)
    assert tuple(computed) == REPORTED_METRIC_NAMES


def test_accuracy_is_reported_but_is_not_one_of_the_seven() -> None:
    # ADR-0002 fixes seven metrics and rejects accuracy. The bundle carries accuracy so the
    # rejection can be demonstrated; METRIC_NAMES is what decides anything.
    assert len(METRIC_NAMES) == 7
    assert ACCURACY_METRIC not in METRIC_NAMES
    assert ACCURACY_METRIC in REPORTED_METRIC_NAMES


def test_compute_metrics_agrees_with_the_individual_functions() -> None:
    computed = compute_metrics(SIMPLE_LABELS, SIMPLE_SCORES)
    assert computed["pr_auc"] == pytest.approx(pr_auc(SIMPLE_LABELS, SIMPLE_SCORES))
    assert computed["roc_auc"] == pytest.approx(roc_auc(SIMPLE_LABELS, SIMPLE_SCORES))
    assert computed["ks"] == pytest.approx(ks_statistic(SIMPLE_LABELS, SIMPLE_SCORES))
    assert computed["gini"] == pytest.approx(gini(SIMPLE_LABELS, SIMPLE_SCORES))
    assert computed["brier"] == pytest.approx(brier_score(SIMPLE_LABELS, SIMPLE_SCORES))
    assert computed["precision_at_top_10pct"] == pytest.approx(
        precision_at_top_percent(SIMPLE_LABELS, SIMPLE_SCORES, 10.0)
    )
    assert computed["precision_at_top_5pct"] == pytest.approx(
        precision_at_top_percent(SIMPLE_LABELS, SIMPLE_SCORES, 5.0)
    )


def test_a_perfect_ranking_maxes_every_ranking_metric() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    computed = compute_metrics(labels, scores)
    assert computed["roc_auc"] == pytest.approx(1.0)
    assert computed["pr_auc"] == pytest.approx(1.0)
    assert computed["ks"] == pytest.approx(1.0)
    assert computed["gini"] == pytest.approx(1.0)


def test_an_inverted_ranking_is_the_mirror_of_a_perfect_one() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(labels, scores) == pytest.approx(0.0)
    assert gini(labels, scores) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# The guard rails
# ---------------------------------------------------------------------------


def test_a_single_class_is_refused_rather_than_returning_nan() -> None:
    with pytest.raises(MetricInputError, match="single class"):
        roc_auc([1, 1, 1], [0.2, 0.5, 0.9])


def test_a_label_that_is_not_zero_or_one_is_refused() -> None:
    with pytest.raises(MetricInputError, match="only 0 and 1"):
        pr_auc([0, 1, 2], [0.2, 0.5, 0.9])


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(MetricInputError, match="rows"):
        pr_auc([0, 1], [0.2, 0.5, 0.9])


def test_an_empty_input_is_refused() -> None:
    with pytest.raises(MetricInputError, match="No rows"):
        pr_auc([], [])


def test_a_non_finite_score_is_refused() -> None:
    with pytest.raises(MetricInputError, match="nan or inf"):
        pr_auc([0, 1], [0.2, np.nan])


@pytest.mark.parametrize("percent", [0.0, -5.0, 100.1])
def test_a_percent_outside_the_open_interval_is_refused(percent: float) -> None:
    with pytest.raises(MetricInputError, match="percent"):
        precision_at_top_percent(SIMPLE_LABELS, SIMPLE_SCORES, percent)
