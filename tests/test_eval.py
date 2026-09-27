"""The metric must match the problem statement exactly -- everything keys off it."""

import pytest

from ber.eval import blocking_recall, entity_fbeta, fbeta, score_predictions


def test_problem_statement_worked_example():
    """The PS gives: predict 3, truth 2, 2 correct -> 0.714."""
    score = entity_fbeta({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert score == pytest.approx(0.7143, abs=1e-4)


def test_fbeta_formula_matches_stated_form():
    """F_0.5 = (1.25 * P * R) / (0.25 * P + R)."""
    p, r = 0.6, 0.9
    assert fbeta(p, r, 0.5) == pytest.approx((1.25 * p * r) / (0.25 * p + r))


def test_correct_singleton_scores_one():
    assert entity_fbeta(set(), set()) == 1.0


def test_any_prediction_on_a_singleton_scores_zero():
    assert entity_fbeta({"S2-1"}, set()) == 0.0


def test_missing_every_match_scores_zero():
    assert entity_fbeta(set(), {"S2-1"}) == 0.0


def test_precision_is_weighted_above_recall():
    """beta=0.5 must punish a false merge harder than a miss."""
    over = entity_fbeta({"S2-1", "S2-2"}, {"S2-1"})   # one spurious extra
    under = entity_fbeta({"S2-1"}, {"S2-1", "S2-2"})  # one missed
    assert over < under


def test_entities_absent_from_predictions_count_as_empty():
    """A missing row is an empty prediction, not an excused entity."""
    score = score_predictions({"a": {"x"}}, {"a": {"x"}, "b": {"y"}})
    assert score.n_entities == 2
    assert score.macro_fbeta == pytest.approx(0.5)


def test_macro_average_weights_entities_equally():
    """A 1-match entity counts as much as a 10-match one."""
    truth = {"a": {"x"}, "b": {f"y{i}" for i in range(10)}}
    only_small = score_predictions({"a": {"x"}}, truth).macro_fbeta
    only_big = score_predictions({"b": truth["b"]}, truth).macro_fbeta
    assert only_small == pytest.approx(only_big)


def test_blocking_recall_counts_kept_pairs():
    truth = {"a": {"x", "y"}, "b": {"z"}}
    candidates = {"a": {"x", "q"}, "b": {"z"}}
    report = blocking_recall(candidates, truth)
    assert report["true_pairs_total"] == 3
    assert report["true_pairs_kept"] == 2
    assert report["pair_recall"] == pytest.approx(2 / 3)


def test_singletons_count_as_perfect_recall_for_blocking():
    report = blocking_recall({"a": set()}, {"a": set()})
    assert report["macro_entity_recall"] == 1.0
