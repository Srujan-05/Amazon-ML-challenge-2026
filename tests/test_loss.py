"""The loss must actually encode the evaluation criterion."""

import pytest
import torch

from ber.loss import soft_fbeta_loss, weighted_bce_loss

# entity 0: one true match among three candidates | entity 1: a singleton
LABELS = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
GROUPS = torch.tensor([0, 0, 0, 1, 1])


def loss_of(logits):
    return soft_fbeta_loss(torch.tensor(logits), LABELS, GROUPS, 2, 0.5).item()


def test_perfect_prediction_is_near_zero_loss():
    assert loss_of([8.0, -8.0, -8.0, -8.0, -8.0]) < 0.01


def test_false_merges_cost_more_than_misses():
    """F_0.5 weights precision 2x; the surrogate must inherit that."""
    over_merging = loss_of([8.0, 8.0, 8.0, 8.0, 8.0])
    missing_all = loss_of([-8.0, -8.0, -8.0, -8.0, -8.0])
    assert over_merging > missing_all


def test_predicting_on_a_singleton_is_penalized():
    """Entity 1 has no true matches; any mass there must cost something."""
    clean = loss_of([8.0, -8.0, -8.0, -8.0, -8.0])
    dirty = loss_of([8.0, -8.0, -8.0, 8.0, -8.0])
    assert dirty > clean


def test_gradient_pushes_only_the_true_positive_up():
    logits = torch.zeros(5, requires_grad=True)
    soft_fbeta_loss(logits, LABELS, GROUPS, 2, 0.5).backward()
    assert logits.grad[0] < 0
    assert (logits.grad[1:] > 0).all()


def test_loss_is_a_per_entity_average_not_a_pair_average():
    """One entity with many pairs must not outweigh one with few."""
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])
    small = torch.tensor([0, 0, 0, 0, 1])      # 4 pairs vs 1 pair
    logits = torch.tensor([8.0, -8.0, -8.0, -8.0, -8.0])
    # The 1-pair entity is wrong; it should move the mean by about 1/2, not 1/5.
    loss = soft_fbeta_loss(logits, labels, small, 2, 0.5).item()
    assert loss == pytest.approx(0.5, abs=0.05)


def test_weighted_bce_penalizes_false_positives_more():
    fp = weighted_bce_loss(torch.tensor([4.0]), torch.tensor([0.0]), 2.0).item()
    fn = weighted_bce_loss(torch.tensor([-4.0]), torch.tensor([1.0]), 2.0).item()
    assert fp > fn
