"""Loss functions derived from the evaluation criterion.

The leaderboard metric is macro F_0.5 over Source-1 entities. It is not
differentiable -- it depends on a thresholded set intersection -- so a network
cannot optimize it directly. Two options exist, and both are provided:

``soft_fbeta``
    Replace the hard set operations with their expectations under the predicted
    probabilities. Within one entity, ``tp = sum(p_i * y_i)``,
    ``fp = sum(p_i * (1 - y_i))`` and ``fn = sum((1 - p_i) * y_i)`` are all
    differentiable, and F_beta composed from them is a smooth surrogate for the
    real thing. Crucially it is computed **per entity and then averaged**, which is
    what macro-averaging means -- a pairwise loss would instead optimize a
    micro-average and quietly ignore that every entity counts equally regardless
    of how many matches it has.

``weighted_bce``
    Plain binary cross-entropy with false positives weighted up. Simpler, and the
    honest baseline to check the surrogate against.

Entities with no true matches are the reason the surrogate needs care. 5.58% of
entities are singletons, and each scores a full 1.0 for a correctly empty
prediction. F_beta is degenerate there (``tp`` is 0 whatever you predict), so those
groups instead score ``1 - mean(p)``, which is maximised exactly when the model
predicts nothing -- the behaviour the metric rewards.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def soft_fbeta_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_index: torch.Tensor,
    n_groups: int,
    beta: float = 0.5,
) -> torch.Tensor:
    """``1 - mean(soft F_beta)`` over entity groups.

    ``group_index`` maps each pair to its entity's row in ``[0, n_groups)``. Pairs
    of one entity must be in the same batch, which is why the data loader batches
    by entity rather than by pair.
    """
    probs = torch.sigmoid(logits)
    zeros = torch.zeros(n_groups, dtype=probs.dtype, device=probs.device)

    true_pos = zeros.clone().index_add_(0, group_index, probs * labels)
    predicted = zeros.clone().index_add_(0, group_index, probs)
    actual = zeros.clone().index_add_(0, group_index, labels)
    counts = zeros.clone().index_add_(0, group_index, torch.ones_like(probs))

    precision = true_pos / (predicted + EPS)
    recall = true_pos / (actual + EPS)
    b2 = beta * beta
    fbeta = (1.0 + b2) * precision * recall / (b2 * precision + recall + EPS)

    # Singletons: no positives exist, so reward driving all mass to zero.
    empty_score = 1.0 - predicted / (counts + EPS)
    fbeta = torch.where(actual > 0, fbeta, empty_score)
    return 1.0 - fbeta.mean()


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    fp_weight: float = 2.0,
) -> torch.Tensor:
    """BCE with false positives up-weighted.

    ``fp_weight = 2.0`` mirrors F_0.5's 2x emphasis on precision over recall.
    """
    weights = torch.where(labels > 0.5, torch.ones_like(labels), torch.full_like(labels, fp_weight))
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=weights
    )


def build_loss(cfg):
    """Return ``fn(logits, labels, group_index, n_groups) -> loss`` from config."""
    settings = cfg["loss"]
    kind = settings["kind"]
    beta = float(settings["beta"])
    fp_weight = float(settings["fp_weight"])

    if kind == "soft_fbeta":
        def loss_fn(logits, labels, group_index, n_groups):
            return soft_fbeta_loss(logits, labels, group_index, n_groups, beta)
        return loss_fn
    if kind == "weighted_bce":
        def loss_fn(logits, labels, group_index, n_groups):
            return weighted_bce_loss(logits, labels, fp_weight)
        return loss_fn
    raise ValueError(f"unknown loss kind {kind!r}")
