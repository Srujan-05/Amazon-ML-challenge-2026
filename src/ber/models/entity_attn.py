"""Entity-context pair classifier — NN variant 2.

The insight this model exists to exploit: **a candidate's correctness depends on
its competitors.** Each S2/S3 record belongs to at most one S1 entity (verified
over all 7,638,365 ground-truth pairs), so candidates within an entity are
competing, not independent. A pairwise model cannot see that — presented with a
mediocre candidate, it cannot know whether it is the best available or the fifth
copy of a much better match.

So each pair's hidden representation is concatenated with a **context vector**
pooled over every candidate of the same entity (mean and max, deep-sets style).
The scorer then sees both "how good is this pair" and "how good is it *relative to
the field*", which is what the exclusivity constraint actually asks.

This is the second NN variant instead of a literal siamese tower because a siamese
architecture would need the raw 512-d embeddings of both records per pair — ~40 GB
for 10M training pairs — to learn a similarity the dense cosine feature already
summarizes. Sibling competition is the signal that is genuinely unexploited.
"""

from __future__ import annotations

from .base import register
from .torch_base import TorchPairModel


@register
class EntityAttentionModel(TorchPairModel):
    name = "entity_attn"

    def build_net(self, n_features: int):
        import torch
        from torch import nn

        proj_dim = int(self.settings["proj_dim"])
        hidden = list(self.settings["hidden"])
        dropout = float(self.settings["dropout"])

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, proj_dim),
                    nn.BatchNorm1d(proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                # Input to the scorer: the pair itself plus mean- and max-pooled
                # context over its siblings.
                layers = []
                size = proj_dim * 3
                for width in hidden:
                    layers += [nn.Linear(size, width), nn.BatchNorm1d(width), nn.ReLU(),
                               nn.Dropout(dropout)]
                    size = width
                self.scorer = nn.Sequential(*layers)
                self.head = nn.Linear(size, 1)

            def forward(self, x, group_index, n_groups):
                encoded = self.encoder(x)

                counts = torch.zeros(n_groups, device=x.device, dtype=encoded.dtype)
                counts.index_add_(0, group_index, torch.ones_like(encoded[:, 0]))
                counts = counts.clamp(min=1.0).unsqueeze(-1)

                summed = torch.zeros(n_groups, encoded.shape[1], device=x.device,
                                     dtype=encoded.dtype)
                summed.index_add_(0, group_index, encoded)
                mean_ctx = summed / counts

                # scatter-max over each entity's candidates
                max_ctx = torch.full(
                    (n_groups, encoded.shape[1]), float("-inf"),
                    device=x.device, dtype=encoded.dtype,
                )
                max_ctx = max_ctx.index_reduce(
                    0, group_index, encoded, "amax", include_self=False
                )
                max_ctx = torch.nan_to_num(max_ctx, neginf=0.0)

                context = torch.cat([mean_ctx[group_index], max_ctx[group_index]], dim=-1)
                combined = torch.cat([encoded, context], dim=-1)
                return self.head(self.scorer(combined)).squeeze(-1)

        torch.manual_seed(int(self.cfg["training"]["random_seed"]))
        return Net()
