"""Feed-forward pair classifier — NN variant 1.

A straightforward MLP over the 49 pair features. Its job in the comparison is to
answer whether the trees are leaving anything on the table on this feature set,
and to be the clean baseline against which the entity-attention variant is
measured: same features, same loss, same schedule, the only difference being that
this model scores each pair in isolation.
"""

from __future__ import annotations

from .base import register
from .torch_base import TorchPairModel


@register
class MLPModel(TorchPairModel):
    name = "mlp"

    def build_net(self, n_features: int):
        import torch
        from torch import nn

        hidden = list(self.settings["hidden"])
        dropout = float(self.settings["dropout"])

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                size = n_features
                for width in hidden:
                    layers += [nn.Linear(size, width), nn.BatchNorm1d(width), nn.ReLU(),
                               nn.Dropout(dropout)]
                    size = width
                self.body = nn.Sequential(*layers)
                self.head = nn.Linear(size, 1)

            def forward(self, x, group_index, n_groups):  # noqa: ARG002 - shared signature
                return self.head(self.body(x)).squeeze(-1)

        torch.manual_seed(int(self.cfg["training"]["random_seed"]))
        return Net()
