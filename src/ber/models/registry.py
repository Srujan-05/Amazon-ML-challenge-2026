"""Lazy model registry.

Models are resolved by name and imported **only when needed**. That is a hard
requirement on macOS, not a style preference: PyTorch, scikit-learn and FAISS each
ship their own ``libomp.dylib``, and LightGBM loads another. With PyTorch already
imported, LightGBM's first ``Dataset`` construction segfaults -- reproducibly, and
not fixable with ``KMP_DUPLICATE_LIB_OK``.

Eagerly importing every model to populate a registry would therefore make the tree
models uninstantiable in any process that had touched torch. Keeping the mapping as
strings, and combining it with the one-subprocess-per-model training in
``ber.train``, means the two OpenMP runtimes never meet.
"""

from __future__ import annotations

import importlib

#: name -> (module, class). Order is the default training and reporting order.
MODEL_SPECS: dict[str, tuple[str, str]] = {
    "lightgbm": ("ber.models.lightgbm_model", "LightGBMModel"),
    "xgboost": ("ber.models.xgboost_model", "XGBoostModel"),
    "mlp": ("ber.models.mlp", "MLPModel"),
    "entity_attn": ("ber.models.entity_attn", "EntityAttentionModel"),
}

DEFAULT_MODELS = list(MODEL_SPECS)

#: Models that must not share a process with a torch model.
TREE_MODELS = frozenset({"lightgbm", "xgboost"})


def model_names() -> list[str]:
    return list(MODEL_SPECS)


def get_model_class(name: str):
    """Import and return the model class for ``name``."""
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_SPECS)}")
    module_path, class_name = MODEL_SPECS[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build(name: str, cfg):
    """Instantiate a model by name."""
    return get_model_class(name)(cfg)
