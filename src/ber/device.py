"""Accelerator selection for the classifier models.

``CLAUDE.md`` asks for automatic CUDA/MPS switching. Note the embedder does not
use this: it runs behind an HTTP endpoint (llama.cpp Metal locally, vLLM on
CUDA remotely), so its acceleration is the server's concern, not torch's.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def pick_device(preference: str = "auto") -> str:
    """Return a torch device string, honouring an explicit preference.

    ``auto`` prefers MPS, then CUDA, then CPU. An explicit choice that is not
    available falls back with a warning rather than failing the run.
    """
    import torch

    available = {"cpu": True}
    available["cuda"] = torch.cuda.is_available()
    available["mps"] = torch.backends.mps.is_available()

    if preference != "auto":
        if available.get(preference):
            return preference
        log.warning("device %r unavailable; falling back to auto selection", preference)

    for candidate in ("mps", "cuda", "cpu"):
        if available.get(candidate):
            if candidate != "cpu":
                log.info("using %s accelerator", candidate)
            return candidate
    return "cpu"
