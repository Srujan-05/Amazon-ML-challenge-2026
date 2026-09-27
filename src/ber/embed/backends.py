"""Embedding backends behind one interface.

All three speak the same ``embed(texts) -> (n, dim) float32`` contract, so the
pipeline does not know or care which is running:

``llamacpp``
    ``llama-server --embedding`` on Apple Silicon. Metal-accelerated, and the
    only local option that uses the GPU -- vLLM forces
    ``VLLM_TARGET_DEVICE=cpu`` on macOS and runs 20-30x slower.
``vllm``
    ``vllm serve --task embed`` on CUDA. Same OpenAI-compatible wire format, so
    moving to AWS is a config change.
``torch-mps``
    In-process transformers, for when running a server is not worth it.

Vectors come back L2-normalized at ``dim`` dimensions. Qwen3-Embedding supports
MRL, so truncating the native 1024 to 512 and renormalizing is a supported
operation rather than a hack -- it is how the model is designed to be shortened.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return vectors / norms


def apply_mrl(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Truncate to ``dim`` dimensions and renormalize (the MRL protocol).

    Truncation alone leaves the vector off the unit sphere, which breaks the
    equivalence between inner product and cosine that the index relies on.
    """
    if dim and vectors.shape[1] > dim:
        vectors = vectors[:, :dim]
    return l2_normalize(np.ascontiguousarray(vectors, dtype=np.float32))


class EmbeddingBackend(ABC):
    """Contract shared by every backend."""

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch, returning ``(len(texts), dim)`` float32, L2-normalized."""

    async def health(self) -> str:
        """Human-readable readiness description; raises if unusable."""
        probe = await self.embed(["health check"])
        return f"{type(self).__name__} ok, native dim {probe.shape[1]}"

    async def aclose(self) -> None:
        """Release resources. Safe to call more than once; no-op by default."""
        return None


class OpenAIEmbeddingsBackend(EmbeddingBackend):
    """Any server exposing OpenAI's ``/v1/embeddings`` -- llama.cpp or vLLM.

    Retries with exponential backoff: this runs for hours over millions of
    records, and a transient connection reset must not lose a shard.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        dim: int,
        timeout: float = 600.0,
        max_retries: int = 5,
    ):
        super().__init__(dim)
        self.endpoint = endpoint
        self.model = model
        self.max_retries = max_retries
        import httpx

        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        payload = {"model": self.model, "input": texts}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(self.endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
                # The API does not promise ordering; `index` is authoritative.
                rows = sorted(body["data"], key=lambda item: item["index"])
                if len(rows) != len(texts):
                    raise ValueError(f"expected {len(texts)} embeddings, got {len(rows)}")
                raw = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
                return apply_mrl(raw, self.dim)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                delay = 2.0**attempt
                log.warning(
                    "embed batch failed (attempt %d/%d): %s; retrying in %.0fs",
                    attempt + 1, self.max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"embedding failed after {self.max_retries} attempts") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()


class TorchBackend(EmbeddingBackend):
    """In-process transformers encoder, no server.

    Uses last-token pooling, which is what Qwen3-Embedding is trained with.
    Chosen device follows the usual MPS -> CUDA -> CPU preference.
    """

    def __init__(self, model: str, dim: int, max_tokens: int = 96, device: str | None = None):
        super().__init__(dim)
        import torch
        from transformers import AutoModel, AutoTokenizer

        from ..device import pick_device

        self.torch = torch
        self.device = device or pick_device("auto")
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model, dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device).eval()

    async def embed(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        batch = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_tokens, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model(**batch).last_hidden_state
            # Left padding means the final position is always a real token.
            pooled = output[:, -1, :]
        return apply_mrl(pooled.float().cpu().numpy(), self.dim)


def build_backend(cfg) -> EmbeddingBackend:
    """Construct the backend named by ``embedding.backend`` in the config."""
    embedding = cfg["embedding"]
    kind = embedding["backend"]
    dim = embedding["dim"]
    if kind in ("llamacpp", "vllm"):
        return OpenAIEmbeddingsBackend(
            endpoint=embedding["endpoint"], model=embedding["model"], dim=dim
        )
    if kind == "torch-mps":
        return TorchBackend(
            model=embedding["model"], dim=dim, max_tokens=embedding["max_tokens"]
        )
    raise ValueError(f"unknown embedding backend {kind!r}")
