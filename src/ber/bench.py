"""Embedder benchmark — the gate before the full embedding pass.

Embedding 24.2M records is the single largest cost in this project, and the
estimates for it span an order of magnitude depending on model, quantization and
backend. Rather than launch a multi-hour job on an estimate, this encodes a
sample, measures the actual rate, and projects the full run.

It also measures what matters more than speed: **cross-script retrieval recall**.
The dense channel exists to rescue pairs whose names are in Devanagari or Tamil, so
its value is measured on exactly those pairs -- a model that is fast but cannot
bridge scripts contributes nothing here.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time

import numpy as np

from .embed.backends import build_backend
from .embed.instruction import format_document, format_query
from .embed.run_embed import build_texts, count_records
from .io import read_records
from .normalize import embedding_text, is_non_latin

log = logging.getLogger(__name__)


def run_benchmark(cfg, split: str = "train", sample: int = 20_000) -> dict:
    """Measure throughput and cross-script recall; project the full-run time."""
    return asyncio.run(_benchmark(cfg, split, sample))


async def _benchmark(cfg, split: str, sample: int) -> dict:
    embedding = cfg["embedding"]
    backend = build_backend(cfg)
    result: dict = {
        "model": embedding["model"],
        "backend": embedding["backend"],
        "dim": embedding["dim"],
        "sample": sample,
    }
    try:
        result["health"] = await backend.health()

        records = list(
            itertools.islice(read_records(cfg.paths.source_file(split, "source1")), sample)
        )
        texts = build_texts(records, cfg.instruction)

        started = time.perf_counter()
        vectors = await _embed_all(
            backend, texts, int(embedding["batch_size"]), int(embedding["concurrency"])
        )
        elapsed = time.perf_counter() - started
        rate = len(texts) / elapsed

        total = sum(
            count_records(cfg.paths.source_file(s, src))
            for s in ("train", "test")
            for src in cfg["sources"].values()
            if cfg.paths.source_file(s, src).is_file()
        )
        bytes_per_vector = int(embedding["dim"]) * 2
        result.update(
            {
                "records_per_second": round(rate, 1),
                "elapsed_seconds": round(elapsed, 2),
                "total_records_to_embed": total,
                "projected_hours_full_run": round(total / rate / 3600, 2) if rate else None,
                "projected_storage_gb": round(total * bytes_per_vector / 1024**3, 2),
                "vector_norm_mean": round(float(np.linalg.norm(vectors, axis=1).mean()), 5),
            }
        )
        result["cross_script"] = await _cross_script_check(cfg, backend)
    finally:
        await backend.aclose()
    return result


async def _embed_all(backend, texts, batch_size, concurrency) -> np.ndarray:
    from .embed.run_embed import embed_batches

    return await embed_batches(backend, texts, batch_size, concurrency)


async def _cross_script_check(cfg, backend) -> dict:
    """Does the model place a transliterated name near its Latin original?

    A handful of real cross-script pairs from the data. The comparison is
    cosine(query form of the Latin record, document form of the non-Latin record),
    which is exactly how the pipeline uses it, against a control pair that shares a
    brand word but is a different business.
    """
    pairs = [
        ("Raj Investments LLP", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "India"),
        ("Ss Food Private Limited", "एसएस फूड प्राइवेट लिमिटेड", "India"),
        ("Ram Marketing Private Limited", "राम मार्केटिंग प्राइवेट लिमिटेड", "India"),
        ("Aditya Properties LLP", "आदित्य प्रॉपर्टीज एलएलपी", "India"),
    ]
    control = ("Helio Tech Private Limited", "Helio Constructions Private Limited", "India")

    instruction = cfg.instruction
    queries = [format_query(embedding_text(a, "", c), instruction) for a, _b, c in pairs]
    documents = [format_document(embedding_text(b, "", c)) for _a, b, c in pairs]
    queries.append(format_query(embedding_text(control[0], "", control[2]), instruction))
    documents.append(format_document(embedding_text(control[1], "", control[2])))

    q_vectors = await backend.embed(queries)
    d_vectors = await backend.embed(documents)
    similarities = (q_vectors * d_vectors).sum(axis=1)

    true_pairs = similarities[:-1]
    return {
        "cross_script_pairs": [
            {"latin": a, "other": b, "cosine": round(float(s), 4), "non_latin": is_non_latin(b)}
            for (a, b, _c), s in zip(pairs, true_pairs, strict=True)
        ],
        "mean_true_pair_cosine": round(float(true_pairs.mean()), 4),
        "control_same_brand_different_industry": round(float(similarities[-1]), 4),
        "separates_control": bool(true_pairs.mean() > similarities[-1]),
    }
