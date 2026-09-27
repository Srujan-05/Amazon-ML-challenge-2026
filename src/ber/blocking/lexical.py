"""Lexical candidate generation.

This is the channel that carries recall. Measured on the training ground truth,
address and name token overlap alone reaches a **~99.4% ceiling**: only 0.58% of
true pairs have both name and address Jaccard <= 0.2.

Why a typed-token inverted index rather than char-3gram TF-IDF cosine: the
comparable side holds up to 4.72M records per country. A full
``X_query @ X_comparable.T`` over character trigrams produces a product with
nearly no sparsity -- common trigrams like ``" st"`` link a query to almost every
document, so the intermediate would be billions of nonzeros. Word-level tokens
with aggressive document-frequency filtering keep that product genuinely sparse,
which is what makes the matmul tractable. Character n-grams are still used, but
as *features* on the <=50 surviving candidates, where they cost nothing.

Four token channels are emitted per record, each with its own prefix so an
address token can never match a name token, and its own weight:

``d:``  digit tokens (>=3 digits) -- 80.3% of true matches share one, so boosted
``a:``  address tokens
``n:``  name tokens
``s:``  romanized, repeat-squeezed name tokens -- the cross-script channel
``g:``  character 3-grams of the *concatenated* romanized name

The ``g:`` channel earns its place on two miss patterns found by measurement:
names recorded as domains (``backalleybistr0.com`` for "Back Alley Bistro"),
where word tokens cannot align at all, and single-character typos
(``Bnlistdft`` for "Belinsoft") on records whose address is empty. It is applied
to names only -- they are short, so the cost is bounded, unlike trigrams over
4.7M addresses.

Hashing is used instead of a vocabulary dict: at 4.7M records the vocabulary runs
to millions of distinct house numbers and PIN codes, and a Python dict of that
size costs more than the postings themselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..io import Record
from ..normalize import address_tokens, digit_tokens, name_tokens, romanized_tokens, squeeze_repeats

log = logging.getLogger(__name__)

#: 2^22 buckets. Collisions create a few spurious candidates, which the
#: classifier discards -- far cheaper than a multi-million-entry vocabulary.
N_FEATURES = 2**22

#: Default channel weights, overridden by ``blocking.lexical.channel_weights``.
#: Tuned by sweep against training ground truth (4k entities, 714k comparable
#: pool). These reach **99.0% pair recall at K=50**, versus 97.2% for a flat
#: weighting and 98.7% without the trigram channel. Over-boosting the digit
#: channel actively hurts (d=3.0 -> 92.5%): a shared house number is strong
#: evidence, but there are only so many house numbers, so it crowds out the name
#: signal. Re-run ``ber block --sweep`` after changing the normalizer.
CHANNEL_WEIGHTS = {"d": 1.3, "a": 1.2, "n": 1.0, "s": 0.9, "g": 0.6}

#: Domain-style names carry no useful token structure once the TLD is stripped.
_TLDS = (".com", ".net", ".org", ".co", ".io", ".in", ".fr", ".biz", ".info")


def record_keys(record: Record, min_digits: int = 3) -> list[str]:
    """Typed blocking keys for one record.

    The romanized channel is only emitted for non-Latin names, where it is the
    only thing that can bridge ``ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`` to ``Raj Investments``.
    For Latin names it would duplicate the ``n:`` channel.
    """
    country = record.country
    keys = [f"a:{t}" for t in address_tokens(record.address, country)]
    keys += [f"n:{t}" for t in name_tokens(record.name, country)]
    keys += [f"d:{t}" for t in digit_tokens(record.address, min_digits)]
    roman = romanized_tokens(record.name, country, kind="name")
    squeezed = [squeeze_repeats(t) for t in roman]
    keys += [f"s:{t}" for t in squeezed]
    keys += [f"g:{g}" for g in name_ngrams(record.name, country)]
    return keys


def name_ngrams(name: str, country: str, n: int = 3) -> set[str]:
    """Character n-grams over the concatenated romanized name.

    Concatenating before slicing is deliberate: it makes ``backalleybistro`` and
    ``Back Alley Bistro`` produce the same trigrams, which is what rescues
    domain-style names.
    """
    tokens = romanized_tokens(name, country, kind="name")
    joined = "".join(tokens)
    for tld in _TLDS:
        if joined.endswith(tld.replace(".", "")):
            joined = joined[: -len(tld) + 1]
            break
    if len(joined) < n:
        return {joined} if joined else set()
    return {joined[i : i + n] for i in range(len(joined) - n + 1)}


def _hash_keys(keys: list[str]) -> np.ndarray:
    """Stable 32-bit bucket per key.

    ``hash()`` is salted per process, so it cannot be used: the comparable index
    is built in one run and queried in another.
    """
    import hashlib

    out = np.empty(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "little") % N_FEATURES
    return out


def build_matrix(
    key_lists: list[list[str]], weights: dict[str, float] | None = None
) -> sp.csr_matrix:
    """Binary record-by-bucket sparse matrix, with per-channel weights applied."""
    weights = weights or CHANNEL_WEIGHTS
    indptr = np.zeros(len(key_lists) + 1, dtype=np.int64)
    all_indices: list[np.ndarray] = []
    all_data: list[np.ndarray] = []
    for i, keys in enumerate(key_lists):
        if keys:
            buckets = _hash_keys(keys)
            row_weights = np.asarray(
                [weights.get(k[0], 1.0) for k in keys], dtype=np.float32
            )
            # A key repeated inside one record carries no extra evidence.
            buckets, first = np.unique(buckets, return_index=True)
            row_weights = row_weights[first]
        else:
            buckets = np.empty(0, dtype=np.int64)
            row_weights = np.empty(0, dtype=np.float32)
        all_indices.append(buckets)
        all_data.append(row_weights)
        indptr[i + 1] = indptr[i] + buckets.size
    indices = np.concatenate(all_indices) if all_indices else np.empty(0, dtype=np.int64)
    data = np.concatenate(all_data) if all_data else np.empty(0, dtype=np.float32)
    return sp.csr_matrix(
        (data, indices.astype(np.int32), indptr), shape=(len(key_lists), N_FEATURES)
    )


@dataclass
class LexicalIndex:
    """Inverted index over one country's comparable records.

    ``idf`` and ``keep`` are learned from the comparable side and then applied
    unchanged to queries, so both sides live in the same weighted space.
    """

    ids: np.ndarray
    matrix: sp.csr_matrix                      # L2-normalized, IDF-weighted
    idf: np.ndarray = field(repr=False)
    keep: np.ndarray = field(repr=False)
    weights: dict[str, float] = field(default_factory=lambda: dict(CHANNEL_WEIGHTS))

    @property
    def n_records(self) -> int:
        return self.matrix.shape[0]

    @classmethod
    def build(
        cls,
        records: list[Record],
        min_digits: int = 3,
        max_df_ratio: float = 0.3,
        min_df: int = 1,
        weights: dict[str, float] | None = None,
    ) -> LexicalIndex:
        """Build from the comparable-side records of one country.

        ``max_df_ratio`` is the lever that keeps the query matmul sparse: a bucket
        present in more than that fraction of records is dropped outright rather
        than merely down-weighted, because IDF weighting alone would still leave
        its postings in the product.
        """
        weights = dict(weights or CHANNEL_WEIGHTS)
        ids = np.asarray([r.entity_id for r in records], dtype=object)
        matrix = build_matrix([record_keys(r, min_digits) for r in records], weights)
        n = matrix.shape[0]

        df = np.asarray((matrix > 0).sum(axis=0)).ravel()
        max_df = max(int(max_df_ratio * n), 1)
        keep = (df >= min_df) & (df <= max_df)
        idf = np.zeros(N_FEATURES, dtype=np.float32)
        good = keep & (df > 0)
        idf[good] = np.log((1.0 + n) / (1.0 + df[good])).astype(np.float32) + 1.0

        log.info(
            "lexical index: %d records, %d active buckets (dropped %d over df>%d)",
            n, int(good.sum()), int(((df > max_df)).sum()), max_df,
        )
        weighted = _apply_weights(matrix, idf)
        return cls(ids=ids, matrix=weighted, idf=idf, keep=keep, weights=weights)

    def query(self, records: list[Record], top_n: int, min_digits: int = 3):
        """Top-``top_n`` comparable records per query record.

        Returns ``(indices, scores)`` arrays of shape ``(len(records), top_n)``,
        padded with ``-1`` / ``0.0`` where a query had fewer hits.
        """
        query_matrix = _apply_weights(
            build_matrix([record_keys(r, min_digits) for r in records], self.weights), self.idf
        )
        # Sparse in both operands, and DF-filtering keeps the product sparse.
        product = (query_matrix @ self.matrix.T).tocsr()
        return _top_n_per_row(product, top_n)


def _apply_weights(matrix: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Multiply by IDF in place, drop zeroed buckets, then L2-normalize rows."""
    from sklearn.preprocessing import normalize

    matrix = matrix.copy()
    matrix.data *= idf[matrix.indices]
    matrix.eliminate_zeros()
    return normalize(matrix, norm="l2", axis=1, copy=False)


def _top_n_per_row(product: sp.csr_matrix, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row top-``top_n`` by score, descending."""
    n_rows = product.shape[0]
    out_idx = np.full((n_rows, top_n), -1, dtype=np.int64)
    out_score = np.zeros((n_rows, top_n), dtype=np.float32)
    indptr, indices, data = product.indptr, product.indices, product.data
    for row in range(n_rows):
        start, stop = indptr[row], indptr[row + 1]
        if start == stop:
            continue
        row_scores = data[start:stop]
        row_indices = indices[start:stop]
        if row_scores.size > top_n:
            part = np.argpartition(row_scores, -top_n)[-top_n:]
            row_scores, row_indices = row_scores[part], row_indices[part]
        order = np.argsort(-row_scores)
        take = order.size
        out_idx[row, :take] = row_indices[order]
        out_score[row, :take] = row_scores[order]
    return out_idx, out_score
