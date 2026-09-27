"""Pair feature extraction.

Blocking decides what is *reachable*; these features decide what is *right*.

The design is driven by one measured failure of naive similarity: ``Helio Tech
Pvt`` and ``Helio Constructions Pvt`` score **83.9** on rapidfuzz
``token_set_ratio`` despite being different businesses. Plain string similarity
cannot separate them, because the shared token ``Helio`` is common to both and the
metric does not know that ``Tech`` and ``Constructions`` are the discriminative
parts. So the feature set carries **IDF-weighted** overlap and an explicit
``rare_token_mismatch``: the rarest token present on exactly one side. That is
the signal that makes the call.

Three more measured facts shape the rest:

* 12.2% of true matches have zero name-token overlap, so a romanized
  repeat-squeezed ``skeleton`` similarity is computed alongside the surface form.
* 3.3% of S2/S3 records have no address at all, so address features carry
  explicit presence flags rather than a misleading 0.0.
* Each S2/S3 record belongs to at most one S1 entity, so ``contention`` -- how many
  S1 entities are competing for this record -- is a real signal, and the only
  feature that lets a per-pair model anticipate the exclusivity constraint.

Derived per-record data is cached, because with K=50 each comparable record is
touched by several entities and tokenization is the dominant cost.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from .io import Record
from .normalize import (
    acronym,
    address_tokens,
    char_ngrams,
    digit_tokens,
    is_non_latin,
    name_tokens,
    romanized_tokens,
    squeeze_repeats,
)

#: Legal forms worth comparing directly -- a Pvt Ltd and an LLP at the same
#: address are usually different registrations, not a match.
LEGAL_MARKERS = frozenset(
    {"private", "limited", "llp", "llc", "incorporated", "corporation",
     "company", "sarl", "sasu", "sas", "sci", "sa", "plc", "gmbh"}
)

FEATURE_NAMES: list[str] = [
    # name, surface form
    "name_jaccard", "name_containment", "name_idf_jaccard", "name_idf_containment",
    "rare_token_mismatch", "shared_max_idf",
    "name_trigram_jaccard", "name_ratio", "name_token_set_ratio", "name_partial_ratio",
    "name_acronym_match", "name_len_ratio", "name_token_count_diff",
    # name, cross-script
    "skeleton_ratio", "skeleton_token_set_ratio", "script_mismatch", "both_non_latin",
    # legal form
    "legal_jaccard", "legal_both_present",
    # address
    "addr_jaccard", "addr_containment", "addr_idf_jaccard", "addr_idf_containment",
    "addr_trigram_jaccard", "addr_token_set_ratio", "addr_ratio",
    "digit_overlap", "digit_jaccard", "digit_count_min",
    "addr_present_both", "addr_missing_one", "addr_len_ratio", "tail_token_match",
    # channel evidence
    "lex_score", "lex_rank", "lex_found", "dense_score", "dense_rank", "dense_found",
    "rrf", "found_by_both",
    # structural
    "cand_is_s2", "cand_is_s3", "rank_in_entity", "n_candidates",
    "lex_margin", "dense_margin", "rrf_margin", "contention",
]

N_FEATURES = len(FEATURE_NAMES)


def build_idf(records: list[Record], kind: str) -> dict[str, float]:
    """Inverse document frequency over one country's comparable records.

    Country-scoped because token rarity is local: ``chennai`` is common in the
    India partition and absent from the US one.
    """
    df: collections.Counter[str] = collections.Counter()
    tokenizer = name_tokens if kind == "name" else address_tokens
    for record in records:
        text = record.name if kind == "name" else record.address
        df.update(set(tokenizer(text, record.country)))
    n = max(len(records), 1)
    return {token: math.log((1.0 + n) / (1.0 + count)) + 1.0 for token, count in df.items()}


@dataclass(slots=True)
class Derived:
    """Everything about one record that pair features need, computed once."""

    name_tokens: frozenset[str]
    addr_tokens: frozenset[str]
    name_trigrams: frozenset[str]
    addr_trigrams: frozenset[str]
    digits: frozenset[str]
    norm_name: str
    norm_addr: str
    skeleton: str
    skeleton_tokens: frozenset[str]
    acronym: str
    legal: frozenset[str]
    non_latin: bool
    tail: str
    n_name_tokens: int


def derive(record: Record) -> Derived:
    """Compute the cached per-record view."""
    country = record.country
    ntok = name_tokens(record.name, country)
    atok = address_tokens(record.address, country)
    roman = [squeeze_repeats(t) for t in romanized_tokens(record.name, country, kind="name")]
    skeleton = " ".join(roman)
    norm_name = " ".join(ntok)
    norm_addr = " ".join(atok)
    return Derived(
        name_tokens=frozenset(ntok),
        addr_tokens=frozenset(atok),
        name_trigrams=frozenset(char_ngrams("".join(ntok), 3)),
        addr_trigrams=frozenset(char_ngrams(norm_addr, 3)),
        digits=frozenset(digit_tokens(record.address, min_digits=1)),
        norm_name=norm_name,
        norm_addr=norm_addr,
        skeleton=skeleton,
        skeleton_tokens=frozenset(roman),
        acronym=acronym(ntok),
        legal=frozenset(ntok) & LEGAL_MARKERS,
        non_latin=is_non_latin(record.name),
        # The last address token is usually the state/region after normalization.
        tail=atok[-1] if atok else "",
        n_name_tokens=len(ntok),
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap over the smaller set -- tolerant of one side being verbose."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _idf_overlap(
    a: frozenset[str], b: frozenset[str], idf: dict[str, float], default: float
) -> tuple[float, float, float, float]:
    """IDF-weighted Jaccard and containment, plus rarest-mismatch and rarest-shared.

    ``rare_token_mismatch`` is the feature that separates ``Helio Tech`` from
    ``Helio Constructions``: both share a common brand token, but each carries a
    rare token the other lacks.
    """
    if not a or not b:
        return 0.0, 0.0, 0.0, 0.0
    shared = a & b
    union = a | b
    only = union - shared
    w_shared = sum(idf.get(t, default) for t in shared)
    w_union = sum(idf.get(t, default) for t in union)
    w_a = sum(idf.get(t, default) for t in a)
    w_b = sum(idf.get(t, default) for t in b)
    jaccard = w_shared / w_union if w_union else 0.0
    containment = w_shared / min(w_a, w_b) if min(w_a, w_b) > 0 else 0.0
    rare_mismatch = max((idf.get(t, default) for t in only), default=0.0)
    max_shared = max((idf.get(t, default) for t in shared), default=0.0)
    return jaccard, containment, rare_mismatch, max_shared


def _ratio(a: float, b: float) -> float:
    """Symmetric size ratio in [0, 1]."""
    hi = max(a, b)
    return min(a, b) / hi if hi > 0 else 0.0


@dataclass
class PairContext:
    """Per-entity context that a single pair cannot see on its own."""

    n_candidates: int
    best_lex: float
    best_dense: float
    best_rrf: float
    contention: collections.Counter = field(default_factory=collections.Counter)


class FeatureExtractor:
    """Builds feature vectors for candidate pairs of one country."""

    def __init__(
        self,
        name_idf: dict[str, float],
        addr_idf: dict[str, float],
        cache_limit: int = 2_000_000,
    ):
        self.name_idf = name_idf
        self.addr_idf = addr_idf
        # Unseen tokens are maximally rare; approximate with the observed ceiling.
        self.name_default = max(name_idf.values(), default=1.0)
        self.addr_default = max(addr_idf.values(), default=1.0)
        self._cache: dict[str, Derived] = {}
        self._cache_limit = cache_limit

    def derived(self, record: Record) -> Derived:
        got = self._cache.get(record.entity_id)
        if got is None:
            got = derive(record)
            if len(self._cache) < self._cache_limit:
                self._cache[record.entity_id] = got
        return got

    def clear(self) -> None:
        self._cache.clear()

    def pair(
        self,
        s1: Record,
        cand: Record,
        lex_score: float,
        lex_rank: int,
        dense_score: float,
        dense_rank: int,
        rrf: float,
        rank_in_entity: int,
        context: PairContext,
    ) -> np.ndarray:
        """Feature vector for one candidate pair, ordered as :data:`FEATURE_NAMES`."""
        a, b = self.derived(s1), self.derived(cand)

        n_idf_j, n_idf_c, rare_mismatch, shared_max = _idf_overlap(
            a.name_tokens, b.name_tokens, self.name_idf, self.name_default
        )
        a_idf_j, a_idf_c, _, _ = _idf_overlap(
            a.addr_tokens, b.addr_tokens, self.addr_idf, self.addr_default
        )
        shared_digits = a.digits & b.digits
        addr_both = bool(a.addr_tokens) and bool(b.addr_tokens)

        values = [
            _jaccard(a.name_tokens, b.name_tokens),
            _containment(a.name_tokens, b.name_tokens),
            n_idf_j, n_idf_c, rare_mismatch, shared_max,
            _jaccard(a.name_trigrams, b.name_trigrams),
            fuzz.ratio(a.norm_name, b.norm_name) / 100.0,
            fuzz.token_set_ratio(a.norm_name, b.norm_name) / 100.0,
            fuzz.partial_ratio(a.norm_name, b.norm_name) / 100.0,
            float(bool(a.acronym) and a.acronym == b.acronym),
            _ratio(len(a.norm_name), len(b.norm_name)),
            float(abs(a.n_name_tokens - b.n_name_tokens)),
            # Skeletons are compared symmetrically -- squeezing is destructive, so
            # it is only meaningful when both sides get the same treatment.
            fuzz.ratio(a.skeleton, b.skeleton) / 100.0,
            fuzz.token_set_ratio(a.skeleton, b.skeleton) / 100.0,
            float(a.non_latin != b.non_latin),
            float(a.non_latin and b.non_latin),
            _jaccard(a.legal, b.legal),
            float(bool(a.legal) and bool(b.legal)),
            _jaccard(a.addr_tokens, b.addr_tokens),
            _containment(a.addr_tokens, b.addr_tokens),
            a_idf_j, a_idf_c,
            _jaccard(a.addr_trigrams, b.addr_trigrams),
            fuzz.token_set_ratio(a.norm_addr, b.norm_addr) / 100.0,
            fuzz.ratio(a.norm_addr, b.norm_addr) / 100.0,
            float(len(shared_digits)),
            _jaccard(a.digits, b.digits),
            float(min(len(a.digits), len(b.digits))),
            float(addr_both),
            float(bool(a.addr_tokens) != bool(b.addr_tokens)),
            _ratio(len(a.norm_addr), len(b.norm_addr)),
            float(bool(a.tail) and a.tail == b.tail),
            lex_score, float(lex_rank), float(lex_rank >= 0),
            dense_score, float(dense_rank), float(dense_rank >= 0),
            rrf, float(lex_rank >= 0 and dense_rank >= 0),
            float(cand.entity_id.startswith("S2-")),
            float(cand.entity_id.startswith("S3-")),
            float(rank_in_entity),
            float(context.n_candidates),
            lex_score - context.best_lex,
            dense_score - context.best_dense,
            rrf - context.best_rrf,
            float(context.contention.get(cand.entity_id, 1)),
        ]
        return np.asarray(values, dtype=np.float32)
