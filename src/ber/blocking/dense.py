"""Dense ANN candidate channel.

The rescue channel. The lexical channels reach 99.0% pair recall at K=50, and
**65% of what they still miss has a non-Latin business name** -- a Devanagari or
Tamil record whose address is also too sparse or too corrupted to bridge. No
string metric reaches those; a multilingual embedding does.

It is deliberately *not* the primary channel. Its top-n is unioned with the
lexical candidates rather than replacing them, so a weak embedding degrades the
result gracefully instead of capping recall.
"""

from __future__ import annotations

import logging

import numpy as np

from ..embed.backends import EmbeddingBackend
from ..embed.instruction import format_query
from ..io import Record
from ..normalize import embedding_text
from ..vectors.store import SearchResult, VectorStore

log = logging.getLogger(__name__)


class DenseChannel:
    """Queries a :class:`VectorStore` for the nearest comparable records."""

    def __init__(self, store: VectorStore, top_n: int):
        self.store = store
        self.top_n = top_n

    def search_vectors(self, country: str, vectors: np.ndarray) -> SearchResult:
        """Neighbours for pre-computed query vectors (the normal path).

        S1 vectors already exist in ``base_vectors`` from the embedding pass, so
        inference re-uses them rather than re-encoding 1.7M records.
        """
        if country not in self.store.countries():
            n = vectors.shape[0]
            return SearchResult(
                np.full((n, 0), "", dtype=object), np.zeros((n, 0), dtype=np.float32)
            )
        return self.store.search(country, vectors, self.top_n)

    async def search_records(
        self, records: list[Record], backend: EmbeddingBackend, instruction: str
    ) -> SearchResult:
        """Embed records on the fly, then search.

        Used by the benchmark and by ad-hoc queries; the full pipeline uses
        :meth:`search_vectors` against the stored base vectors instead.
        """
        if not records:
            return SearchResult(np.empty((0, 0), dtype=object), np.empty((0, 0), np.float32))
        country = records[0].country
        if any(r.country != country for r in records):
            raise ValueError("search_records expects a single-country block")
        texts = [
            format_query(embedding_text(r.name, r.address, r.country), instruction)
            for r in records
        ]
        vectors = await backend.embed(texts)
        return self.search_vectors(country, vectors)
