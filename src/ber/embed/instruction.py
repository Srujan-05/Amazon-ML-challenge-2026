"""Instruction formatting for the embedder.

Qwen3-Embedding is instruction-aware: the *query* side carries a task
instruction, the *document* side does not. That asymmetry lines up with how this
pipeline retrieves -- every Source-1 entity is a query issued against the pool of
Source-2/3 records -- so S1 records are embedded as queries and S2/S3 as
documents.

The instruction text itself lives in ``configs/instruction.txt`` and is meant to
be edited. Nothing here hardcodes it.
"""

from __future__ import annotations

QUERY_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"

#: Records from this source are embedded as instructed queries; all others as
#: plain documents. S1 is the deduplicated reference source we search *from*.
QUERY_SOURCE = "S1"


def format_query(text: str, instruction: str) -> str:
    """Wrap a record in the instruction, for the side we retrieve from."""
    return QUERY_TEMPLATE.format(instruction=instruction.strip(), text=text)


def format_document(text: str) -> str:
    """Documents are embedded bare -- the convention the model was trained on."""
    return text


def format_for_source(text: str, source: str, instruction: str) -> str:
    """Pick the query or document form based on the record's source prefix."""
    if source == QUERY_SOURCE:
        return format_query(text, instruction)
    return format_document(text)
