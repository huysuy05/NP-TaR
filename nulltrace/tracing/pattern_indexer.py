"""Find training documents that contain label tokens in label-like contexts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

DEFAULT_CONTEXT_PATTERNS: Sequence[str] = (
    "label",
    "labels",
    "category",
    "categories",
    "topic",
    "topics",
    "section",
    "tag",
    "tags",
    "class",
    "classes",
)


@dataclass
class DocumentTrace:
    """Summary of label occurrences within a document."""

    doc_id: str
    text_length: int
    label_counts: Dict[str, int]
    context_hits: Dict[str, int]
    metadata: Dict[str, object]


def _compile_regex(pattern: str, label: str) -> re.Pattern[str]:
    escaped_pattern = re.escape(pattern)
    escaped_label = re.escape(label)
    expression = rf"{escaped_pattern}\s*[:\-\|]?\s*{escaped_label}"
    return re.compile(expression, flags=re.IGNORECASE)


def _count_token_occurrences(text: str, token: str) -> int:
    return len(re.findall(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE))


def trace_document(
    doc: Mapping[str, object],
    label_tokens: Sequence[str],
    context_patterns: Sequence[str] | None = None,
    text_key: str = "text",
    doc_id_key: str = "id",
) -> DocumentTrace:
    """Return label statistics for a single document."""
    context_patterns = context_patterns or DEFAULT_CONTEXT_PATTERNS
    text = str(doc.get(text_key, ""))
    doc_id = str(doc.get(doc_id_key, doc.get("uid", "doc")))
    text_length = len(text.split())

    label_counts = {token: _count_token_occurrences(text, token) for token in label_tokens}

    context_hits: Dict[str, int] = {token: 0 for token in label_tokens}
    for token in label_tokens:
        for pattern in context_patterns:
            regex = _compile_regex(pattern, token)
            context_hits[token] += len(regex.findall(text))

    metadata = {k: v for k, v in doc.items() if k not in {text_key}}

    return DocumentTrace(
        doc_id=doc_id,
        text_length=text_length,
        label_counts=label_counts,
        context_hits=context_hits,
        metadata=metadata,
    )


def trace_corpus(
    documents: Iterable[Mapping[str, object]],
    label_tokens: Sequence[str],
    context_patterns: Sequence[str] | None = None,
    text_key: str = "text",
    doc_id_key: str = "id",
) -> Iterator[DocumentTrace]:
    """Yield traces for every document in a corpus."""
    for doc in documents:
        yield trace_document(
            doc=doc,
            label_tokens=label_tokens,
            context_patterns=context_patterns,
            text_key=text_key,
            doc_id_key=doc_id_key,
        )
