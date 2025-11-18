"""Utility helpers for working with JSONL corpora."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping


def read_jsonl(path: str | Path) -> List[MutableMapping[str, object]]:
    """Load a JSONL file into memory."""
    records: List[MutableMapping[str, object]] = []
    for record in stream_jsonl(path):
        records.append(record)
    return records


def stream_jsonl(path: str | Path) -> Iterator[MutableMapping[str, object]]:
    """Yield records from a JSON lines file one by one."""
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records = json.loads(line)
            if isinstance(records, Mapping):
                yield dict(records)
            else:
                raise ValueError(f"Expected mapping per line, got {type(records)}")


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> None:
    """Persist an iterable of mapping objects to JSONL."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")
