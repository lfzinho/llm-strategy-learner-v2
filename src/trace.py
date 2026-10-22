"""Append-only JSONL trace of every LLM call, so a failed run can be diagnosed afterwards."""
import json
import os
import time
from typing import Any, Dict, Iterator, Optional


class TraceLog:
    """One JSON object per line. Disabled (no-op) until a path is set."""

    def __init__(self, path: Optional[str] = None):
        self.path = path

    def set_path(self, path: Optional[str]):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, kind: str, **fields: Any):
        if not self.path:
            return
        record: Dict[str, Any] = {"ts": round(time.time(), 3), "kind": kind, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_trace(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
