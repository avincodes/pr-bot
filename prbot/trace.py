"""Structured JSON trace logging for agent runs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceLogger:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    root: Path = field(default_factory=lambda: Path("traces"))
    _fh: Any = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{self.run_id}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
