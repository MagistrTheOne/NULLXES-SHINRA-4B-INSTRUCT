"""Atomic JSONL metrics and status ledger for one training run."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunLedger:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.metrics_path = self.root / "metrics.jsonl"
        self.status_path = self.root / "status.json"

    def append(self, event: dict[str, Any]) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_status(self, status: dict[str, Any]) -> None:
        payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **status}
        fd, tmp = tempfile.mkstemp(prefix=".status-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.status_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def consumed_tokens(self) -> int:
        if not self.status_path.exists():
            return 0
        return int(json.loads(self.status_path.read_text(encoding="utf-8")).get("consumed_tokens", 0))
