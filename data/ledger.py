"""Append-only token/train ledger. Local JSON is the source of truth."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHARD_STATUSES = ("pending", "active", "consumed", "deleted")


class LedgerError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLedger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.run_dir / "ledger.jsonl"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.status_path = self.run_dir / "status.json"

    def append(self, event: dict[str, Any]) -> None:
        row = {"ts": utc_now(), **event}
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        row = {"ts": utc_now(), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def write_status(self, status: dict[str, Any]) -> None:
        payload = {"ts": utc_now(), **status}
        self.status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        return json.loads(self.status_path.read_text(encoding="utf-8"))

    def produced_tokens(self) -> int:
        return int(self.read_status().get("produced_tokens", 0))

    def consumed_tokens(self) -> int:
        return int(self.read_status().get("consumed_tokens", 0))
