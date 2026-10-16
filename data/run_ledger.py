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
        current: dict[str, Any] = {}
        if self.status_path.is_file():
            try:
                loaded = json.loads(self.status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"existing run status is unreadable; refusing to discard it: {exc}") from exc
            if not isinstance(loaded, dict):
                raise RuntimeError("existing run status is not a JSON object; refusing to discard it")
            current = loaded
            current.pop("updated_at", None)
        merged = {**current, **status}
        if status.get("status") == "running":
            for key in ("error_type", "error_message", "failure_record"):
                merged.pop(key, None)
        if "processed_tokens" in merged and "consumed_tokens" not in status:
            merged["consumed_tokens"] = merged["processed_tokens"]
        if "consumed_tokens" in merged and "processed_tokens" not in status:
            merged["processed_tokens"] = merged["consumed_tokens"]
        if "checkpoint" in status and status["checkpoint"] is not None:
            merged["last_checkpoint"] = status["checkpoint"]
        elif "checkpoint" not in merged and merged.get("last_checkpoint") is not None:
            merged["checkpoint"] = merged["last_checkpoint"]
        payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **merged}
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

    def record_failure(self, exception_type: str, message: str, traceback_text: str) -> Path:
        """Atomically preserve the latest Python exception and mark the run failed."""
        current: dict[str, Any] = {}
        if self.status_path.is_file():
            try:
                loaded = json.loads(self.status_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
            except (OSError, json.JSONDecodeError):
                current = {}
        current.pop("updated_at", None)

        failures_dir = self.root / "failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        fd, tmp = tempfile.mkstemp(prefix=".failure-", suffix=".tmp", dir=failures_dir)
        temp_token = Path(tmp).name.removeprefix(".failure-").removesuffix(".tmp")
        failure_path = failures_dir / f"failure-{timestamp}-{os.getpid()}-{temp_token}.json"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exception_type": exception_type,
            "message": message,
            "traceback": traceback_text,
            "last_checkpoint": current.get("checkpoint"),
            "step": current.get("step"),
            "processed_tokens": current.get("processed_tokens"),
        }
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, failure_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

        failed_status = {
            **current,
            "status": "failed",
            "error_type": exception_type,
            "error_message": message,
            "failure_record": str(failure_path),
            "last_checkpoint": current.get("checkpoint"),
        }
        self.write_status(failed_status)
        return failure_path

    def consumed_tokens(self) -> int:
        if not self.status_path.exists():
            return 0
        return int(json.loads(self.status_path.read_text(encoding="utf-8")).get("consumed_tokens", 0))
