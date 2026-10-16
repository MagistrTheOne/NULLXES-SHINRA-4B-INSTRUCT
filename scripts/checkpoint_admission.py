"""Admission labels that keep audit-only diagnostic checkpoints out of production paths."""

from __future__ import annotations

import json
from pathlib import Path


def checkpoint_admission(checkpoint: str | Path) -> str:
    root = Path(checkpoint)
    try:
        record = json.loads((root / "training_admission.json").read_text(encoding="utf-8"))
        model_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint admission metadata is missing or unreadable: {root}") from exc
    admission = record.get("admission")
    if (record.get("schema") != "nullxes.shinra.training-admission.v1"
            or admission not in {"production", "diagnostic_audit"}
            or model_config.get("shinra_training_admission") != admission
            or record.get("production_usable") is not (admission == "production")):
        raise ValueError(f"checkpoint admission metadata is inconsistent: {root}")
    return admission


def require_production_checkpoint(checkpoint: str | Path) -> None:
    if checkpoint_admission(checkpoint) != "production":
        raise ValueError("audit-only diagnostic checkpoints are not admitted for inference or publication")
