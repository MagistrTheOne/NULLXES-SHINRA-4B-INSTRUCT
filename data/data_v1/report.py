"""Machine-readable Phase B canary report. Not a README."""

from __future__ import annotations

from typing import Any

MAX_CANARY_TOKENS = 20_000_000

REPORT_KEYS = (
    "status",
    "documents_seen",
    "documents_kept",
    "documents_rejected",
    "tokens_kept",
    "exact_duplicates",
    "near_duplicates",
    "probe_collisions",
    "special_token_violations",
    "domains",
    "languages",
    "sources",
)


def empty_report(*, max_canary_tokens: int = MAX_CANARY_TOKENS, run_mode: str = "fixture") -> dict[str, Any]:
    if max_canary_tokens > MAX_CANARY_TOKENS:
        raise ValueError("cannot raise Phase B canary token cap")
    if run_mode not in {"fixture", "canary"}:
        raise ValueError("run_mode must be fixture|canary")
    return {
        "status": "fail",
        "documents_seen": 0,
        "documents_kept": 0,
        "documents_rejected": 0,
        "tokens_kept": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "probe_collisions": 0,
        "special_token_violations": 0,
        "domains": {},
        "languages": {},
        "sources": {},
        "phase": "B",
        "training": False,
        "network": False,
        "max_canary_tokens": max_canary_tokens,
        "token_count_mode": "whitespace",
        "s0_v0_generators": 0,
        "raw_url_sidecar": 0,
        "over_budget": 0,
        "sidecar_invalid": 0,
        "domain_violations": 0,
        "split_violations": 0,
        "language_violations": 0,
        "langid_mismatch": 0,
        "allowlist_misses": 0,
        "run_mode": run_mode,
        "disk_free_gb": None,
        "input_bytes": 0,
        "reject_reasons": {},
        "gates": {},
    }


def bump(mapping: dict[str, int], key: str, n: int = 1) -> None:
    mapping[key] = int(mapping.get(key, 0)) + n


def add_reject(report: dict[str, Any], reason: str) -> None:
    report["documents_rejected"] += 1
    bump(report["reject_reasons"], reason)


def finalize_status(report: dict[str, Any]) -> dict[str, Any]:
    max_tokens = int(report["max_canary_tokens"])
    gates = {
        "max_canary_tokens": {
            "limit": max_tokens,
            "value": report["tokens_kept"],
            "ok": report["tokens_kept"] <= max_tokens and report["over_budget"] == 0,
        },
        "s0_v0_generators": {"count": report["s0_v0_generators"], "ok": report["s0_v0_generators"] == 0},
        "probe_overlap": {"count": report["probe_collisions"], "ok": report["probe_collisions"] == 0},
        "raw_url_sidecar": {"count": report["raw_url_sidecar"], "ok": report["raw_url_sidecar"] == 0},
        "special_token_ids": {
            "count": report["special_token_violations"],
            "ok": report["special_token_violations"] == 0,
        },
        "split": {"ok": report["split_violations"] == 0},
        "domain": {"ok": report["domain_violations"] == 0},
        "language": {"ok": report["language_violations"] == 0 and report["langid_mismatch"] == 0},
        "kept_nonempty": {"ok": report["documents_kept"] > 0},
        "token_count_mode": {
            "value": report["token_count_mode"],
            "ok": (
                (report["run_mode"] == "fixture" and report["token_count_mode"] in {"whitespace", "tokenizer"})
                or (report["run_mode"] == "canary" and report["token_count_mode"] == "tokenizer")
            ),
        },
        "allowlist": {
            "ok": report["run_mode"] == "fixture" or report.get("allowlist_misses", 0) == 0
        },
    }
    report["gates"] = gates
    report["status"] = "pass" if all(g["ok"] for g in gates.values()) else "fail"
    return report


def public_report(report: dict[str, Any]) -> dict[str, Any]:
    """User-facing required keys first; extras remain for the dry-run."""
    out = {k: report[k] for k in REPORT_KEYS}
    for key, value in report.items():
        if key not in out:
            out[key] = value
    return out
