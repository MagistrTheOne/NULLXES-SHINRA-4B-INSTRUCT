"""Phase B corpus canary: local ingest, hard gates, JSON report. No training. No network."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from data.data_v1 import PhaseBError
from data.data_v1.classify import resolve_domain, resolve_language, resolve_source_type, resolve_split
from data.data_v1.contamination import (
    assert_no_probe_overlap,
    assert_no_v0_generator,
    assert_sidecar_offline,
    forbidden_token_ids,
    frozen_probe_fingerprints,
    special_surface_hits,
)
from data.data_v1.dedup import CanaryDedup
from data.data_v1.ingest import ensure_local_path, iter_records
from data.data_v1.normalize import document_id, is_empty, normalize_document
from data.data_v1.phase_a import SOURCE_ID_RE, SCHEMA_VERSION
from data.data_v1.report import (
    MAX_CANARY_TOKENS,
    add_reject,
    bump,
    empty_report,
    finalize_status,
    public_report,
)
from data.data_v1.sources import ALLOWLIST_PATH, load_allowlist, source_index

ROOT = Path(__file__).resolve().parents[2]


class WhitespaceCounter:
    mode = "whitespace"

    def count(self, text: str) -> int:
        return max(1, len(text.split())) if text.strip() else 0


class TokenizerCounter:
    mode = "tokenizer"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count(self, text: str) -> tuple[int, list[int]]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoded, "ids"):
            ids = list(encoded.ids)
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            ids = list(encoded["input_ids"])
        else:
            ids = list(encoded)
        return len(ids), ids


def _uri_hash(source_id: str, path: str, snapshot: str) -> str:
    payload = f"{source_id}\n{path}\n{snapshot}".encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_id(record: dict[str, Any]) -> str:
    raw = str(record.get("source_id") or "local-canary")
    if not SOURCE_ID_RE.match(raw):
        raise PhaseBError("source_id failed pattern")
    return raw


def build_sidecar(text: str, record: dict[str, Any], *, tokens: int) -> dict[str, Any]:
    source_id = _source_id(record)
    snapshot = "phase-b-local"
    prov_in = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    snapshot = str(prov_in.get("snapshot") or record.get("snapshot") or snapshot)
    uri_hash = prov_in.get("uri_hash") or _uri_hash(source_id, str(record.get("_path", "local")), snapshot)
    license_in = record.get("license") if isinstance(record.get("license"), dict) else None
    license_doc = license_in or {"id": "fixture-local", "redistribution": True}
    doc_id = document_id(text)
    sidecar: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document_id": doc_id,
        "source_id": source_id,
        "source_type": resolve_source_type(record.get("source_type")),
        "domain": resolve_domain(text, record.get("domain")),
        "language": resolve_language(text, record.get("language")),
        "split": resolve_split(record.get("split")),
        "license": {"id": str(license_doc.get("id") or "fixture-local"), "redistribution": bool(license_doc.get("redistribution", True))},
        "provenance": {"uri_hash": uri_hash, "snapshot": snapshot},
        "quality": record.get("quality") if isinstance(record.get("quality"), dict) else {"score": 0.8, "flags": []},
        "dedup": {"exact_hash": doc_id, "near_group": None},
        "stats": {"chars": len(text), "tokens": int(tokens)},
    }
    if sidecar["source_type"] == "synthetic":
        sidecar["generator"] = record.get("generator")
    else:
        sidecar["generator"] = record.get("generator", None)
    return sidecar


def _count_tokens(text: str, counter: Any) -> tuple[int, list[int] | None]:
    if isinstance(counter, TokenizerCounter):
        n, ids = counter.count(text)
        return n, ids
    return int(counter.count(text)), None


def _input_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_local_tokenizer(path: str | Path) -> Any:
    root = ensure_local_path(path)
    json_file = root / "tokenizer.json" if root.is_dir() else root
    if not json_file.is_file():
        raise PhaseBError("tokenizer.json not found on local path")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(json_file))


def assert_allowed_sidecar(sidecar: dict[str, Any], index: dict[str, dict[str, Any]]) -> None:
    spec = index.get(sidecar["source_id"])
    if spec is None:
        raise PhaseBError("allowlist_miss")
    if sidecar["language"] not in spec["languages"]:
        raise PhaseBError("allowlist_miss")
    if sidecar["domain"] not in spec["allowed_domains"]:
        raise PhaseBError("allowlist_miss")
    if sidecar["source_type"] != spec["source_type"]:
        raise PhaseBError("allowlist_miss")


def run_canary(
    source: str | Path,
    *,
    max_tokens: int = MAX_CANARY_TOKENS,
    tokenizer: Any | None = None,
    mode: str = "fixture",
    allowlist_path: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"fixture", "canary"}:
        raise PhaseBError("mode must be fixture|canary")
    if max_tokens > MAX_CANARY_TOKENS:
        raise PhaseBError("cannot raise Phase B canary token cap")
    if max_tokens < 1:
        raise PhaseBError("max_tokens must be >= 1")
    root = ensure_local_path(source)
    allowed: dict[str, dict[str, Any]] | None = None
    if mode == "canary":
        if tokenizer is None:
            raise PhaseBError("real corpus canary requires tokenizer (add_special_tokens=False)")
        allowlist = load_allowlist(Path(allowlist_path) if allowlist_path else ALLOWLIST_PATH)
        if not allowlist["sources"]:
            raise PhaseBError("allowlist empty; acquisition closed")
        allowed = source_index(allowlist)
        max_in = float(allowlist["disk"]["max_materialization_gb"]) * 1024**3
        min_free = float(allowlist["disk"]["min_free_gb"])
        nbytes = _input_bytes(root)
        free_gb = shutil.disk_usage(root).free / 1024**3
        if nbytes > max_in:
            raise PhaseBError("canary input exceeds 8GB materialization cap")
        if free_gb < min_free:
            raise PhaseBError("disk headroom below Phase B minimum")
    report = empty_report(max_canary_tokens=max_tokens, run_mode=mode)
    if mode == "canary":
        report["input_bytes"] = _input_bytes(root)
        report["disk_free_gb"] = round(shutil.disk_usage(root).free / 1024**3, 2)
        counter: Any = TokenizerCounter(tokenizer)
    else:
        counter = TokenizerCounter(tokenizer) if tokenizer is not None else WhitespaceCounter()
    report["token_count_mode"] = counter.mode
    fingerprints = frozen_probe_fingerprints()
    dedup = CanaryDedup()
    for record in iter_records(root):
        report["documents_seen"] += 1
        try:
            raw = record.get("text", "")
            text = normalize_document(raw if isinstance(raw, str) else "")
            if is_empty(text):
                add_reject(report, "empty")
                continue
            hits = special_surface_hits(text)
            if hits:
                report["special_token_violations"] += 1
                add_reject(report, "special_token")
                continue
            assert_no_v0_generator(record)
            assert_no_probe_overlap(text, fingerprints)
            sidecar = build_sidecar(text, record, tokens=0)
            assert_sidecar_offline(sidecar)
            if allowed is not None:
                assert_allowed_sidecar(sidecar, allowed)
            n_tokens, ids = _count_tokens(text, counter)
            if ids is not None:
                bad = forbidden_token_ids(ids)
                if bad:
                    report["special_token_violations"] += 1
                    add_reject(report, "special_token_id")
                    continue
            sidecar["stats"]["tokens"] = n_tokens
            verdict, _group = dedup.check(text)
            if verdict == "exact":
                report["exact_duplicates"] += 1
                add_reject(report, "exact_duplicate")
                continue
            if verdict == "near":
                report["near_duplicates"] += 1
                add_reject(report, "near_duplicate")
                continue
            if report["tokens_kept"] + n_tokens > max_tokens:
                report["over_budget"] += 1
                add_reject(report, "over_budget")
                continue
            bump(report["domains"], sidecar["domain"])
            bump(report["languages"], sidecar["language"])
            bump(report["sources"], sidecar["source_id"])
            report["documents_kept"] += 1
            report["tokens_kept"] += n_tokens
        except PhaseBError as exc:
            reason = str(exc)
            if reason == "probe_overlap":
                report["probe_collisions"] += 1
                add_reject(report, "probe_overlap")
            elif reason == "s0_v0_generator":
                report["s0_v0_generators"] += 1
                add_reject(report, "s0_v0_generator")
            elif reason == "raw_url_sidecar":
                report["raw_url_sidecar"] += 1
                add_reject(report, "raw_url_sidecar")
            elif reason == "langid_mismatch":
                report["langid_mismatch"] += 1
                add_reject(report, "langid_mismatch")
            elif reason == "allowlist_miss":
                report["allowlist_misses"] += 1
                add_reject(report, "allowlist_miss")
            elif "domain" in reason:
                report["domain_violations"] += 1
                add_reject(report, "domain")
            elif "split" in reason:
                report["split_violations"] += 1
                add_reject(report, "split")
            elif "language" in reason:
                report["language_violations"] += 1
                add_reject(report, "language")
            else:
                report["sidecar_invalid"] += 1
                add_reject(report, "sidecar_invalid")
    report["exact_duplicates"] = max(report["exact_duplicates"], dedup.exact_duplicates)
    report["near_duplicates"] = max(report["near_duplicates"], dedup.near_duplicates)
    return public_report(finalize_status(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA DATA V1 Phase B corpus canary (local, no train, no network)")
    parser.add_argument("--input", required=True, help="local jsonl/txt/json file or directory")
    parser.add_argument("--report", type=Path, default=None, help="write JSON report here")
    parser.add_argument("--max-tokens", type=int, default=MAX_CANARY_TOKENS)
    parser.add_argument("--mode", choices=("fixture", "canary"), default="fixture")
    parser.add_argument("--tokenizer", type=Path, default=None, help="local tokenizer.json or artifacts dir")
    parser.add_argument("--allowlist", type=Path, default=None)
    args = parser.parse_args()
    tok = load_local_tokenizer(args.tokenizer) if args.tokenizer is not None else None
    body = run_canary(
        args.input,
        max_tokens=args.max_tokens,
        tokenizer=tok,
        mode=args.mode,
        allowlist_path=args.allowlist,
    )
    text = json.dumps(body, indent=2, ensure_ascii=False)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
