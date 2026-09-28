"""Load and validate DATA V1 source allowlist. No downloaders. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.data_v1 import PhaseBError
from data.data_v1.phase_a import DOMAINS, LANGUAGES, SOURCE_ID_RE, SOURCE_TYPES

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_PATH = ROOT / "data" / "specs" / "sources.allowlist.json"
SCHEMA_PATH = ROOT / "data" / "specs" / "sources.schema.json"
SOURCES_SPEC_PATH = ROOT / "data" / "specs" / "SHINRA_DATA_V1_SOURCES.md"

SCHEMA_VERSION = "shinra-data-v1-sources"
ACQUISITION_METHODS = ("local_materialized",)
FORMATS = ("jsonl", "txt", "json")
HASH_STRATEGIES = ("content_sha256", "source_id_snapshot_path")
MAX_CANARY_TOKENS = 20_000_000
MAX_MATERIALIZATION_GB = 8.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_source_record(src: dict[str, Any]) -> None:
    required = (
        "source_id",
        "snapshot",
        "license_id",
        "redistribution",
        "source_type",
        "languages",
        "allowed_domains",
        "acquisition_method",
        "local_materialization_format",
        "expected_token_range",
        "provenance_hash_strategy",
    )
    extra = set(src) - set(required)
    if extra:
        raise PhaseBError(f"source extra keys: {sorted(extra)}")
    missing = [k for k in required if k not in src]
    if missing:
        raise PhaseBError(f"source missing keys: {missing}")
    if not SOURCE_ID_RE.match(str(src["source_id"])):
        raise PhaseBError("source_id failed pattern")
    if not str(src["snapshot"]).strip():
        raise PhaseBError("snapshot required")
    if not str(src["license_id"]).strip():
        raise PhaseBError("license_id required")
    lic = str(src["license_id"]).casefold()
    if "http://" in lic or "https://" in lic:
        raise PhaseBError("license_id must not be a URL")
    if not isinstance(src["redistribution"], bool):
        raise PhaseBError("redistribution must be boolean")
    if src["redistribution"] is not True:
        raise PhaseBError("redistribution must be true to ship")
    if src["source_type"] not in SOURCE_TYPES:
        raise PhaseBError("source_type not in natural|synthetic")
    langs = src["languages"]
    if not isinstance(langs, list) or not langs or any(x not in LANGUAGES for x in langs):
        raise PhaseBError("languages must be a non-empty subset of en|ru")
    domains = src["allowed_domains"]
    if not isinstance(domains, list) or not domains or any(x not in DOMAINS for x in domains):
        raise PhaseBError("allowed_domains must use the frozen enum")
    if src["acquisition_method"] not in ACQUISITION_METHODS:
        raise PhaseBError("acquisition_method must be local_materialized")
    if src["local_materialization_format"] not in FORMATS:
        raise PhaseBError("local_materialization_format")
    rng = src["expected_token_range"]
    if not isinstance(rng, dict) or not isinstance(rng.get("min"), int) or not isinstance(rng.get("max"), int):
        raise PhaseBError("expected_token_range min/max")
    if rng["min"] < 0 or rng["max"] < 1 or rng["max"] > MAX_CANARY_TOKENS or rng["min"] > rng["max"]:
        raise PhaseBError("expected_token_range out of canary bounds")
    if src["provenance_hash_strategy"] not in HASH_STRATEGIES:
        raise PhaseBError("provenance_hash_strategy")
    blob = json.dumps(src, ensure_ascii=False)
    if "http://" in blob.casefold() or "https://" in blob.casefold():
        raise PhaseBError("raw URL is forbidden in source records")


def validate_allowlist(doc: dict[str, Any]) -> None:
    required = ("schema_version", "acquisition", "downloaders", "disk", "token_count", "sources")
    extra = set(doc) - set(required)
    if extra:
        raise PhaseBError(f"allowlist extra keys: {sorted(extra)}")
    missing = [k for k in required if k not in doc]
    if missing:
        raise PhaseBError(f"allowlist missing keys: {missing}")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise PhaseBError("allowlist schema_version")
    if doc["acquisition"] not in {"closed", "local_materialized"}:
        raise PhaseBError("acquisition")
    if doc["downloaders"] is not False:
        raise PhaseBError("downloaders must be false")
    disk = doc["disk"]
    if float(disk["max_materialization_gb"]) > MAX_MATERIALIZATION_GB:
        raise PhaseBError("max_materialization_gb exceeds 8")
    if float(disk["min_free_gb"]) < 0:
        raise PhaseBError("min_free_gb")
    if not str(disk.get("scratch", "")).strip():
        raise PhaseBError("disk.scratch required")
    tc = doc["token_count"]
    if tc.get("fixture") != "whitespace_allowed":
        raise PhaseBError("token_count.fixture")
    if tc.get("canary") != "tokenizer_required":
        raise PhaseBError("token_count.canary")
    if tc.get("add_special_tokens") is not False:
        raise PhaseBError("add_special_tokens must be false")
    if tc.get("max_canary_tokens") != MAX_CANARY_TOKENS:
        raise PhaseBError("token_count.max_canary_tokens")
    sources = doc["sources"]
    if not isinstance(sources, list):
        raise PhaseBError("sources must be a list")
    if doc["acquisition"] == "closed" and sources:
        raise PhaseBError("closed acquisition cannot list sources")
    seen: set[str] = set()
    for src in sources:
        validate_source_record(src)
        sid = src["source_id"]
        if sid in seen:
            raise PhaseBError(f"duplicate source_id {sid}")
        seen.add(sid)


def load_allowlist(path: Path | None = None) -> dict[str, Any]:
    target = path or ALLOWLIST_PATH
    doc = load_json(target)
    validate_allowlist(doc)
    return doc


def source_index(allowlist: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["source_id"]: row for row in allowlist["sources"]}


def assert_production_allowlist_closed() -> None:
    doc = load_allowlist(ALLOWLIST_PATH)
    schema = load_json(SCHEMA_PATH)
    if schema.get("properties", {}).get("downloaders", {}).get("const") is not False:
        raise PhaseBError("sources schema must forbid downloaders")
    if doc["sources"]:
        raise PhaseBError("production allowlist must stay empty until a later commit")
    if doc["acquisition"] != "closed":
        raise PhaseBError("production acquisition must be closed")
    spec = SOURCES_SPEC_PATH.read_text(encoding="utf-8").casefold()
    if "tokenizer required" not in spec:
        raise PhaseBError("SOURCES spec must require tokenizer for real canary")
    if "whitespace" not in spec:
        raise PhaseBError("SOURCES spec must allow whitespace only for fixtures")
