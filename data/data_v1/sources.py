"""Load and validate DATA V1 source allowlist. No downloaders. No network."""

from __future__ import annotations

import json
import re
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
LICENSE_CAVEATS = ("none", "common_crawl_third_party_rights")
SOURCE_CAVEATS = (
    "none",
    "common_crawl_tou",
    "third_party_page_rights",
    "bounded_local_slice_only",
    "full_dataset_forbidden",
)
MAX_CANARY_TOKENS = 20_000_000
GIB = 1024**3
GLOBAL_MAX_MATERIALIZED_BYTES = 8 * GIB
SOURCE_MAX_MATERIALIZED_BYTES = 4 * GIB
SOURCE_MAX_TOKENS = 8_000_000
PRODUCTION_SOURCE_IDS = ("fineweb-edu-en", "fineweb2-ru")
TEST_FIXTURE_SOURCE_IDS = frozenset({"fixture-canary-en", "ru"})
SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

GOVERNED = {
    "fineweb-edu-en": {
        "repository": "HuggingFaceFW/fineweb-edu",
        "subset": None,
        "languages": ("en",),
        "domains": frozenset({"general", "knowledge", "longform"}),
        "max_bytes": SOURCE_MAX_MATERIALIZED_BYTES,
        "max_tokens": SOURCE_MAX_TOKENS,
    },
    "fineweb2-ru": {
        "repository": "HuggingFaceFW/fineweb-2",
        "subset": "rus_Cyrl",
        "languages": ("ru",),
        "domains": frozenset({"general", "knowledge", "longform"}),
        "max_bytes": SOURCE_MAX_MATERIALIZED_BYTES,
        "max_tokens": SOURCE_MAX_TOKENS,
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _no_url(obj: Any) -> None:
    blob = json.dumps(obj, ensure_ascii=False).casefold()
    if "http://" in blob or "https://" in blob:
        raise PhaseBError("raw URL is forbidden in source records")


def validate_source_record(src: dict[str, Any]) -> None:
    required = (
        "source_id",
        "source_type",
        "languages",
        "allowed_domains",
        "acquisition_method",
        "local_materialization_format",
        "expected_token_range",
        "max_materialized_bytes",
        "license",
        "upstream",
        "materialization",
        "provenance_hash_strategy",
        "caveats",
    )
    extra = set(src) - set(required)
    if extra:
        raise PhaseBError(f"source extra keys: {sorted(extra)}")
    missing = [k for k in required if k not in src]
    if missing:
        raise PhaseBError(f"source missing keys: {missing}")
    sid = str(src["source_id"])
    if not SOURCE_ID_RE.match(sid):
        raise PhaseBError("source_id failed pattern")
    if sid not in GOVERNED and sid not in TEST_FIXTURE_SOURCE_IDS:
        raise PhaseBError("unknown source_id")
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
    nbytes = src["max_materialized_bytes"]
    if not isinstance(nbytes, int) or nbytes < 1 or nbytes > GLOBAL_MAX_MATERIALIZED_BYTES:
        raise PhaseBError("max_materialized_bytes must be in (0, 8 GiB]")
    if src["provenance_hash_strategy"] not in HASH_STRATEGIES:
        raise PhaseBError("provenance_hash_strategy")
    caveats = src["caveats"]
    if not isinstance(caveats, list) or any(c not in SOURCE_CAVEATS for c in caveats):
        raise PhaseBError("caveats")
    _validate_license(src["license"])
    _validate_upstream(src["upstream"])
    _validate_materialization(src["materialization"])
    if sid in GOVERNED:
        spec = GOVERNED[sid]
        if tuple(langs) != spec["languages"]:
            raise PhaseBError("source language mismatch")
        if frozenset(domains) != spec["domains"]:
            raise PhaseBError("source domain outside allowlist")
        if src["upstream"]["repository"] != spec["repository"]:
            raise PhaseBError("upstream repository mismatch")
        if src["upstream"]["subset"] != spec["subset"]:
            raise PhaseBError("FineWeb2 wrong subset" if sid == "fineweb2-ru" else "upstream subset mismatch")
        if src["upstream"]["revision"] is not None:
            raise PhaseBError("unverified upstream revision")
        if nbytes > spec["max_bytes"]:
            raise PhaseBError("max_materialized_bytes exceeds 4 GiB for this source")
        if rng["max"] > spec["max_tokens"]:
            raise PhaseBError("expected_token_range.max exceeds 8,000,000 for this source")
        lic = src["license"]
        if lic["dataset_id"] != "ODC-By-1.0":
            raise PhaseBError("dataset license must be ODC-By-1.0")
        if lic["underlying_content_caveat"] != "common_crawl_third_party_rights":
            raise PhaseBError("underlying_content_caveat required")
        if lic["common_crawl_tou"] is not True:
            raise PhaseBError("common_crawl_tou must be true")
        if src["source_type"] != "natural":
            raise PhaseBError("governed sources must be natural")
        if src["local_materialization_format"] != "jsonl":
            raise PhaseBError("governed sources must be jsonl")
    _no_url(src)


def _validate_license(lic: dict[str, Any]) -> None:
    required = ("dataset_id", "dataset_redistribution", "underlying_content_caveat", "common_crawl_tou")
    if set(lic) != set(required):
        raise PhaseBError("license keys")
    if not str(lic["dataset_id"]).strip():
        raise PhaseBError("license.dataset_id")
    if "http://" in str(lic["dataset_id"]).casefold() or "https://" in str(lic["dataset_id"]).casefold():
        raise PhaseBError("raw URL is forbidden in license.dataset_id")
    if not isinstance(lic["dataset_redistribution"], bool):
        raise PhaseBError("dataset_redistribution must be boolean")
    if lic["underlying_content_caveat"] not in LICENSE_CAVEATS:
        raise PhaseBError("underlying_content_caveat")
    if not isinstance(lic["common_crawl_tou"], bool):
        raise PhaseBError("common_crawl_tou")
    if lic["common_crawl_tou"] is True and lic["underlying_content_caveat"] != "common_crawl_third_party_rights":
        raise PhaseBError("Common Crawl ToU requires underlying-content caveat")


def _validate_upstream(up: dict[str, Any]) -> None:
    if set(up) != {"repository", "subset", "revision"}:
        raise PhaseBError("upstream keys")
    if not REPO_RE.match(str(up["repository"])):
        raise PhaseBError("upstream.repository must be org/name, not a URL")
    if up["subset"] is not None and not isinstance(up["subset"], str):
        raise PhaseBError("upstream.subset")
    if up["revision"] is not None and not isinstance(up["revision"], str):
        raise PhaseBError("upstream.revision")


def _validate_materialization(mat: dict[str, Any]) -> None:
    if set(mat) != {"status", "slice_id", "content_sha256"}:
        raise PhaseBError("materialization keys")
    if mat["status"] not in {"not_materialized", "materialized"}:
        raise PhaseBError("materialization.status")
    if mat["status"] == "not_materialized":
        if mat["slice_id"] is not None or mat["content_sha256"] is not None:
            raise PhaseBError("unresolved materialization must have null slice_id and content_sha256")
        return
    if not isinstance(mat["slice_id"], str) or not mat["slice_id"].strip():
        raise PhaseBError("materialized slice_id required")
    if not isinstance(mat["content_sha256"], str) or not SHA256_RE.match(mat["content_sha256"]):
        raise PhaseBError("materialized content_sha256 required")


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
    if set(disk) != {"max_materialized_bytes", "min_free_gb", "scratch"}:
        raise PhaseBError("disk keys")
    cap = disk["max_materialized_bytes"]
    if not isinstance(cap, int) or cap < 1 or cap > GLOBAL_MAX_MATERIALIZED_BYTES:
        raise PhaseBError("disk.max_materialized_bytes must be in (0, 8 GiB]")
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
    seen: set[str] = set()
    total_bytes = 0
    for src in sources:
        validate_source_record(src)
        sid = src["source_id"]
        if sid in seen:
            raise PhaseBError(f"duplicate source_id {sid}")
        seen.add(sid)
        total_bytes += int(src["max_materialized_bytes"])
    if total_bytes > cap:
        raise PhaseBError("aggregate materialization exceeds 8 GiB")
    _no_url(doc)


def assert_canary_materialized(allowlist: dict[str, Any]) -> None:
    if not allowlist.get("sources"):
        raise PhaseBError("allowlist empty; acquisition closed")
    for src in allowlist["sources"]:
        mat = src.get("materialization") or {}
        if mat.get("status") != "materialized":
            raise PhaseBError("materialization unresolved; real canary closed")
        sha = mat.get("content_sha256")
        if not isinstance(sha, str) or not SHA256_RE.match(sha):
            raise PhaseBError("materialization unresolved; real canary closed")


def load_allowlist(path: Path | None = None) -> dict[str, Any]:
    target = path or ALLOWLIST_PATH
    doc = load_json(target)
    validate_allowlist(doc)
    return doc


def source_index(allowlist: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["source_id"]: row for row in allowlist["sources"]}


def assert_production_allowlist_governance() -> None:
    doc = load_allowlist(ALLOWLIST_PATH)
    schema = load_json(SCHEMA_PATH)
    if schema.get("properties", {}).get("downloaders", {}).get("const") is not False:
        raise PhaseBError("sources schema must forbid downloaders")
    if doc["acquisition"] != "closed":
        raise PhaseBError("production acquisition must stay closed")
    ids = [row["source_id"] for row in doc["sources"]]
    if tuple(ids) != PRODUCTION_SOURCE_IDS:
        raise PhaseBError("production allowlist must be exactly fineweb-edu-en, fineweb2-ru")
    for row in doc["sources"]:
        if row["materialization"]["status"] != "not_materialized":
            raise PhaseBError("production sources must remain not_materialized")
        if row["materialization"]["content_sha256"] is not None:
            raise PhaseBError("production content_sha256 must be null")
        if row["upstream"]["revision"] is not None:
            raise PhaseBError("unverified revision must stay null")
        if row["provenance_hash_strategy"] != "content_sha256":
            raise PhaseBError("production slice identity must be content_sha256")
    if doc["disk"]["max_materialized_bytes"] != GLOBAL_MAX_MATERIALIZED_BYTES:
        raise PhaseBError("production disk cap must be 8 GiB in bytes")
    if float(doc["disk"]["min_free_gb"]) < 20:
        raise PhaseBError("production min_free_gb must be >= 20")
    spec = SOURCES_SPEC_PATH.read_text(encoding="utf-8").casefold()
    for needle in (
        "tokenizer required",
        "whitespace",
        "rus_cyrl",
        "odc-by",
        "common crawl",
        "not_materialized",
        "allowlist approval",
        "235.7",
        "dataset_redistribution",
        "underlying_content_caveat",
    ):
        if needle not in spec:
            raise PhaseBError(f"SOURCES spec missing {needle}")
