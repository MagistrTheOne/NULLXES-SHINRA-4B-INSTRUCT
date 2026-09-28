"""Phase A offline contracts: sidecar schema, frozen probes, contamination fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SIDECAR_SCHEMA_PATH = ROOT / "data" / "specs" / "sidecar.schema.json"
DATA_SPEC_PATH = ROOT / "data" / "specs" / "SHINRA_DATA_V1.md"
PROBE_DIR = ROOT / "eval" / "probes" / "v1"
MANIFEST_PATH = PROBE_DIR / "manifest.json"
PROBE_SPEC_PATH = ROOT / "eval" / "specs" / "SHINRA_PROBE_V1.md"
BASELINE_DIR = ROOT / "eval" / "baselines" / "s0-step-00001358"
BASELINE_GENERATIONS = BASELINE_DIR / "generations.jsonl"
BASELINE_SCORES = BASELINE_DIR / "scores.json"
BASELINE_RUN = BASELINE_DIR / "run_manifest.json"

SCHEMA_VERSION = "shinra-data-v1"
PROBE_VERSION = "shinra-probe-v1"

DOMAINS = (
    "general",
    "longform",
    "knowledge",
    "semantic",
    "code",
    "math",
    "reasoning",
    "structured",
)
LANGUAGES = ("en", "ru")
SOURCE_TYPES = ("natural", "synthetic")
SPLITS = ("train", "validation")
SCORERS = (
    "numeric_exact",
    "choice_exact",
    "text_exact",
    "text_contains",
    "json_parse",
    "forbidden_lexicon",
    "identity_absent",
    "chat_absent",
    "generation_log",
)
PROBE_FILES = (
    "completion_en.jsonl",
    "completion_ru.jsonl",
    "syntax.jsonl",
    "negation.jsonl",
    "coreference.jsonl",
    "semantic.jsonl",
    "math.jsonl",
    "code.jsonl",
    "structured.jsonl",
    "long_context.jsonl",
    "ood.jsonl",
    "negative_identity.jsonl",
    "negative_chat.jsonl",
)
FAMILY_TO_FILE = {
    "completion_en": "completion_en.jsonl",
    "completion_ru": "completion_ru.jsonl",
    "syntax": "syntax.jsonl",
    "negation": "negation.jsonl",
    "coreference": "coreference.jsonl",
    "semantic": "semantic.jsonl",
    "math": "math.jsonl",
    "code": "code.jsonl",
    "structured": "structured.jsonl",
    "long_context": "long_context.jsonl",
    "ood": "ood.jsonl",
    "negative_identity": "negative_identity.jsonl",
    "negative_chat": "negative_chat.jsonl",
}

TOKENIZER_CONTRACT = {
    "bos_id": 1,
    "eot_id": 2,
    "document_end_id": 18,
    "pad_id": 3,
}

GENERATION_CONTRACT = {
    "do_sample": False,
    "use_cache": False,
    "max_new_tokens": 128,
}

CHECKPOINT_BASELINE = "s0/step-00001358"

S0_CLOSED_LEXICON = (
    "mira",
    "tomas",
    "alina",
    "gleb",
    "nadir",
    "irena",
    "pavel",
    "saskia",
    "мира",
    "томас",
    "алина",
    "глеб",
    "надир",
    "ирена",
    "павел",
    "саския",
    "near the door",
    "counted them twice",
    '"ok": true',
    '"ok":true',
    "kassel",
    "ostrow",
    "selva",
    "marlow",
    "кассель",
    "остров",
    "сельва",
    "марлоу",
)

S0_V0_GENERATORS = (
    "language_core.v0",
    "semantic_primitives.v0",
    "compositional.v0",
    "knowledge_shaped.v0",
    "structured.v0",
)

IDENTITY_NEEDLES = (
    "i am shinra",
    "i'm shinra",
    "я шинра",
    "я — шинра",
    "я - шинра",
    "меня зовут shinra",
    "my name is shinra",
    "created by nullxes",
    "nullxes shinra",
)

CHAT_NEEDLES = (
    "### instruction",
    "### response",
    "user:",
    "assistant:",
    "system:",
    "human:",
    "вопрос:",
    "ответ:",
    "пользователь:",
    "ассистент:",
)

SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
QUALITY_FLAGS = {
    "boilerplate",
    "too_short",
    "too_long",
    "pii_suspect",
    "repetition",
    "low_alpha",
    "langid_mismatch",
}


class PhaseAError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_document_bytes(text: str) -> bytes:
    return unicodedata.normalize("NFKC", text).replace("\x00", "").strip().encode("utf-8")


def document_id_for_text(text: str) -> str:
    return "sha256:" + sha256_bytes(canonical_document_bytes(text))


def normalize_fingerprint(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", folded).strip()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PhaseAError(f"{path}:{line_no}: invalid json") from exc
        if not isinstance(row, dict):
            raise PhaseAError(f"{path}:{line_no}: record must be an object")
        rows.append(row)
    return rows


def _reject_raw_urls(obj: Any, path: str = "$") -> None:
    if isinstance(obj, str):
        folded = obj.casefold()
        if "http://" in folded or "https://" in folded:
            raise PhaseAError(f"raw URL is forbidden in sidecar at {path}")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _reject_raw_urls(value, f"{path}.{key}")
        return
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            _reject_raw_urls(value, f"{path}[{i}]")


def validate_sidecar(doc: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "document_id",
        "source_id",
        "source_type",
        "domain",
        "language",
        "split",
        "license",
        "provenance",
        "quality",
        "dedup",
        "stats",
    )
    extra = set(doc) - set(required) - {"generator"}
    if extra:
        raise PhaseAError(f"sidecar extra keys: {sorted(extra)}")
    missing = [k for k in required if k not in doc]
    if missing:
        raise PhaseAError(f"sidecar missing keys: {missing}")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise PhaseAError("schema_version must be shinra-data-v1")
    if not SHA256_RE.match(str(doc["document_id"])):
        raise PhaseAError("document_id must be sha256:<64 hex>")
    if not SOURCE_ID_RE.match(str(doc["source_id"])):
        raise PhaseAError("source_id failed pattern")
    if doc["source_type"] not in SOURCE_TYPES:
        raise PhaseAError(f"source_type not in {SOURCE_TYPES}")
    if doc["domain"] not in DOMAINS:
        raise PhaseAError(f"domain not in {DOMAINS}")
    if doc["language"] not in LANGUAGES:
        raise PhaseAError(f"language not in {LANGUAGES}")
    if doc["split"] not in SPLITS:
        raise PhaseAError("split must be train|validation (probes are not a corpus split)")
    if "probe" in str(doc["split"]):
        raise PhaseAError("probe is not a sidecar split")
    gen = doc.get("generator", None)
    if doc["source_type"] == "synthetic":
        if not isinstance(gen, str) or not gen:
            raise PhaseAError("synthetic sidecar requires generator")
        if gen in S0_V0_GENERATORS or gen.endswith(".v0"):
            raise PhaseAError("S0 *.v0 generators are forbidden in DATA V1")
    elif gen not in (None,):
        raise PhaseAError("natural sidecar generator must be null/absent")
    lic = doc["license"]
    if not isinstance(lic, dict) or not lic.get("id") or not isinstance(lic.get("redistribution"), bool):
        raise PhaseAError("license.id and license.redistribution required")
    if set(lic) - {"id", "redistribution"}:
        raise PhaseAError("license extra keys")
    prov = doc["provenance"]
    if "uri" in prov or "url" in prov:
        raise PhaseAError("raw URL is forbidden in sidecar")
    if set(prov) - {"uri_hash", "snapshot"}:
        raise PhaseAError("provenance extra keys")
    if not SHA256_RE.match(str(prov.get("uri_hash", ""))):
        raise PhaseAError("provenance.uri_hash must be sha256:<64 hex>")
    if not str(prov.get("snapshot", "")).strip():
        raise PhaseAError("provenance.snapshot required")
    quality = doc["quality"]
    if not isinstance(quality.get("score"), (int, float)) or not 0.0 <= float(quality["score"]) <= 1.0:
        raise PhaseAError("quality.score must be in [0, 1]")
    flags = quality.get("flags")
    if not isinstance(flags, list) or any(f not in QUALITY_FLAGS for f in flags):
        raise PhaseAError("quality.flags must use the closed flag enum")
    dedup = doc["dedup"]
    if not SHA256_RE.match(str(dedup.get("exact_hash", ""))):
        raise PhaseAError("dedup.exact_hash must be sha256:<64 hex>")
    if dedup.get("near_group") is not None and not isinstance(dedup.get("near_group"), str):
        raise PhaseAError("dedup.near_group must be string or null")
    stats = doc["stats"]
    if not isinstance(stats.get("chars"), int) or stats["chars"] < 0:
        raise PhaseAError("stats.chars")
    if not isinstance(stats.get("tokens"), int) or stats["tokens"] < 0:
        raise PhaseAError("stats.tokens")
    _reject_raw_urls(doc)


def validate_probe_record(row: dict[str, Any], filename: str) -> None:
    required = ("id", "family", "language", "prompt", "scorer", "max_new_tokens", "tags")
    missing = [k for k in required if k not in row]
    if missing:
        raise PhaseAError(f"{filename} {row.get('id')}: missing {missing}")
    if row["family"] not in FAMILY_TO_FILE:
        raise PhaseAError(f"unknown family {row['family']}")
    if FAMILY_TO_FILE[row["family"]] != filename:
        raise PhaseAError(f"{row['id']}: family {row['family']} does not belong in {filename}")
    if row["language"] not in LANGUAGES:
        raise PhaseAError(f"{row['id']}: language")
    if row["scorer"] not in SCORERS:
        raise PhaseAError(f"{row['id']}: scorer {row['scorer']}")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise PhaseAError(f"{row['id']}: empty prompt")
    if not isinstance(row["max_new_tokens"], int) or row["max_new_tokens"] < 1:
        raise PhaseAError(f"{row['id']}: max_new_tokens")
    if row["max_new_tokens"] > GENERATION_CONTRACT["max_new_tokens"]:
        raise PhaseAError(f"{row['id']}: max_new_tokens exceeds frozen generation cap")
    if not isinstance(row.get("tags"), list) or not all(isinstance(t, str) for t in row["tags"]):
        raise PhaseAError(f"{row['id']}: tags")
    if row["scorer"] in {"choice_exact", "numeric_exact", "text_exact", "text_contains"} and row.get("expected") in (None, ""):
        raise PhaseAError(f"{row['id']}: expected required for {row['scorer']}")
    if row["scorer"] == "numeric_exact" and not isinstance(row["expected"], (int, float)):
        raise PhaseAError(f"{row['id']}: numeric_exact expected must be a number")
    if row["scorer"] == "choice_exact" and str(row["expected"]) not in {"A", "B", "C"}:
        raise PhaseAError(f"{row['id']}: choice_exact expected must be A|B|C")


def load_all_probes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in PROBE_FILES:
        path = PROBE_DIR / name
        if not path.is_file():
            raise PhaseAError(f"missing probe file {name}")
        for row in load_jsonl(path):
            validate_probe_record(row, name)
            if row["id"] in seen:
                raise PhaseAError(f"duplicate probe id {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    return rows


def fingerprint_set(rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        prompt = row["prompt"]
        out.add(normalize_fingerprint(prompt))
        expected = row.get("expected")
        if expected is not None:
            out.add(normalize_fingerprint(str(expected)))
            out.add(normalize_fingerprint(f"{prompt} {expected}"))
            out.add(normalize_fingerprint(prompt + str(expected)))
        if len(prompt) >= 40:
            for i in range(0, len(prompt) - 39):
                span = normalize_fingerprint(prompt[i : i + 40])
                if span:
                    out.add(span)
    return {item for item in out if item}


def assert_text_not_contaminated(text: str, fingerprints: set[str]) -> None:
    norm = normalize_fingerprint(text)
    if not norm:
        return
    if norm in fingerprints:
        raise PhaseAError("document matches a frozen probe fingerprint")
    for fp in fingerprints:
        if len(fp) >= 40 and fp in norm:
            raise PhaseAError("document contains a long matching probe span")


def file_hashes() -> dict[str, str]:
    return {name: "sha256:" + sha256_file(PROBE_DIR / name) for name in PROBE_FILES}


def bundle_hash(file_map: dict[str, str]) -> str:
    payload = json.dumps(
        {
            "probe_version": PROBE_VERSION,
            "tokenizer_contract": TOKENIZER_CONTRACT,
            "generation": GENERATION_CONTRACT,
            "checkpoint_baseline": CHECKPOINT_BASELINE,
            "files": {k: file_map[k] for k in PROBE_FILES},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(payload.encode("utf-8"))


def manifest_body(file_map: dict[str, str]) -> dict[str, Any]:
    bundle = bundle_hash(file_map)
    body = {
        "probe_version": PROBE_VERSION,
        "tokenizer_contract": TOKENIZER_CONTRACT,
        "checkpoint_baseline": CHECKPOINT_BASELINE,
        "generation": GENERATION_CONTRACT,
        "files": {k: file_map[k] for k in PROBE_FILES},
        "probe_bundle_sha256": bundle,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["manifest_sha256"] = sha256_bytes(canonical.encode("utf-8"))
    return body


def write_manifest() -> dict[str, Any]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    load_all_probes()
    body = manifest_body(file_hashes())
    MANIFEST_PATH.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return body


def validate_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise PhaseAError("missing eval/probes/v1/manifest.json")
    stored = load_json(MANIFEST_PATH)
    expected = manifest_body(file_hashes())
    if stored.get("probe_version") != PROBE_VERSION:
        raise PhaseAError("probe_version mismatch")
    if stored.get("tokenizer_contract") != TOKENIZER_CONTRACT:
        raise PhaseAError("tokenizer_contract is frozen; do not edit")
    if stored.get("generation") != GENERATION_CONTRACT:
        raise PhaseAError("generation contract is frozen; do not edit")
    if stored.get("generation", {}).get("use_cache") is not False:
        raise PhaseAError("use_cache=false is part of the frozen eval contract")
    if stored.get("generation", {}).get("do_sample") is not False:
        raise PhaseAError("do_sample=false is part of the frozen eval contract")
    if stored.get("files") != expected["files"]:
        raise PhaseAError("probe JSONL hashes do not match manifest; start probe-v1.1, do not patch v1")
    if stored.get("probe_bundle_sha256") != expected["probe_bundle_sha256"]:
        raise PhaseAError("probe_bundle_sha256 mismatch")
    if stored.get("manifest_sha256") != expected["manifest_sha256"]:
        raise PhaseAError("manifest_sha256 mismatch")
    return stored


def validate_sidecar_schema_file() -> None:
    schema = load_json(SIDECAR_SCHEMA_PATH)
    if schema.get("additionalProperties") is not False:
        raise PhaseAError("sidecar schema must set additionalProperties false")
    if schema.get("properties", {}).get("split", {}).get("enum") != list(SPLITS):
        raise PhaseAError("sidecar schema split enum must be train|validation")
    if "probe" in schema.get("properties", {}).get("split", {}).get("enum", []):
        raise PhaseAError("probe must not be a sidecar split")
    domain = schema.get("properties", {}).get("domain", {}).get("enum")
    if domain != list(DOMAINS):
        raise PhaseAError("sidecar domain enum mismatch")
    lang = schema.get("properties", {}).get("language", {}).get("enum")
    if lang != list(LANGUAGES):
        raise PhaseAError("sidecar language enum mismatch")
    source = schema.get("properties", {}).get("source_type", {}).get("enum")
    if source != list(SOURCE_TYPES):
        raise PhaseAError("sidecar source_type enum mismatch")


def load_baseline_generations() -> list[dict[str, Any]]:
    if not BASELINE_GENERATIONS.is_file():
        raise PhaseAError("missing eval/baselines/s0-step-00001358/generations.jsonl")
    rows = load_jsonl(BASELINE_GENERATIONS)
    if not rows:
        raise PhaseAError("baseline generations.jsonl is empty")
    seen: set[str] = set()
    for row in rows:
        if "id" not in row or "generation" not in row:
            raise PhaseAError("baseline generation row needs id and generation")
        if row["id"] in seen:
            raise PhaseAError(f"duplicate baseline generation id {row['id']}")
        seen.add(row["id"])
    return rows


def validate_baseline(probe_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    probes = {row["id"]: row for row in (probe_rows if probe_rows is not None else load_all_probes())}
    gens = load_baseline_generations()
    if not BASELINE_RUN.is_file():
        raise PhaseAError("missing eval/baselines/s0-step-00001358/run_manifest.json")
    if not BASELINE_SCORES.is_file():
        raise PhaseAError("missing eval/baselines/s0-step-00001358/scores.json")
    run = load_json(BASELINE_RUN)
    if run.get("checkpoint_baseline") != CHECKPOINT_BASELINE:
        raise PhaseAError("baseline run_manifest checkpoint_baseline mismatch")
    if run.get("generation") != GENERATION_CONTRACT:
        raise PhaseAError("baseline run_manifest generation contract is frozen")
    if run.get("tokenizer_contract") != TOKENIZER_CONTRACT:
        raise PhaseAError("baseline run_manifest tokenizer_contract is frozen")
    if run.get("probe_version") != PROBE_VERSION:
        raise PhaseAError("baseline run_manifest probe_version mismatch")
    if run.get("coverage") not in {"partial", "full"}:
        raise PhaseAError("baseline coverage must be partial|full")
    computed: list[dict[str, Any]] = []
    for gen_row in gens:
        probe = probes.get(gen_row["id"])
        if probe is None:
            raise PhaseAError(f"baseline generation {gen_row['id']} is not in PROBE V1")
        computed.append(score_generation(probe, str(gen_row["generation"])))
    stored = load_json(BASELINE_SCORES)
    if stored.get("items") != computed:
        raise PhaseAError("baseline scores.json does not match frozen scorers; do not hand-edit")
    if stored.get("coverage") != run.get("coverage"):
        raise PhaseAError("baseline scores coverage mismatch")
    if run.get("coverage") == "full" and len(computed) != len(probes):
        raise PhaseAError("full baseline must score every PROBE V1 id")
    return {
        "n_scored": len(computed),
        "n_probes": len(probes),
        "coverage": run["coverage"],
    }


def validate_phase_a() -> dict[str, Any]:
    if not DATA_SPEC_PATH.is_file():
        raise PhaseAError("missing data/specs/SHINRA_DATA_V1.md")
    if not PROBE_SPEC_PATH.is_file():
        raise PhaseAError("missing eval/specs/SHINRA_PROBE_V1.md")
    spec = PROBE_SPEC_PATH.read_text(encoding="utf-8")
    if "use_cache" not in spec or "false" not in spec.casefold():
        raise PhaseAError("PROBE V1 spec must freeze use_cache=false")
    validate_sidecar_schema_file()
    rows = load_all_probes()
    fingerprints = fingerprint_set(rows)
    if len(fingerprints) < 20:
        raise PhaseAError("fingerprint set too small")
    stored = validate_manifest()
    baseline = validate_baseline(rows)
    return {
        "ok": True,
        "n_probes": len(rows),
        "n_fingerprints": len(fingerprints),
        "manifest_sha256": stored["manifest_sha256"],
        "probe_bundle_sha256": stored["probe_bundle_sha256"],
        "baseline_coverage": baseline["coverage"],
        "baseline_n_scored": baseline["n_scored"],
    }


def _first_choice(text: str) -> str | None:
    match = re.search(r"\b([ABC])\b", text.strip())
    return match.group(1) if match else None


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


def score_generation(row: dict[str, Any], generation: str) -> dict[str, Any]:
    scorer = row["scorer"]
    gen = generation or ""
    folded = normalize_fingerprint(gen)
    passed: bool | None
    detail: str
    if scorer == "generation_log":
        passed = None
        detail = "logged"
    elif scorer == "numeric_exact":
        got = _first_number(gen)
        passed = got is not None and got == float(row["expected"])
        detail = f"got={got}"
    elif scorer == "choice_exact":
        got = _first_choice(gen)
        passed = got == str(row["expected"])
        detail = f"got={got}"
    elif scorer == "text_exact":
        passed = gen.strip() == str(row["expected"]).strip()
        detail = "exact"
    elif scorer == "text_contains":
        passed = str(row["expected"]).casefold() in gen.casefold()
        detail = "contains"
    elif scorer == "json_parse":
        blob = row["prompt"] + gen
        passed = False
        detail = "no-json"
        for opener, closer in (("{", "}"), ("[", "]")):
            start = blob.find(opener)
            end = blob.rfind(closer)
            if start >= 0 and end > start:
                try:
                    json.loads(blob[start : end + 1])
                    passed = True
                    detail = "parsed"
                    break
                except json.JSONDecodeError:
                    detail = "invalid-json"
    elif scorer == "forbidden_lexicon":
        hits = [item for item in S0_CLOSED_LEXICON if item in folded]
        passed = not hits
        detail = ",".join(hits) if hits else "clean"
    elif scorer == "identity_absent":
        hits = [item for item in IDENTITY_NEEDLES if item in folded]
        passed = not hits
        detail = ",".join(hits) if hits else "clean"
    elif scorer == "chat_absent":
        hits = [item for item in CHAT_NEEDLES if item in folded]
        passed = not hits
        detail = ",".join(hits) if hits else "clean"
    else:
        raise PhaseAError(f"unknown scorer {scorer}")
    return {"id": row["id"], "scorer": scorer, "pass": passed, "detail": detail}


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(description="SHINRA DATA V1 Phase A validator")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        body = write_manifest()
        print(json.dumps({"wrote": str(MANIFEST_PATH), **{k: body[k] for k in ("manifest_sha256", "probe_bundle_sha256")}}, indent=2))
        return
    print(json.dumps(validate_phase_a(), indent=2))


if __name__ == "__main__":
    main()
