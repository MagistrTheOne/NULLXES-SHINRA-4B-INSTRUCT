"""Record and manifest contracts for SHINRA-V2-PRETRAIN-SYNTH-v0."""

from __future__ import annotations

from typing import Any, Mapping

CORPUS_ID = "SHINRA-V2-PRETRAIN-SYNTH-v0"
LICENSE = "nullxes-synthetic-v0"
LAYERS = (
    "language_core",
    "semantic_primitives",
    "compositional",
    "knowledge_shaped",
    "structured",
)
LANGUAGES = ("en", "ru")
RECORD_FIELDS = ("text", "layer", "language", "generator", "seed", "semantics")
MANIFEST_FIELDS = (
    "corpus_id",
    "family",
    "language",
    "seed",
    "records",
    "chars",
    "tokens",
    "sha256",
    "license",
    "generator",
)
MAX_RECORDS = 20_000


class SynthSchemaError(ValueError):
    pass


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def validate_record(record: Mapping[str, Any]) -> None:
    missing = [key for key in RECORD_FIELDS if key not in record]
    if missing:
        raise SynthSchemaError(f"record missing {missing}")
    extra = [key for key in record if key not in RECORD_FIELDS]
    if extra:
        raise SynthSchemaError(f"record extra fields {extra}")
    if record["layer"] not in LAYERS:
        raise SynthSchemaError(f"unknown layer {record['layer']}")
    if record["language"] not in LANGUAGES:
        raise SynthSchemaError(f"unknown language {record['language']}")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise SynthSchemaError("empty text")
    if not isinstance(record["generator"], str) or not record["generator"]:
        raise SynthSchemaError("generator id required")
    if not isinstance(record["seed"], int) or record["seed"] < 0:
        raise SynthSchemaError("seed must be a non-negative int")
    if not isinstance(record["semantics"], dict):
        raise SynthSchemaError("semantics must be an object")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [key for key in MANIFEST_FIELDS if key not in manifest]
    if missing:
        raise SynthSchemaError(f"manifest missing {missing}")
    extra = [key for key in manifest if key not in MANIFEST_FIELDS]
    if extra:
        raise SynthSchemaError(f"manifest extra fields {extra}")
    if manifest["corpus_id"] != CORPUS_ID:
        raise SynthSchemaError("wrong corpus_id")
    if manifest["license"] != LICENSE:
        raise SynthSchemaError("wrong license")
    if manifest["family"] not in LAYERS:
        raise SynthSchemaError(f"unknown family {manifest['family']}")
    if manifest["language"] not in LANGUAGES:
        raise SynthSchemaError(f"unknown language {manifest['language']}")
    for key in ("seed", "records", "chars", "tokens"):
        if not isinstance(manifest[key], int) or manifest[key] < (0 if key == "seed" else 1):
            raise SynthSchemaError(f"invalid {key}")
    sha = manifest["sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise SynthSchemaError("sha256 must be 64 lowercase hex chars")
