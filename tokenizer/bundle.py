"""Content-addressed manifests for the legacy Fast and Native bundles."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

FAST_SCHEMA = "nullxes.shinra.tokenizer-bundle.v1"
NATIVE_SCHEMA = "nullxes.shinra.native-tokenizer-bundle.v1"
FAST_FILES = ("tokenizer.model", "tokenizer.vocab", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "chat_template.jinja", "tokenizer_stats.json", "corpus_stats.json")
NATIVE_FILES = ("tokenizer.model", "tokenizer.vocab", "tokenizer_config.json",
                "special_tokens_map.json", "chat_template.jinja", "tokenizer_stats.json", "corpus_stats.json")
FILES = FAST_FILES
MANIFEST = "tokenizer_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _special_id_map(config: dict) -> dict[str, int]:
    tok = config["tokenizer"]
    result = {tok["special_tokens"][key]: int(tok[f"{key}_id"])
              for key in ("unk", "bos", "end_of_turn", "pad", "end_of_document")}
    result.update({tok["special_tokens"][key]: int(value) for key, value in tok["special_token_ids"].items()})
    return result


def contract(config: dict, backend: str | None = None) -> dict:
    tok = config["tokenizer"]
    backend = backend or tok.get("backend", "sentencepiece_fast")
    keys = ("algorithm", "vocab_size", "byte_fallback", "character_coverage", "normalization",
            "split_digits", "unk_id", "bos_id", "end_of_turn_id", "pad_id", "end_of_document_id",
            "special_tokens", "special_token_ids", "add_bos_in_tokenizer", "chat_template_adds_bos",
            "chat_turn_stop_id")
    result = {"tokenizer": {key: tok[key] for key in keys},
              "unicode_normalization": config["data"]["cleaning"]["unicode_normalization"],
              "model_max_length": config["model"]["max_position_embeddings"],
              "text_policy": "nfc-preserve-whitespace-reject-controls-and-metaspace.v1",
              "chat_policy": "basic-text-single-bos-reject-overflow.v1"}
    if backend == "sentencepiece_native":
        result["backend"] = backend
        result["native_model_sha256"] = tok["native_model_sha256"]
    return result


def write_native_metadata(root: Path, config: dict, *, provenance: dict[str, Any]) -> None:
    """Write deterministic Native sidecars without generating a Fast tokenizer."""
    from .special_tokens import render_chat_template

    tok = config["tokenizer"]
    payload = {"tokenizer_class": "ShinraNativeTokenizer", "shinra_tokenizer_backend": "sentencepiece_native",
               "algorithm": tok["algorithm"], "byte_fallback": bool(tok["byte_fallback"]),
               "split_digits": bool(tok["split_digits"]), "native_model_sha256": tok["native_model_sha256"],
               "vocab_size": int(tok["vocab_size"]), "model_max_length": int(config["model"]["max_position_embeddings"]),
               "add_bos_token": False, "add_eos_token": False, "normalization": tok["normalization"],
               "unicode_normalization": config["data"]["cleaning"]["unicode_normalization"],
               "special_token_ids": {key: int(value) for key, value in {
                   **{key: tok[f"{key}_id"] for key in ("unk", "bos", "end_of_turn", "pad", "end_of_document")},
                   **tok["special_token_ids"],
               }.items()}, "special_tokens": tok["special_tokens"], "provenance": provenance}
    (root / "tokenizer_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    ids = _special_id_map(config)
    special_map = {"unk_token": tok["special_tokens"]["unk"], "bos_token": tok["special_tokens"]["bos"],
                   "eos_token": tok["special_tokens"]["end_of_turn"], "pad_token": tok["special_tokens"]["pad"],
                   "additional_special_tokens": [token for token, index in sorted(ids.items(), key=lambda pair: pair[1])
                                                  if index >= 4]}
    (root / "special_tokens_map.json").write_text(
        json.dumps(special_map, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (root / "chat_template.jinja").write_text(render_chat_template(config), encoding="utf-8")


def _schema_and_files(backend: str) -> tuple[str, tuple[str, ...]]:
    if backend == "sentencepiece_native":
        return NATIVE_SCHEMA, NATIVE_FILES
    if backend == "sentencepiece_fast":
        return FAST_SCHEMA, FAST_FILES
    raise ValueError(f"unknown tokenizer backend: {backend}")


def write_manifest(root: Path, config: dict, *, backend: str | None = None,
                   provenance: dict[str, Any] | None = None) -> str:
    backend = backend or config["tokenizer"].get("backend", "sentencepiece_fast")
    schema, required_files = _schema_and_files(backend)
    manifest = {"schema": schema, "contract": contract(config, backend),
                "files": {name: file_sha256(root / name) for name in required_files}}
    if backend == "sentencepiece_native":
        manifest["backend"] = backend
    if backend == "sentencepiece_native" and not provenance:
        raise ValueError("Native bundle manifests require verified source provenance")
    if provenance is not None:
        manifest["provenance"] = provenance
    manifest["manifest_sha256"] = _digest(manifest)
    temporary = root / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(root / MANIFEST)
    return manifest["manifest_sha256"]


def verify_bundle(root: str | Path, config: dict, *, require_frozen: bool = True,
                  backend: str | None = None) -> str:
    root = Path(root)
    backend = backend or config["tokenizer"].get("backend", "sentencepiece_fast")
    schema, required_files = _schema_and_files(backend)
    for name in ("added_tokens.json", "chat_templates"):
        if (root / name).exists():
            raise ValueError(f"unmanifested tokenizer override is forbidden: {name}")
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    backend_matches = manifest.get("backend") == backend if backend == "sentencepiece_native" else (
        manifest.get("backend") in (None, "sentencepiece_fast")
    )
    if claimed != _digest(manifest) or manifest.get("schema") != schema or not backend_matches:
        raise ValueError("tokenizer bundle manifest identity is invalid")
    if manifest.get("contract") != contract(config, backend):
        raise ValueError("tokenizer bundle contract differs from configs/shinra.yaml")
    files = manifest.get("files", {})
    if set(files) != set(required_files):
        raise ValueError("tokenizer bundle file set is incomplete or unexpected")
    for name in required_files:
        path = root / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != files[name]:
            raise ValueError(f"tokenizer bundle file missing or SHA256 mismatch: {name}")

    if backend == "sentencepiece_native":
        if file_sha256(root / "tokenizer.model") != config["tokenizer"].get("native_model_sha256"):
            raise ValueError("native SentencePiece model SHA256 differs from the verified model identity")
        metadata = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
        tok = config["tokenizer"]
        provenance = metadata.get("provenance")
        manifest_provenance = manifest.get("provenance")
        if not isinstance(provenance, dict) or not isinstance(manifest_provenance, dict):
            raise ValueError("Native bundle provenance must be an object in metadata and manifest")
        if (metadata.get("shinra_tokenizer_backend") != backend
                or metadata.get("tokenizer_class") != "ShinraNativeTokenizer"
                or metadata.get("algorithm") != tok["algorithm"]
                or metadata.get("byte_fallback") is not tok["byte_fallback"]
                or metadata.get("split_digits") is not tok["split_digits"]
                or metadata.get("native_model_sha256") != tok["native_model_sha256"]
                or metadata.get("vocab_size") != tok["vocab_size"]
                or metadata.get("model_max_length") != config["model"]["max_position_embeddings"]
                or metadata.get("normalization") != tok["normalization"]
                or metadata.get("add_bos_token") is not False
                or metadata.get("add_eos_token") is not False
                or metadata.get("unicode_normalization") != config["data"]["cleaning"]["unicode_normalization"]
                or manifest_provenance != provenance
                or provenance.get("source_model_sha256") != tok["native_model_sha256"]
                or metadata.get("special_token_ids") != {key: int(value) for key, value in {
                    **{key: tok[f"{key}_id"] for key in
                       ("unk", "bos", "end_of_turn", "pad", "end_of_document")},
                    **tok["special_token_ids"],
                }.items()}
                or metadata.get("special_tokens") != tok["special_tokens"]):
            raise ValueError("Native tokenizer sidecar metadata differs from YAML")
        from .special_tokens import render_chat_template
        if (root / "chat_template.jinja").read_text(encoding="utf-8") != render_chat_template(config):
            raise ValueError("Native bundle chat template differs from YAML")
        expected_map = {"unk_token": tok["special_tokens"]["unk"],
                        "bos_token": tok["special_tokens"]["bos"],
                        "eos_token": tok["special_tokens"]["end_of_turn"],
                        "pad_token": tok["special_tokens"]["pad"],
                        "additional_special_tokens": [token for token, index in
                            sorted(_special_id_map(config).items(), key=lambda pair: pair[1]) if index >= 4]}
        if json.loads((root / "special_tokens_map.json").read_text(encoding="utf-8")) != expected_map:
            raise ValueError("Native bundle special token map differs from YAML")

    expected = config["tokenizer"].get("artifact_sha256")
    if (require_frozen and not expected) or (expected and expected != claimed):
        raise ValueError("tokenizer bundle SHA256 is not frozen or differs from configs/shinra.yaml")
    return claimed


def copy_bundle(source: Path, destination: Path, config: dict) -> None:
    verify_bundle(source, config)
    backend = config["tokenizer"].get("backend", "sentencepiece_fast")
    _, required_files = _schema_and_files(backend)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*required_files, MANIFEST):
        if (source / name).resolve() != (destination / name).resolve():
            shutil.copy2(source / name, destination / name)
    verify_bundle(destination, config)
