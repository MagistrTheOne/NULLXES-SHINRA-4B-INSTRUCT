"""Content-addressed tokenizer bundles; verification never imports ML code."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

FILES = ("tokenizer.model", "tokenizer.vocab", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "chat_template.jinja", "tokenizer_stats.json", "corpus_stats.json")
MANIFEST = "tokenizer_manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def contract(config: dict) -> dict:
    tok = config["tokenizer"]
    keys = ("algorithm", "vocab_size", "byte_fallback", "character_coverage", "normalization",
            "split_digits", "unk_id", "bos_id", "end_of_turn_id", "pad_id", "end_of_document_id",
            "special_tokens", "special_token_ids", "add_bos_in_tokenizer", "chat_template_adds_bos",
            "chat_turn_stop_id")
    return {"tokenizer": {key: tok[key] for key in keys},
            "unicode_normalization": config["data"]["cleaning"]["unicode_normalization"],
            "model_max_length": config["model"]["max_position_embeddings"],
            "text_policy": "nfc-preserve-whitespace-reject-controls-and-metaspace.v1",
            "chat_policy": "basic-text-single-bos-reject-overflow.v1"}


def write_manifest(root: Path, config: dict) -> str:
    manifest = {"schema": "nullxes.shinra.tokenizer-bundle.v1", "contract": contract(config),
                "files": {name: file_sha256(root / name) for name in FILES}}
    manifest["manifest_sha256"] = _digest(manifest)
    temporary = root / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(root / MANIFEST)
    return manifest["manifest_sha256"]


def verify_bundle(root: str | Path, config: dict, *, require_frozen: bool = True) -> str:
    root = Path(root)
    for name in ("added_tokens.json", "chat_templates"):
        if (root / name).exists():
            raise ValueError(f"unmanifested tokenizer override is forbidden: {name}")
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _digest(manifest) or manifest.get("schema") != "nullxes.shinra.tokenizer-bundle.v1":
        raise ValueError("tokenizer bundle manifest identity is invalid")
    if manifest.get("contract") != contract(config):
        raise ValueError("tokenizer bundle contract differs from configs/shinra.yaml")
    files = manifest.get("files", {})
    if set(files) != set(FILES):
        raise ValueError("tokenizer bundle file set is incomplete or unexpected")
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != files[name]:
            raise ValueError(f"tokenizer bundle file missing or SHA256 mismatch: {name}")
    expected = config["tokenizer"].get("artifact_sha256")
    if (require_frozen and not expected) or (expected and expected != claimed):
        raise ValueError("tokenizer bundle SHA256 is not frozen or differs from configs/shinra.yaml")
    return claimed


def copy_bundle(source: Path, destination: Path, config: dict) -> None:
    verify_bundle(source, config)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (*FILES, MANIFEST):
        if (source / name).resolve() != (destination / name).resolve():
            shutil.copy2(source / name, destination / name)
    verify_bundle(destination, config)
