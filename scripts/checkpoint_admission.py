"""Single offline admission contract for complete SHINRA training checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def checkpoint_admission(checkpoint: str | Path) -> str:
    root = Path(checkpoint)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"checkpoint must be a local, non-symlink directory: {root}")

    manifest_path = root / "checkpoint_manifest.json"
    manifest = _read_json(manifest_path, "checkpoint integrity manifest")
    claimed_manifest_sha = manifest.pop("manifest_sha256", None)
    expected_files = manifest.get("files")
    if (manifest.get("schema") != "nullxes.shinra.training-checkpoint.v1"
            or not isinstance(expected_files, dict) or not expected_files
            or claimed_manifest_sha != _canonical_sha256(manifest)):
        raise ValueError(f"checkpoint integrity manifest is invalid: {root}")

    actual_files: dict[str, str] = {}
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"checkpoint symlinks are forbidden: {item}")
        if item.is_file() and item != manifest_path:
            actual_files[item.relative_to(root).as_posix()] = _file_sha256(item)
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        mismatched = sorted(name for name in set(actual_files) & set(expected_files)
                            if actual_files[name] != expected_files[name])
        raise ValueError(
            "checkpoint files differ from integrity manifest: "
            f"missing={missing}, extra={extra}, sha_mismatch={mismatched}"
        )

    required = {
        "config.json", "training_admission.json", "trainer_state.pt", "train_state.json",
        "tokenizer.model", "tokenizer.vocab", "tokenizer_config.json", "special_tokens_map.json",
        "chat_template.jinja", "tokenizer_stats.json", "corpus_stats.json", "tokenizer_manifest.json",
    }
    missing_required = sorted(required - set(actual_files))
    weight_files = [name for name in actual_files if
                    (name.startswith("model") and name.endswith(".safetensors"))
                    or (name.startswith("pytorch_model") and name.endswith(".bin"))]
    if missing_required or not weight_files:
        raise ValueError(
            f"checkpoint is incomplete: missing_required={missing_required}, model_weight_files={weight_files}"
        )

    tokenizer_manifest = _read_json(root / "tokenizer_manifest.json", "Native tokenizer bundle manifest")
    tokenizer_claim = tokenizer_manifest.pop("manifest_sha256", None)
    tokenizer_files = tokenizer_manifest.get("files")
    if (tokenizer_manifest.get("schema") != "nullxes.shinra.native-tokenizer-bundle.v1"
            or not isinstance(tokenizer_files, dict)
            or tokenizer_claim != hashlib.sha256(
                json.dumps(tokenizer_manifest, sort_keys=True, ensure_ascii=True).encode("utf-8")
            ).hexdigest()
            or set(tokenizer_files) != {
                "tokenizer.model", "tokenizer.vocab", "tokenizer_config.json", "special_tokens_map.json",
                "chat_template.jinja", "tokenizer_stats.json", "corpus_stats.json",
            }
            or any(actual_files.get(name) != digest for name, digest in tokenizer_files.items())):
        raise ValueError(f"Native tokenizer bundle manifest is invalid or incomplete: {root}")

    record = _read_json(root / "training_admission.json", "checkpoint admission metadata")
    model_config = _read_json(root / "config.json", "model config")
    train_state = _read_json(root / "train_state.json", "trainer state")
    admission = record.get("admission")
    if (record.get("schema") != "nullxes.shinra.training-admission.v1"
            or admission not in {"production", "diagnostic_audit"}
            or model_config.get("shinra_training_admission") != admission
            or train_state.get("admission") != admission
            or train_state.get("step") != manifest.get("step")
            or record.get("production_usable") is not (admission == "production")):
        raise ValueError(f"checkpoint admission metadata is inconsistent: {root}")
    return admission


def require_production_checkpoint(checkpoint: str | Path) -> None:
    if checkpoint_admission(checkpoint) != "production":
        raise ValueError("UNAVAILABLE: audit-only diagnostic checkpoints are not admitted for serving")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a local SHINRA checkpoint before serving")
    parser.add_argument("checkpoint", help="local checkpoint directory")
    args = parser.parse_args(argv)
    try:
        require_production_checkpoint(args.checkpoint)
    except ValueError as exc:
        raise SystemExit(f"UNAVAILABLE: {exc}") from exc


if __name__ == "__main__":
    main()
