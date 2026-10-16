"""Resumable, bounded Colab build from pinned sources through packed training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from configs.loader import CONFIG_PATH, load_config
from data.hashing import sha256_file
from data.numeric_integrity import _load_raw_manifest, prepare_local_corpus
from data.pack import pack_documents
from data.sources import acquire_source
from tokenizer import load_shinra_tokenizer


class CorpusBuildError(RuntimeError):
    """The candidate corpus could not be built or did not meet its token target."""


def _manifest_sha(body: dict[str, Any]) -> str:
    value = dict(body)
    value.pop("manifest_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _verify_clean_corpus(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"clean corpus manifest is missing or invalid: {exc}") from exc
    if manifest.get("schema") != "nullxes.shinra.corpus.v2" \
            or manifest.get("manifest_sha256") != _manifest_sha(manifest):
        raise CorpusBuildError("clean corpus manifest schema or self-hash is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise CorpusBuildError("clean corpus manifest has no checksummed files")
    for name, digest in files.items():
        file_path = path / name
        if not file_path.is_file() or sha256_file(file_path) != digest:
            raise CorpusBuildError(f"clean corpus file checksum mismatch: {name}")
    return manifest


def _verify_packed_split(path: Path, split: str, source_manifest_sha: str) -> dict[str, Any]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusBuildError(f"packed {split} manifest is missing or invalid: {exc}") from exc
    if manifest.get("schema") != "nullxes.shinra.packed.v3" \
            or manifest.get("manifest_sha256") != _manifest_sha(manifest) \
            or manifest.get("split") != split \
            or manifest.get("source_manifest_sha256") != source_manifest_sha:
        raise CorpusBuildError(f"packed {split} manifest identity or self-hash is invalid")
    files, hashes = manifest.get("shard_files"), manifest.get("shard_sha256")
    if not isinstance(files, list) or not files or not isinstance(hashes, dict) or set(files) != set(hashes):
        raise CorpusBuildError(f"packed {split} shard inventory is incomplete")
    for name in files:
        shard = path / name
        if not shard.is_file() or sha256_file(shard) != hashes[name]:
            raise CorpusBuildError(f"packed {split} shard checksum mismatch: {name}")
    return manifest


def build_corpus(tokenizer_path: str | Path, output_dir: str | Path, *,
                 config_path: str | Path = CONFIG_PATH, resume: bool = False,
                 minimum_free_gb: float | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    tokenizer = load_shinra_tokenizer(tokenizer_path, config, require_frozen_bundle=False)
    tokenizer_bundle_sha = tokenizer.shinra_bundle_sha256
    target = config["data"]["target_build"]
    source_caps = target["source_candidate_targets"]
    output = Path(output_dir).resolve()
    stage = output.parent / f".{output.name}.partial"
    if output.exists():
        raise CorpusBuildError(f"refusing to overwrite completed corpus artifact: {output}")
    if stage.exists() and not resume:
        raise CorpusBuildError(f"partial build exists; pass --resume to verify and continue: {stage}")
    if resume and not stage.is_dir():
        raise CorpusBuildError(f"no partial build exists to resume: {stage}")
    free_gb_required = float(config["runtime"]["minimum_free_disk_gb"]
                             if minimum_free_gb is None else minimum_free_gb)
    if free_gb_required <= 0:
        raise CorpusBuildError("minimum_free_gb must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    available_gb = shutil.disk_usage(output.parent).free / (1024 ** 3)
    if available_gb < free_gb_required:
        raise CorpusBuildError(
            f"insufficient free space before build: available={available_gb:.2f} GiB, "
            f"required={free_gb_required:.2f} GiB"
        )
    if not stage.exists():
        stage.mkdir()

    def require_disk_headroom() -> None:
        free_gb = shutil.disk_usage(output.parent).free / (1024 ** 3)
        if free_gb < free_gb_required:
            raise CorpusBuildError(
                f"pausing resumable build before the next phase: free={free_gb:.2f} GiB, "
                f"configured minimum={free_gb_required:.2f} GiB"
            )

    source_dirs: list[Path] = []
    for source_id, token_cap in source_caps.items():
        safe_name = source_id.replace("/", "--")
        source_dir = stage / "raw" / safe_name
        source_dirs.append(source_dir)
        if source_dir.exists():
            source_manifest = _load_raw_manifest(source_dir / "manifest.json")
            if source_manifest.get("requested_max_target_tokens") != int(token_cap) \
                    or source_manifest.get("target_limit_reached") is not True \
                    or source_manifest.get("tokenizer_bundle_sha256") != tokenizer_bundle_sha:
                raise CorpusBuildError(
                    f"existing source snapshot {source_id} did not reach its pinned Native-token cap "
                    "or uses another tokenizer bundle"
                )
            continue
        require_disk_headroom()
        source_manifest = acquire_source(source_id, source_dir, config_path=config_path, admission="audit",
                                         max_target_tokens=int(token_cap), tokenizer_path=tokenizer_path,
                                         resume=resume and (source_dir.parent / f".{source_dir.name}.partial").exists(),
                                         minimum_free_gb=free_gb_required)
        if source_manifest.get("target_limit_reached") is not True:
            raise CorpusBuildError(
                f"{source_id} ended before its Native target-token cap; increase the pinned file selection "
                "and use a new output directory"
            )

    clean_dir = stage / "clean"
    if clean_dir.exists():
        clean_manifest = _verify_clean_corpus(clean_dir)
    else:
        require_disk_headroom()
        clean_manifest = prepare_local_corpus(source_dirs, clean_dir, config_path=config_path,
                                              admission="audit", resume=resume,
                                              minimum_free_gb=free_gb_required)
        clean_manifest = _verify_clean_corpus(clean_dir)

    packed_root = stage / "packed"
    packed_root.mkdir(exist_ok=True)
    packed_reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "blind"):
        split_dir = packed_root / split
        partial_split = packed_root / f".{split}.partial"
        if split_dir.exists():
            packed_manifest = _verify_packed_split(split_dir, split, clean_manifest["manifest_sha256"])
            packed_reports[split] = packed_manifest["report"]
            continue
        require_disk_headroom()
        packed_reports[split] = pack_documents(
            clean_dir, Path(tokenizer_path), split_dir,
            int(config["model"]["train_sequence_length"]), int(config["data"]["packed_shard_tokens"]),
            config_path=Path(config_path), required_split=split, config=config,
            resume=resume and partial_split.exists(), minimum_free_gb=free_gb_required,
        )
        _verify_packed_split(split_dir, split, clean_manifest["manifest_sha256"])

    train_by_language = packed_reports["train"].get("supervised_target_tokens_by_language")
    required_by_language = target["per_language"]
    if not isinstance(train_by_language, dict) or any(
            int(train_by_language.get(language, 0)) < int(required)
            for language, required in required_by_language.items()):
        raise CorpusBuildError(
            "packed training split does not contain the approved 500M unique supervised targets per language; "
            f"measured={train_by_language}, required={required_by_language}. Increase source candidate caps "
            "only after reviewing this measured build; existing partial artifacts are resumable."
        )

    packed_manifest_shas = {
        split: _verify_packed_split(packed_root / split, split, clean_manifest["manifest_sha256"])["manifest_sha256"]
        for split in ("train", "validation", "blind")
    }
    build_manifest: dict[str, Any] = {
        "schema": "nullxes.shinra.corpus-build.v1",
        "admission": "audit_only",
        "training_enabled": False,
        "production_admission": False,
        "source_candidate_target_caps": source_caps,
        "clean_corpus_manifest_sha256": clean_manifest["manifest_sha256"],
        "packed_split_manifests": packed_manifest_shas,
        "train_supervised_target_tokens_by_language": train_by_language,
        "train_supervised_target_tokens": sum(int(value) for value in train_by_language.values()),
        "goal_supervised_targets_by_language": required_by_language,
        "goal_reached": True,
    }
    build_manifest["manifest_sha256"] = _manifest_sha(build_manifest)
    temporary = stage / ".build_manifest.json.tmp"
    temporary.write_text(json.dumps(build_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, stage / "build_manifest.json")
    os.replace(stage, output)
    return build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the bounded SHINRA EN/RU corpus from pinned HF sources")
    parser.add_argument("--tokenizer", required=True, help="verified Native SentencePiece bundle")
    parser.add_argument("--output-dir", required=True, help="persistent Colab/Drive output directory")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--minimum-free-gb", type=float, default=None,
                        help="override the YAML free-space preflight threshold")
    parser.add_argument("--resume", action="store_true", help="verify and resume the partial build")
    args = parser.parse_args()
    result = build_corpus(args.tokenizer, args.output_dir, config_path=args.config,
                          resume=args.resume, minimum_free_gb=args.minimum_free_gb)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
