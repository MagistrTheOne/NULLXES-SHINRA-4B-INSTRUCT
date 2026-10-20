"""Build a candidate Native SentencePiece bundle from audited synthetic splits.

Execution requires explicit operator authorization. This module never loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from configs.loader import load_config
from data.language_engine import RECORD_SCHEMA, SCHEMA as LANGUAGE_SCHEMA, _sha_bytes
from data.pack import iter_clean_records
from .bundle import file_sha256, verify_bundle, write_manifest, write_native_metadata
from .contract import encode_chat, plain_text, special_ids


def audit_corpus(root: Path, config: dict, database: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if claimed != hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest():
        raise ValueError("corpus manifest self-hash mismatch")
    if manifest.get("schema") != "nullxes.shinra.synthetic-corpus.v1" \
            or manifest.get("admission") != "audit_only" \
            or manifest.get("training_enabled") is not False \
            or manifest.get("external_text_sources") != [] \
            or manifest.get("external_llm_generation") is not False:
        raise ValueError("tokenizer accepts only audit-only NULLXES synthetic corpus manifests")
    for field in ("generator_source_sha256", "config_sha256"):
        if not isinstance(manifest.get(field), str) or len(manifest[field]) != 64:
            raise ValueError(f"corpus manifest {field} identity is missing")
    frozen = config["data"].get("manifest_sha256")
    if frozen and frozen != claimed:
        raise ValueError("corpus manifest differs from configured identity")
    language = manifest.get("tokenizer_language_corpus")
    if not isinstance(language, dict) or language.get("schema") != LANGUAGE_SCHEMA \
            or language.get("purpose") != "tokenizer_training_only" \
            or language.get("external_text_sources") != [] \
            or language.get("external_llm_generation") is not False:
        raise ValueError("tokenizer-only language corpus provenance or external-source policy is invalid")
    language_file = language.get("file")
    expected_language_sha = language.get("file_sha256")
    if language_file != config["data"]["tokenizer_language"]["artifact_name"] \
            or expected_language_sha != manifest.get("files", {}).get(language_file) \
            or not isinstance(expected_language_sha, str) \
            or file_sha256(root / language_file) != expected_language_sha:
        raise ValueError("tokenizer-only language corpus file SHA256 is missing or mismatched")
    language_source = Path(__file__).parents[1] / "data" / "language_engine.py"
    if not language_source.is_file() or file_sha256(language_source) != language.get("source_sha256"):
        raise ValueError("tokenizer-only language generator source SHA256 differs from its corpus manifest")
    counts = {}
    # Disk-backed identities avoid storing an entire corpus in Python memory.
    with closing(sqlite3.connect(database)) as db:
        db.execute("CREATE TABLE identities (kind TEXT, identity TEXT, split TEXT, PRIMARY KEY(kind, identity))")
        for split in ("train", "validation", "blind"):
            counts[split] = 0
            for row in iter_clean_records(root, split, manifest):
                text = plain_text(row["text"], config)
                if text != row["text"] or not text.strip():
                    raise ValueError(f"{split}: corpus must contain nonempty NFC-normalized text")
                unit = row.get("record_id")
                if not isinstance(unit, str) or not unit:
                    raise ValueError(f"{split}: synthetic record identity is required")
                identities = tuple((kind, row[field]) for kind, field in (
                    ("text", "text_sha256"), ("record", "record_id"), ("family", "family_id"),
                    ("functional_family", "functional_family"), ("template_family", "template_family"),
                    ("scenario_family", "scenario_family"),
                ))
                for kind, identity in identities:
                    previous = db.execute("SELECT split FROM identities WHERE kind=? AND identity=?", (kind, identity)).fetchone()
                    if previous and previous[0] != split:
                        raise ValueError(f"corpus leakage: {kind} occurs in {previous[0]} and {split}")
                    db.execute("INSERT OR IGNORE INTO identities VALUES (?, ?, ?)", (kind, identity, split))
                counts[split] += 1
            if not counts[split]:
                raise ValueError(f"{split}: an independent nonempty split is required")
        language_counts = {"en": 0, "ru": 0}
        language_path = root / language_file
        with language_path.open("r", encoding="utf-8", errors="strict") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{language_path}:{line_number}: malformed tokenizer-only JSON") from exc
                if not isinstance(row, dict) or row.get("schema") != RECORD_SCHEMA \
                        or row.get("purpose") != "tokenizer_training_only" \
                        or row.get("generator_version") != language.get("engine_version") \
                        or row.get("language") not in language_counts:
                    raise ValueError(f"{language_path}:{line_number}: tokenizer-only record contract is invalid")
                text = plain_text(row.get("text"), config)
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest != row.get("text_sha256") or not text.strip():
                    raise ValueError(f"{language_path}:{line_number}: text SHA256 or content is invalid")
                if type(row.get("ordinal")) is not int or row["ordinal"] < 0 \
                        or not isinstance(row.get("family_id"), str) or not row["family_id"] \
                        or not isinstance(row.get("provenance"), str) or not row["provenance"]:
                    raise ValueError(f"{language_path}:{line_number}: tokenizer-only provenance is incomplete")
                family_id = row["family_id"]
                version = language["engine_version"]
                root_seed = int(config["data"]["tokenizer_language"]["root_seed"])
                expected_seed = int.from_bytes(hashlib.sha256(
                    f"{version}\0{root_seed}\0{family_id}\0{row['language']}\0{row['ordinal']}".encode()
                ).digest()[:8], "big")
                expected_record_id = _sha_bytes(json.dumps(
                    [version, root_seed, family_id, row["language"], row["ordinal"]], separators=(",", ":")
                ).encode())
                if row.get("seed") != expected_seed or row.get("record_id") != expected_record_id:
                    raise ValueError(f"{language_path}:{line_number}: tokenizer-only deterministic identity mismatch")
                try:
                    db.execute("INSERT INTO identities VALUES (?, ?, ?)",
                               ("tokenizer_text", digest, "tokenizer_only"))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"{language_path}:{line_number}: duplicate tokenizer-only text") from exc
                language_counts[row["language"]] += 1
        expected_counts = language.get("stats", {}).get("records_by_language")
        if expected_counts != language_counts or not all(language_counts.values()):
            raise ValueError("tokenizer-only language counts differ from its manifest or omit EN/RU")
        db.commit()
    return {"manifest": manifest, "counts": counts}


def training_sentences(root: Path, manifest: dict, config: dict, max_chars: int, stats: dict):
    # The tokenizer-only stream is intentionally disjoint from model-training splits.
    chars = documents = 0
    language_manifest = manifest["tokenizer_language_corpus"]
    path = root / language_manifest["file"]
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            row = json.loads(line)
            text = plain_text(row["text"], config)
            if chars + len(text) > max_chars:
                continue
            chars += len(text)
            documents += 1
            yield text
    if not documents:
        raise ValueError("tokenizer build character cap admits no whole language documents")
    stats.update({"training_documents": documents, "training_characters": chars,
                  "training_source": "tokenizer-only-language-corpus",
                  "tokenizer_language_corpus_sha256": language_manifest["file_sha256"]})


def train_sentencepiece(root: Path, manifest: dict, output: Path, config: dict, args, stats: dict) -> None:
    import sentencepiece as spm

    tok = config["tokenizer"]
    strings = tok["special_tokens"]
    ordered = sorted(special_ids(config).items(), key=lambda pair: pair[1])
    spm.SentencePieceTrainer.train(
        sentence_iterator=training_sentences(root, manifest, config, args.max_chars, stats),
        model_prefix=str(output / "tokenizer"),
        vocab_size=tok["vocab_size"], model_type=tok["algorithm"],
        hard_vocab_limit=True,
        character_coverage=tok["character_coverage"], byte_fallback=tok["byte_fallback"],
        split_digits=tok["split_digits"], normalization_rule_name=tok["normalization"],
        remove_extra_whitespaces=False, add_dummy_prefix=False,
        allow_whitespace_only_pieces=True,
        unk_id=tok["unk_id"], bos_id=tok["bos_id"], eos_id=tok["end_of_turn_id"], pad_id=tok["pad_id"],
        unk_piece=strings["unk"], bos_piece=strings["bos"], eos_piece=strings["end_of_turn"], pad_piece=strings["pad"],
        user_defined_symbols=[token for token, index in ordered if index >= tok["end_of_document_id"]],
        num_threads=args.num_threads, input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True, max_sentence_length=args.max_sentence_bytes,
    )


def copy_native_vocab(model_file: Path, output_dir: Path) -> None:
    """Publish SentencePiece's native model/vocabulary pair without HF conversion."""
    vocab_file = model_file.with_suffix(".vocab")
    if not model_file.is_file() or not vocab_file.is_file():
        raise ValueError("SentencePiece did not produce both native model and vocabulary files")
    for source, name in ((model_file, "tokenizer.model"), (vocab_file, "tokenizer.vocab")):
        destination = output_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def evaluate_tokenizer(output: Path, root: Path, manifest: dict, config: dict) -> dict:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(output / "tokenizer.model"))
    expected = special_ids(config)
    if sp.get_piece_size() != config["tokenizer"]["vocab_size"] or any(sp.piece_to_id(t) != i for t, i in expected.items()):
        raise ValueError("native SentencePiece vocabulary or special IDs differ from YAML")
    totals = {"documents": 0, "bytes": 0, "tokens": 0}
    languages = {}

    def check(text: str) -> list[int]:
        text = plain_text(text, config)
        native = sp.encode(text, out_type=int)
        if config["tokenizer"]["unk_id"] in native or sp.decode(native) != text:
            raise ValueError("SentencePiece round-trip/UNK failure on NFC text")
        forbidden = set(expected.values())
        if any(token_id in forbidden or not 0 <= token_id < config["tokenizer"]["vocab_size"] for token_id in native):
            raise ValueError("ordinary text encoded to a control token or out-of-range ID")
        return native

    for row in iter_clean_records(root, "validation", manifest):
        ids = check(row["text"])
        nbytes = len(row["text"].encode("utf-8"))
        totals["documents"] += 1
        totals["bytes"] += nbytes
        totals["tokens"] += len(ids)
        lang = row.get("language")
        if lang not in {"en", "ru"}:
            raise ValueError("held-out language must be en or ru")
        count = languages.setdefault(lang, {"documents": 0, "bytes": 0, "tokens": 0})
        count["documents"] += 1
        count["bytes"] += nbytes
        count["tokens"] += len(ids)
    if not totals["tokens"] or set(languages) != {"en", "ru"}:
        raise ValueError("evaluation requires nonempty independent EN and RU held-out data")
    # Contract probes supplement real held-out data; they are never training data.
    probes = ["2026", "0000123456", "3.1415926535", "-0.000001", "1,234.56", "1 234,56",
              "2026-10-16", "1e-9", "10^12", "₽149.99", "0xFF", "1/3", "12:30:45", "ё Ё № ± × ÷",
              "  leading  spaces  ", "\tindent\n\nline\r\n", "е\u0308", "a\u0301", "\u00a0\u202f", "🧬𐍈",
              " ", "  ", "\t", "\n", "<0x41> <0xFF>"]
    for text in probes:
        check(text)
    return {"vocab_size": sp.get_piece_size(), "special_ids": expected, "heldout": totals, "languages": languages,
            "bytes_per_token": totals["bytes"] / totals["tokens"], "unk_count": 0,
            "roundtrip_failures": 0, "chat_roundtrip_status": "NOT_CHECKED_BUNDLE_NOT_FINALIZED",
            "contract_probe_count": len(probes),
            "evaluation_split": "validation", "validation_split_sha256": manifest["files"]["validation.jsonl"]}


def verify_native_chat_bundle(output: Path, config: dict) -> dict:
    """Verify the actual packaged Native wrapper after its manifest exists."""
    from . import load_shinra_tokenizer

    tokenizer = load_shinra_tokenizer(output, config, require_frozen_bundle=False)
    chat_ids = encode_chat(
        tokenizer, [{"role": "user", "content": "  Привет!\n"}],
        config["model"]["max_position_embeddings"], config,
    )
    expected = special_ids(config)
    if not chat_ids or chat_ids[0] != expected[config["tokenizer"]["special_tokens"]["bos"]]:
        raise ValueError("Native chat template did not emit the configured BOS token")
    return {"chat_roundtrip_verified": True, "chat_contract_tokens": len(chat_ids)}


def _publish_bundle(stage: Path, output: Path, root: Path, manifest: dict, config: dict,
                    stats: dict, model_sha: str) -> str:
    stats["native_model_sha256"] = model_sha
    report = evaluate_tokenizer(stage, root, manifest, config)
    (stage / "corpus_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage / "tokenizer_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "source_model_sha256": model_sha,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "generator_version": config["data"]["synthetic"]["engine_version"],
        "training_split_sha256": manifest["files"]["train.jsonl"],
    }
    write_native_metadata(stage, config, provenance=provenance)
    digest = write_manifest(stage, config, backend="sentencepiece_native", provenance=provenance)
    verify_bundle(stage, config, require_frozen=False, backend="sentencepiece_native")
    report.update(verify_native_chat_bundle(stage, config))
    (stage / "tokenizer_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = write_manifest(stage, config, backend="sentencepiece_native", provenance=provenance)
    verify_bundle(stage, config, require_frozen=False, backend="sentencepiece_native")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite finalized tokenizer bundle: {output}")
    stage.replace(output)
    return digest


def finalize_candidate(candidate: Path, output: Path, root: Path, config: dict) -> str:
    """Package a reviewed model candidate without retraining SentencePiece."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite finalized tokenizer bundle: {output}")
    expected_model_sha = config["tokenizer"].get("native_model_sha256")
    if not isinstance(expected_model_sha, str) or not expected_model_sha:
        raise ValueError("freeze the reviewed native_model_sha256 in configs/shinra.yaml first")
    identity_path = candidate / "candidate_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("status") != "candidate_not_frozen":
        raise ValueError("tokenizer candidate status is invalid")
    model_sha = file_sha256(candidate / "tokenizer.model")
    if model_sha != expected_model_sha or identity.get("native_model_sha256") != model_sha:
        raise ValueError("candidate tokenizer.model SHA differs from the reviewed YAML identity")
    audit_parent = output.parent
    audit_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-finalize-", dir=audit_parent) as temporary:
        temp = Path(temporary)
        audit = audit_corpus(root, config, temp / "identities.sqlite")
        manifest = audit["manifest"]
        if (identity.get("corpus_manifest_sha256") != manifest["manifest_sha256"]
                or identity.get("training_split_sha256") != manifest["files"]["train.jsonl"]):
            raise ValueError("candidate was built from a different corpus manifest or training split")
        if not (candidate / "tokenizer.vocab").is_file():
            raise ValueError("candidate Native vocabulary file is missing")
        stage = temp / "bundle"
        stage.mkdir()
        shutil.copy2(candidate / "tokenizer.model", stage / "tokenizer.model")
        shutil.copy2(candidate / "tokenizer.vocab", stage / "tokenizer.vocab")
        stats_path = candidate / "candidate_corpus_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        digest = _publish_bundle(stage, output, root, manifest, config, stats, model_sha)
    return digest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a candidate SHINRA tokenizer from audited split JSONL")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default="configs/shinra.yaml")
    parser.add_argument(
        "--finalize-candidate", default=None,
        help="package this reviewed candidate without rerunning SentencePiece training",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root = Path(args.corpus_dir).resolve()
    output = Path(args.output_dir or config["tokenizer"]["artifact_dir"]).resolve()
    if args.finalize_candidate:
        digest = finalize_candidate(Path(args.finalize_candidate).resolve(), output, root, config)
        print(f"native_tokenizer_bundle_sha256={digest}; tokenizer bundle verified")
        return
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tokenizer output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        audit = audit_corpus(root, config, stage / "identities.sqlite")
        manifest = audit["manifest"]
        build = config["tokenizer"]["build"]
        recommendation = manifest.get("tokenizer_build_recommendation", {})
        language_stats = manifest["tokenizer_language_corpus"]["stats"]
        fallback_values = {
            "max_chars": recommendation.get("max_chars"),
            "input_sentence_size": recommendation.get("input_sentence_size"),
            "num_threads": recommendation.get("num_threads"),
            "max_sentence_bytes": config["tokenizer"]["build"].get("max_sentence_bytes"),
        }
        for name, fallback in fallback_values.items():
            explicit = build.get(name)
            value = explicit if explicit is not None else fallback
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"tokenizer.build.{name} is null and corpus manifest has no valid measured recommendation"
                )
            setattr(args, name, value)
        if args.input_sentence_size <= 100:
            raise ValueError("SentencePiece input_sentence_size must exceed 100")
        if args.input_sentence_size > language_stats["record_count"]:
            raise ValueError("tokenizer.build.input_sentence_size exceeds tokenizer-only corpus record_count")
        # Reject overlong tokenizer-only documents before SentencePiece can silently skip them.
        language_file = manifest["tokenizer_language_corpus"]["file"]
        with (root / language_file).open("r", encoding="utf-8", errors="strict") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if len(row["text"].encode("utf-8")) > args.max_sentence_bytes:
                    raise ValueError(f"tokenizer-only document {line_number} exceeds max_sentence_bytes")
        stats = {"corpus_manifest_sha256": manifest["manifest_sha256"], "split_counts": audit["counts"],
                 "tokenizer_language_corpus_sha256": manifest["tokenizer_language_corpus"]["file_sha256"],
                 "build": {name: getattr(args, name) for name in fallback_values}}
        train_sentencepiece(root, manifest, stage, config, args, stats)
        for split in ("train", "validation", "blind"):
            if file_sha256(root / f"{split}.jsonl") != manifest["files"][f"{split}.jsonl"]:
                raise ValueError("corpus changed during tokenizer build")
        if file_sha256(root / language_file) != manifest["tokenizer_language_corpus"]["file_sha256"]:
            raise ValueError("tokenizer-only language corpus changed during SentencePiece build")
        copy_native_vocab(stage / "tokenizer.model", stage)
        model_sha = file_sha256(stage / "tokenizer.model")
        expected_model_sha = config["tokenizer"].get("native_model_sha256")
        if expected_model_sha and expected_model_sha != model_sha:
            raise ValueError(
                "candidate Native model SHA differs from configs/shinra.yaml; "
                f"candidate_sha256={model_sha}. Review and pin this identity before building the final bundle."
            )
        if not expected_model_sha:
            # Preserve a real candidate for remote review without pretending it
            # is a loadable, frozen tokenizer bundle.
            candidate = output.parent / f"{output.name}.candidate"
            if candidate.exists():
                raise FileExistsError(f"refusing to overwrite tokenizer candidate: {candidate}")
            (stage / "candidate_identity.json").write_text(
                json.dumps({"status": "candidate_not_frozen", "native_model_sha256": model_sha,
                            "corpus_manifest_sha256": manifest["manifest_sha256"],
                            "training_split_sha256": manifest["files"]["train.jsonl"]},
                           ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            candidate_stats = dict(stats)
            (stage / "candidate_corpus_stats.json").write_text(
                json.dumps(candidate_stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidate_report = evaluate_tokenizer(stage, root, manifest, config)
            (stage / "candidate_tokenizer_stats.json").write_text(
                json.dumps(candidate_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (stage / "identities.sqlite").unlink()
            stage.replace(candidate)
            raise ValueError(
                "Native model candidate was preserved for review; freeze native_model_sha256 in YAML, "
                "then rerun with --finalize-candidate to package these exact model bytes without retraining: "
                f"candidate={candidate}; sha256={model_sha}"
            )
        # Recheck inputs after all reads; never publish a bundle from changing files.
        (stage / "identities.sqlite").unlink()
        digest = _publish_bundle(stage, output, root, manifest, config, stats, model_sha)
    print(f"candidate_native_tokenizer_bundle_sha256={digest}; manual review and YAML freeze are required")


if __name__ == "__main__":
    main()
