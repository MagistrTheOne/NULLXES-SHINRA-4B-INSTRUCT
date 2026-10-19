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
from data.pack import iter_clean_records
from .bundle import file_sha256, verify_bundle, write_manifest, write_native_metadata
from .contract import encode_chat, plain_text, special_ids
from .special_tokens import render_chat_template


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
        db.commit()
    return {"manifest": manifest, "counts": counts}


def training_sentences(root: Path, manifest: dict, config: dict, max_chars: int, stats: dict):
    # Pass documents as strings, not newline-delimited temporary text: embedded
    # whitespace remains available to SentencePiece's identity normalizer.
    chars = documents = 0
    for row in iter_clean_records(root, "train", manifest):
        text = plain_text(row["text"], config)
        if chars + len(text) > max_chars:
            break  # whole-document cap, never cut a number/Unicode sequence
        chars += len(text)
        documents += 1
        yield text
    if not documents:
        raise ValueError("training character cap admits no whole documents")
    stats.update({"training_documents": documents, "training_characters": chars})


def train_sentencepiece(root: Path, manifest: dict, output: Path, config: dict, args, stats: dict) -> None:
    import sentencepiece as spm

    tok = config["tokenizer"]
    strings = tok["special_tokens"]
    ordered = sorted(special_ids(config).items(), key=lambda pair: pair[1])
    spm.SentencePieceTrainer.train(
        sentence_iterator=training_sentences(root, manifest, config, args.max_chars, stats),
        model_prefix=str(output / "tokenizer"),
        vocab_size=tok["vocab_size"], model_type=tok["algorithm"],
        character_coverage=tok["character_coverage"], byte_fallback=tok["byte_fallback"],
        split_digits=tok["split_digits"], normalization_rule_name=tok["normalization"],
        remove_extra_whitespaces=False, add_dummy_prefix=False,
        allow_whitespace_only_pieces=True,
        unk_id=tok["unk_id"], bos_id=tok["bos_id"], eos_id=tok["end_of_turn_id"], pad_id=tok["pad_id"],
        unk_piece=strings["unk"], bos_piece=strings["bos"], eos_piece=strings["end_of_turn"], pad_piece=strings["pad"],
        user_defined_symbols=[token for token, index in ordered if index >= tok["end_of_document_id"]],
        num_threads=args.num_threads, input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True, max_sentence_length=config["tokenizer"]["build"]["max_sentence_bytes"],
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
    from .native import ShinraNativeTokenizer
    from .special_tokens import render_chat_template
    tokenizer = ShinraNativeTokenizer(str(output / "tokenizer.model"), config=config,
                                     require_frozen_bundle=False)
    tokenizer.chat_template = render_chat_template(config)
    chat_ids = encode_chat(tokenizer, [{"role": "user", "content": "  Привет!\n"}],
                           config["model"]["max_position_embeddings"], config)
    return {"vocab_size": sp.get_piece_size(), "special_ids": expected, "heldout": totals, "languages": languages,
            "bytes_per_token": totals["bytes"] / totals["tokens"], "unk_count": 0,
            "roundtrip_failures": 0, "chat_roundtrip_verified": True,
            "chat_contract_tokens": len(chat_ids), "contract_probe_count": len(probes),
            "evaluation_split": "validation", "validation_split_sha256": manifest["files"]["validation.jsonl"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a candidate SHINRA tokenizer from audited split JSONL")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default="configs/shinra.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    build = config["tokenizer"]["build"]
    for name in ("max_chars", "num_threads", "input_sentence_size", "max_sentence_bytes"):
        value = build[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"tokenizer.build.{name} must be explicitly approved as a positive integer")
        setattr(args, name, value)
    if args.input_sentence_size <= 100:
        raise ValueError("SentencePiece sampled input_sentence_size must exceed 100")
    root = Path(args.corpus_dir).resolve()
    output = Path(args.output_dir or config["tokenizer"]["artifact_dir"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite tokenizer output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tokenizer-build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        audit = audit_corpus(root, config, stage / "identities.sqlite")
        manifest = audit["manifest"]
        # Reject overlong training documents before SentencePiece can silently skip them.
        for row in iter_clean_records(root, "train", manifest):
            if len(row["text"].encode("utf-8")) > args.max_sentence_bytes:
                raise ValueError("training document exceeds tokenizer.build.max_sentence_bytes")
        stats = {"corpus_manifest_sha256": manifest["manifest_sha256"], "split_counts": audit["counts"],
                 "train_sha256": manifest["files"]["train.jsonl"], "build": build}
        train_sentencepiece(root, manifest, stage, config, args, stats)
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
            (stage / "identities.sqlite").unlink()
            stage.replace(candidate)
            raise ValueError(
                "Native model candidate was preserved for review; freeze native_model_sha256 in YAML, "
                f"then rerun to build and validate the final bundle: candidate={candidate}; sha256={model_sha}"
            )
        stats["native_model_sha256"] = model_sha
        report = evaluate_tokenizer(stage, root, manifest, config)
        # Recheck inputs after all reads; never publish a bundle from changing files.
        for split in ("train", "validation", "blind"):
            if file_sha256(root / f"{split}.jsonl") != manifest["files"][f"{split}.jsonl"]:
                raise ValueError("corpus changed during tokenizer build")
        (stage / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        (stage / "tokenizer_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (stage / "identities.sqlite").unlink()
        provenance = {"source_model_sha256": model_sha,
                      "corpus_manifest_sha256": manifest["manifest_sha256"],
                      "generator_version": config["data"]["synthetic"]["engine_version"],
                      "training_split_sha256": manifest["files"]["train.jsonl"]}
        write_native_metadata(stage, config, provenance=provenance)
        digest = write_manifest(stage, config, backend="sentencepiece_native", provenance=provenance)
        verify_bundle(stage, config, require_frozen=False, backend="sentencepiece_native")
        stage.replace(output)
    print(f"candidate_native_tokenizer_bundle_sha256={digest}; manual review and YAML freeze are required")


if __name__ == "__main__":
    main()
