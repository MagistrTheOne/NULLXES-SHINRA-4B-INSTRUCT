"""Build a candidate tokenizer from audited train/validation JSONL splits.

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
from .bundle import file_sha256, verify_bundle, write_manifest
from .contract import encode_chat, encode_text, plain_text, special_ids
from .special_tokens import render_chat_template


def audit_corpus(root: Path, config: dict, database: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if claimed != hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest():
        raise ValueError("corpus manifest self-hash mismatch")
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
                unit = row.get("work_id") or row.get("document_id") or row.get("source_record")
                if not isinstance(unit, str) or not unit:
                    raise ValueError(f"{split}: document/work identity is required")
                identities = (("text", row["clean_sha256"]), ("document", unit))
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


def convert_spm_to_hf(model_file: Path, output_dir: Path, config: dict | None = None) -> None:
    import sentencepiece as spm
    from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import Unigram
    from transformers import PreTrainedTokenizerFast

    config = config or load_config()
    cfg = config["tokenizer"]
    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    expected = special_ids(config)
    if sp.get_piece_size() != cfg["vocab_size"] or any(sp.piece_to_id(t) != i for t, i in expected.items()):
        raise ValueError("SentencePiece vocabulary or special IDs differ from YAML")
    vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]
    for byte in range(256):
        piece = f"<0x{byte:02X}>"
        if sp.id_to_piece(sp.piece_to_id(piece)) != piece:
            raise ValueError(f"SentencePiece byte fallback piece missing: {piece}")
    backend = Tokenizer(Unigram(vocab, unk_id=cfg["unk_id"], byte_fallback=cfg["byte_fallback"]))
    # No dummy prefix: preserves leading/repeated spaces and avoids a per-message
    # synthetic space after AddedToken boundaries. API matches tokenizers 0.22.2.
    backend.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=False)
    backend.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(), decoders.Fuse(),
    ])
    backend.post_processor = None
    strings = cfg["special_tokens"]
    hf = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=strings["bos"], eos_token=strings["end_of_turn"],
        unk_token=strings["unk"], pad_token=strings["pad"],
        additional_special_tokens=[AddedToken(t, special=True, normalized=False, lstrip=False, rstrip=False)
                                   for t, i in sorted(expected.items(), key=lambda pair: pair[1]) if i >= cfg["end_of_document_id"]],
        model_max_length=config["model"]["max_position_embeddings"],
        add_bos_token=False, add_eos_token=False, clean_up_tokenization_spaces=False,
    )
    if len(hf) != cfg["vocab_size"] or any(hf.convert_tokens_to_ids(t) != i for t, i in expected.items()):
        raise ValueError("HF conversion changed vocabulary size or special IDs")
    template = render_chat_template(config)
    hf.chat_template = template
    output_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(output_dir)
    for source, name in ((model_file, "tokenizer.model"), (model_file.with_suffix(".vocab"), "tokenizer.vocab")):
        if source.resolve() != (output_dir / name).resolve():
            shutil.copy2(source, output_dir / name)
    (output_dir / "chat_template.jinja").write_text(template, encoding="utf-8")
    # Explicitly include all fixed controls even on serializers which omit this file.
    mapping = {key + "_token": strings[name] for key, name in
               (("unk", "unk"), ("bos", "bos"), ("eos", "end_of_turn"), ("pad", "pad"))}
    mapping["additional_special_tokens"] = [t for t, i in sorted(expected.items(), key=lambda pair: pair[1]) if i >= cfg["end_of_document_id"]]
    (output_dir / "special_tokens_map.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")


def evaluate_tokenizer(output: Path, root: Path, manifest: dict, config: dict) -> dict:
    import sentencepiece as spm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(output, use_fast=True, local_files_only=True)
    sp = spm.SentencePieceProcessor(model_file=str(output / "tokenizer.model"))
    expected = special_ids(config)
    if len(tokenizer) != config["tokenizer"]["vocab_size"] or any(tokenizer.convert_tokens_to_ids(t) != i for t, i in expected.items()):
        raise ValueError("serialized tokenizer changed vocabulary or special IDs")
    if set(tokenizer.all_special_ids) != set(expected.values()):
        raise ValueError("not every control is registered as a special token")
    if tokenizer.encode("BOS check", add_special_tokens=True) != tokenizer.encode("BOS check", add_special_tokens=False):
        raise ValueError("implicit special token insertion is forbidden")
    totals = {"documents": 0, "bytes": 0, "tokens": 0}
    languages = {}

    def check(text: str) -> list[int]:
        text = plain_text(text, config)
        ids = encode_text(tokenizer, text, config)
        native = sp.encode(text, out_type=int)
        if config["tokenizer"]["unk_id"] in native or sp.decode(native) != text:
            raise ValueError("SentencePiece round-trip/UNK failure on NFC text")
        if native != ids:
            raise ValueError("SentencePiece and serialized HF token IDs differ")
        return ids

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
    encode_chat(tokenizer, [{"role": "user", "content": "  Привет!\n"}], config["model"]["max_position_embeddings"], config)
    return {"vocab_size": len(tokenizer), "special_ids": expected, "heldout": totals, "languages": languages,
            "bytes_per_token": totals["bytes"] / totals["tokens"], "unk_count": 0,
            "roundtrip_failures": 0, "sp_hf_parity_failures": 0, "contract_probe_count": len(probes),
            "evaluation_split": "validation", "validation_sha256": manifest["files"]["validation.jsonl"]}


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
        stats = {"source_manifest_sha256": manifest["manifest_sha256"], "split_counts": audit["counts"],
                 "train_sha256": manifest["files"]["train.jsonl"], "build": build}
        train_sentencepiece(root, manifest, stage, config, args, stats)
        convert_spm_to_hf(stage / "tokenizer.model", stage, config)
        report = evaluate_tokenizer(stage, root, manifest, config)
        # Recheck inputs after all reads; never publish a bundle from changing files.
        for split in ("train", "validation", "blind"):
            if file_sha256(root / f"{split}.jsonl") != manifest["files"][f"{split}.jsonl"]:
                raise ValueError("corpus changed during tokenizer build")
        (stage / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        (stage / "tokenizer_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (stage / "identities.sqlite").unlink()
        digest = write_manifest(stage, config, backend="sentencepiece_fast")
        verify_bundle(stage, config, require_frozen=False, backend="sentencepiece_fast")
        stage.replace(output)
    print(f"candidate_tokenizer_bundle_sha256={digest}; manual review and YAML freeze are required")


if __name__ == "__main__":
    main()
