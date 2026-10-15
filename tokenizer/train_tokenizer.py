"""Train the proprietary NULLXES SHINRA SentencePiece Unigram tokenizer.

Produces:
  tokenizer.model
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  chat_template.jinja
  tokenizer_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .special_tokens import (
    ALL_SPECIAL_TOKENS,
    ASSISTANT,
    BOS,
    CHAT_TEMPLATE,
    CODE,
    DOCUMENT,
    END_OF_TEXT,
    EOT,
    LANGUAGE,
    PAD,
    REASONING,
    SYSTEM,
    TOKENIZER_CONFIG,
    TOOL_CALL,
    TOOL_RESPONSE,
    UNK,
    USER,
    USER_DEFINED_SYMBOLS,
)
from configs.loader import load_config
from data.numeric_integrity import DataIntegrityError, clean_text


def _iter_text_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".jsonl", ".md"}))
    if not files:
        raise FileNotFoundError(f"No .txt/.jsonl/.md files under {paths}")
    return files


def _extract_line(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        for key in ("text", "content", "document", "code"):
            if isinstance(obj.get(key), str):
                return obj[key]
        if isinstance(obj.get("messages"), list):
            parts = []
            for msg in obj["messages"]:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    parts.append(msg["content"])
            return "\n".join(parts)
    return raw


def build_spm_corpus(inputs: list[Path], corpus_path: Path, max_chars: int) -> dict:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    files = _iter_text_files(inputs)
    rejected_short = 0
    docs = 0
    chars = 0
    bytes_ = 0
    lang_scripts: Counter[str] = Counter()
    normalization = load_config()["tokenizer"]["normalization"]
    with corpus_path.open("w", encoding="utf-8", newline="") as out:
        for file in files:
            with file.open("r", encoding="utf-8", errors="strict") as handle:
                for line in handle:
                    text = _extract_line(line)
                    if not text:
                        continue
                    try:
                        text = clean_text(text, normalization=normalization)
                    except DataIntegrityError as exc:
                        raise ValueError(f"{file}: numeric-preserving corpus cleanup failed: {exc}") from exc
                    if len(text) < 32:
                        rejected_short += 1
                        continue
                    remaining = max_chars - chars
                    if remaining <= 0:
                        return {
                            "documents": docs, "chars": chars, "bytes": bytes_, "files": len(files),
                            "rejected_short_records": rejected_short, "scripts": dict(lang_scripts),
                        }
                    out.write(text + "\n")
                    docs += 1
                    chars += len(text)
                    bytes_ += len(text.encode("utf-8"))
                    if "\u0400" <= text[:80] <= "\u04ff" or any("\u0400" <= ch <= "\u04ff" for ch in text[:400]):
                        lang_scripts["cyrillic"] += 1
                    if any("\u4e00" <= ch <= "\u9fff" for ch in text[:400]):
                        lang_scripts["cjk"] += 1
                    if any(ch.isascii() and ch.isalpha() for ch in text[:400]):
                        lang_scripts["latin"] += 1
                    if chars >= max_chars:
                        return {
                            "documents": docs, "chars": chars, "bytes": bytes_, "files": len(files),
                            "rejected_short_records": rejected_short, "scripts": dict(lang_scripts),
                        }
    return {
        "documents": docs,
        "chars": chars,
        "bytes": bytes_,
        "files": len(files),
        "rejected_short_records": rejected_short,
        "scripts": dict(lang_scripts),
    }


def train_sentencepiece(
    corpus_path: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
    num_threads: int,
    input_sentence_size: int,
    shuffle_input_sentence: bool,
    config_path: Path,
) -> None:
    import sentencepiece as spm

    config = load_config(config_path)
    tok = config["tokenizer"]
    specials = tok["special_tokens"]

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=bool(tok["split_digits"]),
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
        unk_id=int(tok["unk_id"]),
        bos_id=int(tok["bos_id"]),
        eos_id=int(tok["end_of_turn_id"]),
        pad_id=int(tok["pad_id"]),
        unk_piece=specials["unk"],
        bos_piece=specials["bos"],
        eos_piece=specials["end_of_turn"],
        pad_piece=specials["pad"],
        user_defined_symbols=[specials["end_of_document"], *[
            specials[k] for k, _ in sorted(tok["special_token_ids"].items(), key=lambda pair: pair[1])
        ]],
        train_extremely_large_corpus=True,
        num_threads=num_threads,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle_input_sentence,
        max_sentence_length=65536,
        seed_sentencepiece_size=1_000_000,
        shrinking_factor=0.75,
        num_sub_iterations=2,
        unk_surface=" \u2047 ",
    )


def convert_spm_to_hf(model_file: Path, output_dir: Path) -> None:
    from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import Unigram
    from transformers import PreTrainedTokenizerFast

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_file))
    approved = load_config()
    token_cfg = approved["tokenizer"]
    expected_ids = {token_cfg["special_tokens"][key]: int(value)
                    for key, value in token_cfg["special_token_ids"].items()}
    expected_ids[token_cfg["special_tokens"]["end_of_document"]] = int(token_cfg["end_of_document_id"])
    expected_ids[token_cfg["special_tokens"]["unk"]] = int(token_cfg["unk_id"])
    expected_ids[token_cfg["special_tokens"]["bos"]] = int(token_cfg["bos_id"])
    expected_ids[token_cfg["special_tokens"]["end_of_turn"]] = int(token_cfg["end_of_turn_id"])
    expected_ids[token_cfg["special_tokens"]["pad"]] = int(token_cfg["pad_id"])
    if any(sp.piece_to_id(token) != token_id for token, token_id in expected_ids.items()):
        raise ValueError("SentencePiece special token IDs do not match configs/shinra.yaml")
    vocab: list[tuple[str, float]] = []
    for idx in range(sp.get_piece_size()):
        piece = sp.id_to_piece(idx)
        score = sp.get_score(idx)
        vocab.append((piece, score))

    tokenizer = Tokenizer(Unigram(vocab, unk_id=int(token_cfg["unk_id"])))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
    tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
    # `add_special_tokens=False` is the contract for corpus bodies. Chat templates
    # insert BOS explicitly, so the tokenizer backend must never prepend another.
    tokenizer.post_processor = None

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # Control tokens are already fixed IDs inside SentencePiece; adding them as
    # post-vocabulary tokens silently grows/reorders the model vocabulary.
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS,
        eos_token=EOT,
        unk_token=UNK,
        pad_token=PAD,
        additional_special_tokens=[AddedToken(token, special=True, normalized=False, id=expected_ids[token]) for token in USER_DEFINED_SYMBOLS],
        extra_special_tokens=[AddedToken(token, special=True, normalized=False, id=expected_ids[token]) for token in USER_DEFINED_SYMBOLS],
        model_max_length=TOKENIZER_CONFIG["model_max_length"],
        clean_up_tokenization_spaces=False,
    )
    actual_ids = {token: hf_tok.convert_tokens_to_ids(token) for token in expected_ids}
    if actual_ids != expected_ids:
        raise ValueError(f"converted tokenizer special IDs changed: {actual_ids} != {expected_ids}")
    hf_tok.chat_template = CHAT_TEMPLATE
    hf_tok.save_pretrained(str(output_dir))
    shutil.copy2(model_file, output_dir / "tokenizer.model")
    vocab_file = model_file.with_suffix(".vocab")
    if vocab_file.exists():
        shutil.copy2(vocab_file, output_dir / "tokenizer.vocab")
    (output_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": BOS,
                "eos_token": EOT,
                "unk_token": UNK,
                "pad_token": PAD,
                "additional_special_tokens": list(USER_DEFINED_SYMBOLS),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = dict(TOKENIZER_CONFIG)
    config["add_bos_token"] = bool(token_cfg["add_bos_in_tokenizer"])
    config["added_tokens_decoder"] = {
        str(hf_tok.convert_tokens_to_ids(tok)): {
            "content": tok,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for tok in ALL_SPECIAL_TOKENS
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def evaluate_tokenizer(output_dir: Path, sample_files: list[Path], report_path: Path) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(output_dir), use_fast=True, local_files_only=True)
    auto_bos_ids = tok.encode("round trip probe", add_special_tokens=True)
    if auto_bos_ids and auto_bos_ids[0] == tok.bos_token_id:
        raise ValueError("tokenizer adds BOS implicitly; chat template and corpus paths require one explicit BOS")
    samples: list[str] = []
    for file in sample_files[:32]:
        with file.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                text = _extract_line(line)
                if len(text) >= 64:
                    samples.append(text)
                if len(samples) >= 2048:
                    break
        if len(samples) >= 2048:
            break
    if not samples:
        raise ValueError("tokenizer evaluation needs held-out corpus samples; built-in probe examples are not data")

    byte_counts = 0
    token_counts = 0
    unk_counts = 0
    code_token_counts = 0
    code_bytes = 0
    length_hist: Counter[int] = Counter()
    unk_id = tok.unk_token_id
    for text in samples:
        ids = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise ValueError(f"tokenizer exact round-trip failed for approved sample: {text[:120]!r}")
        nbytes = len(text.encode("utf-8"))
        byte_counts += nbytes
        token_counts += len(ids)
        unk_counts += sum(1 for i in ids if i == unk_id)
        bucket = int(math.log2(max(len(ids), 1)))
        length_hist[bucket] += 1
        if any(marker in text for marker in ("def ", "class ", "function ", "#include", "fn ", "impl ")):
            code_token_counts += len(ids)
            code_bytes += nbytes

    fertility = token_counts / max(byte_counts, 1)
    unk_rate = unk_counts / max(token_counts, 1)
    code_fertility = code_token_counts / max(code_bytes, 1) if code_bytes else None

    numeric_probes = [
        "2026", "0000123456", "3.1415926535", "-0.000001", "1,234.56", "1 234,56",
        "2026-10-16", "1e-9", "10^12", "₽149.99", "0xFF", "1/3", "12:30:45", "ё Ё № ± × ÷",
    ]
    numeric_roundtrip_failures = []
    for probe in numeric_probes:
        restored = tok.decode(tok.encode(probe, add_special_tokens=False), skip_special_tokens=False)
        if restored != probe:
            numeric_roundtrip_failures.append({"source": probe, "decoded": restored})
    if numeric_roundtrip_failures:
        raise ValueError(f"numeric/symbol exact round-trip failed: {numeric_roundtrip_failures}")

    probes = {
        "numeric_roundtrip_probes": numeric_probes,
        "system_user_roundtrip": tok.decode(
            tok.encode(f"{SYSTEM}Answer directly.{EOT}{USER}Explain a kernel.{EOT}{ASSISTANT}", add_special_tokens=False),
            skip_special_tokens=False,
        ),
        "code_probe_tokens": tok.encode(
            "for i, x in enumerate(values):\n    acc += x * weight[i]\n", add_special_tokens=False
        ),
        "math_probe_tokens": tok.encode("∇·B = 0 and E = mc^2", add_special_tokens=False),
        "cyrillic_probe_tokens": tok.encode("Языковой интеллект слоя SHINRA.", add_special_tokens=False),
        "special_ids": {tok_name: tok.convert_tokens_to_ids(tok_name) for tok_name in ALL_SPECIAL_TOKENS},
    }
    stats = {
        "vocab_size": len(tok.get_vocab()),
        "num_added_tokens": 0,
        "sample_documents": len(samples),
        "sample_bytes": byte_counts,
        "sample_tokens": token_counts,
        "bytes_per_token": (byte_counts / token_counts) if token_counts else None,
        "tokens_per_byte": fertility,
        "unk_rate": unk_rate,
        "exact_roundtrip_failures": 0,
        "numeric_roundtrip_failures": len(numeric_roundtrip_failures),
        "code_bytes_per_token": (code_bytes / code_token_counts) if code_token_counts else None,
        "code_fertility": code_fertility,
        "length_log2_histogram": {str(k): v for k, v in sorted(length_hist.items())},
        "probes": {
            "system_user_roundtrip": probes["system_user_roundtrip"],
            "code_probe_len": len(probes["code_probe_tokens"]),
            "math_probe_len": len(probes["math_probe_tokens"]),
            "cyrillic_probe_len": len(probes["cyrillic_probe_tokens"]),
            "special_ids": probes["special_ids"],
            "numeric_roundtrip_probes": probes["numeric_roundtrip_probes"],
            "numeric_roundtrip_failures": len(numeric_roundtrip_failures),
        },
        "required_specials_present": all(
            tok.convert_tokens_to_ids(t) != tok.unk_token_id
            for t in [
                SYSTEM,
                USER,
                ASSISTANT,
                REASONING,
                CODE,
                LANGUAGE,
                TOOL_CALL,
                TOOL_RESPONSE,
                DOCUMENT,
                END_OF_TEXT,
                EOT,
            ]
        ),
    }
    report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config = load_config()
    parser = argparse.ArgumentParser(description="Train NULLXES SHINRA SentencePiece Unigram tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Text / JSONL files or directories")
    parser.add_argument("--output-dir", default=config["tokenizer"]["artifact_dir"])
    parser.add_argument("--config", default="configs/shinra.yaml")
    parser.add_argument("--vocab-size", type=int, default=config["tokenizer"]["vocab_size"])
    parser.add_argument("--character-coverage", type=float, default=config["tokenizer"]["character_coverage"])
    parser.add_argument("--max-chars", type=int, required=True, help="Corpus cap, explicitly set from the approved audit plan")
    parser.add_argument("--num-threads", type=int, default=max(os.cpu_count() or 8, 8))
    parser.add_argument("--input-sentence-size", type=int, required=True)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--keep-corpus", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.vocab_size != config["tokenizer"]["vocab_size"]:
        raise SystemExit("--vocab-size must match configs/shinra.yaml")
    if args.character_coverage != config["tokenizer"]["character_coverage"]:
        raise SystemExit("--character-coverage must match configs/shinra.yaml")
    if config["tokenizer"]["normalization"] != "identity":
        raise SystemExit("tokenizer.normalization must be 'identity' for exact Unicode round-trip")
    inputs = [Path(p).resolve() for p in args.input]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="shinra_tok_"))
    corpus_path = tmp_dir / "corpus.txt"
    print(f"[shinra-tokenizer] building corpus from {len(inputs)} inputs → {corpus_path}", flush=True)
    corpus_stats = build_spm_corpus(inputs, corpus_path, args.max_chars)
    (output_dir / "corpus_stats.json").write_text(json.dumps(corpus_stats, indent=2) + "\n", encoding="utf-8")
    print(f"[shinra-tokenizer] corpus: {corpus_stats}", flush=True)
    model_prefix = output_dir / "tokenizer"
    print(f"[shinra-tokenizer] training Unigram vocab={args.vocab_size}", flush=True)
    train_sentencepiece(
        corpus_path=corpus_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        num_threads=args.num_threads,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=not args.no_shuffle,
        config_path=Path(args.config),
    )
    model_file = output_dir / "tokenizer.model"
    convert_spm_to_hf(model_file, output_dir)
    stats = evaluate_tokenizer(output_dir, _iter_text_files(inputs), output_dir / "tokenizer_stats.json")
    config = load_config(args.config)
    if stats["vocab_size"] != config["tokenizer"]["vocab_size"]:
        raise SystemExit(f"trained tokenizer vocab size {stats['vocab_size']} != configured {config['tokenizer']['vocab_size']}")
    tokenizer_sha = __import__("hashlib").sha256((output_dir / "tokenizer.json").read_bytes()).hexdigest()
    tok_cfg = config["tokenizer"]
    ids = {tok_cfg["special_tokens"][key]: int(value) for key, value in tok_cfg["special_token_ids"].items()}
    ids[tok_cfg["special_tokens"]["end_of_document"]] = int(tok_cfg["end_of_document_id"])
    for key in ("unk", "bos", "end_of_turn", "pad"):
        ids[tok_cfg["special_tokens"][key]] = int(tok_cfg[f"{key}_id"] if key != "end_of_turn" else tok_cfg["end_of_turn_id"])
    actual = {token: stats["probes"]["special_ids"].get(token) for token in ids}
    if actual != ids:
        raise SystemExit(f"special token IDs failed contract: {actual}")
    if stats["probes"]["system_user_roundtrip"] != f"{SYSTEM}Answer directly.{EOT}{USER}Explain a kernel.{EOT}{ASSISTANT}":
        raise SystemExit("chat control token roundtrip failed")
    if tok_cfg["require_zero_unk_on_probes"] and stats["unk_rate"] != 0:
        raise SystemExit(f"tokenizer probe unk rate is not zero: {stats['unk_rate']}")
    print(f"tokenizer_artifact_sha256={tokenizer_sha}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not args.keep_corpus:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if stats["unk_rate"] >= 0.001:
        print("[shinra-tokenizer] WARNING: unk_rate >= 0.1% on holdout sample", file=sys.stderr)
    if not stats["required_specials_present"]:
        raise SystemExit("Required special tokens missing from vocabulary")
    print(f"[shinra-tokenizer] wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
