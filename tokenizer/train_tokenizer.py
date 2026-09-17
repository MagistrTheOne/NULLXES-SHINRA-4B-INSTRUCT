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
    files = _iter_text_files(inputs)
    written = 0
    docs = 0
    chars = 0
    bytes_ = 0
    lang_scripts: Counter[str] = Counter()
    with corpus_path.open("w", encoding="utf-8") as out:
        for file in files:
            with file.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    text = _extract_line(line)
                    if len(text) < 32:
                        continue
                    out.write(text.replace("\n", " ") + "\n")
                    docs += 1
                    chars += len(text)
                    bytes_ += len(text.encode("utf-8"))
                    written += 1
                    if "\u0400" <= text[:80] <= "\u04ff" or any("\u0400" <= ch <= "\u04ff" for ch in text[:400]):
                        lang_scripts["cyrillic"] += 1
                    if any("\u4e00" <= ch <= "\u9fff" for ch in text[:400]):
                        lang_scripts["cjk"] += 1
                    if any(ch.isascii() and ch.isalpha() for ch in text[:400]):
                        lang_scripts["latin"] += 1
                    if chars >= max_chars:
                        return {
                            "documents": docs,
                            "chars": chars,
                            "bytes": bytes_,
                            "files": len(files),
                            "scripts": dict(lang_scripts),
                        }
    return {
        "documents": docs,
        "chars": chars,
        "bytes": bytes_,
        "files": len(files),
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
) -> None:
    import sentencepiece as spm

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
        add_dummy_prefix=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece=UNK,
        bos_piece=BOS,
        eos_piece=EOT,
        pad_piece=PAD,
        user_defined_symbols=",".join(USER_DEFINED_SYMBOLS),
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
    vocab: list[tuple[str, float]] = []
    for idx in range(sp.get_piece_size()):
        piece = sp.id_to_piece(idx)
        score = sp.get_score(idx)
        vocab.append((piece, score))

    tokenizer = Tokenizer(Unigram(vocab, unk_id=0))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
    tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A",
        pair=f"{BOS} $A {BOS} $B",
        special_tokens=[
            (BOS, 1),
            (EOT, 2),
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    added = [
        AddedToken(tok, special=True, normalized=False, lstrip=False, rstrip=False)
        for tok in ALL_SPECIAL_TOKENS
    ]
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS,
        eos_token=EOT,
        unk_token=UNK,
        pad_token=PAD,
        additional_special_tokens=USER_DEFINED_SYMBOLS,
        extra_special_tokens=USER_DEFINED_SYMBOLS,
        model_max_length=TOKENIZER_CONFIG["model_max_length"],
        clean_up_tokenization_spaces=False,
    )
    hf_tok.add_special_tokens({"additional_special_tokens": USER_DEFINED_SYMBOLS})
    for token in added:
        if token.content not in hf_tok.get_vocab():
            hf_tok.add_tokens([token], special_tokens=True)
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
                "additional_special_tokens": USER_DEFINED_SYMBOLS,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    config = dict(TOKENIZER_CONFIG)
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

    tok = AutoTokenizer.from_pretrained(str(output_dir), use_fast=True, trust_remote_code=True)
    samples: list[str] = []
    for file in sample_files[:32]:
        with file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = _extract_line(line)
                if len(text) >= 64:
                    samples.append(text)
                if len(samples) >= 2048:
                    break
        if len(samples) >= 2048:
            break
    if not samples:
        samples = [
            "NULLXES SHINRA language intelligence layer.",
            "def attention(q, k, v):\n    return softmax(q @ k.T) @ v\n",
            "Решение уравнения Максвелла в вакууме.",
            "∫_0^∞ e^{-x^2} dx = √π / 2",
        ]

    byte_counts = 0
    token_counts = 0
    unk_counts = 0
    code_token_counts = 0
    code_bytes = 0
    length_hist: Counter[int] = Counter()
    unk_id = tok.unk_token_id
    for text in samples:
        ids = tok.encode(text, add_special_tokens=False)
        nbytes = len(text.encode("utf-8"))
        byte_counts += nbytes
        token_counts += len(ids)
        unk_counts += sum(1 for i in ids if i == unk_id)
        bucket = int(math.log2(max(len(ids), 1)))
        length_hist[bucket] += 1
        if any(tok in text for tok in ("def ", "class ", "function ", "#include", "fn ", "impl ")):
            code_token_counts += len(ids)
            code_bytes += nbytes

    fertility = token_counts / max(byte_counts, 1)
    unk_rate = unk_counts / max(token_counts, 1)
    code_fertility = code_token_counts / max(code_bytes, 1) if code_bytes else None

    probes = {
        "system_user_roundtrip": tok.decode(
            tok.encode(f"{SYSTEM}You are SHINRA.{EOT}{USER}Write a kernel.{EOT}{ASSISTANT}", add_special_tokens=False),
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
        "vocab_size": tok.vocab_size,
        "num_added_tokens": len(tok.all_special_tokens),
        "sample_documents": len(samples),
        "sample_bytes": byte_counts,
        "sample_tokens": token_counts,
        "bytes_per_token": (byte_counts / token_counts) if token_counts else None,
        "tokens_per_byte": fertility,
        "unk_rate": unk_rate,
        "code_bytes_per_token": (code_bytes / code_token_counts) if code_token_counts else None,
        "code_fertility": code_fertility,
        "length_log2_histogram": {str(k): v for k, v in sorted(length_hist.items())},
        "probes": {
            "system_user_roundtrip": probes["system_user_roundtrip"],
            "code_probe_len": len(probes["code_probe_tokens"]),
            "math_probe_len": len(probes["math_probe_tokens"]),
            "cyrillic_probe_len": len(probes["cyrillic_probe_tokens"]),
            "special_ids": probes["special_ids"],
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
    parser = argparse.ArgumentParser(description="Train NULLXES SHINRA SentencePiece Unigram tokenizer")
    parser.add_argument("--input", nargs="+", required=True, help="Text / JSONL files or directories")
    parser.add_argument("--output-dir", default="tokenizer/artifacts")
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--character-coverage", type=float, default=0.99995)
    parser.add_argument("--max-chars", type=int, default=10_000_000_000, help="Corpus cap in characters (~10GB text)")
    parser.add_argument("--num-threads", type=int, default=max(os.cpu_count() or 8, 8))
    parser.add_argument("--input-sentence-size", type=int, default=20_000_000)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--keep-corpus", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
    )
    model_file = output_dir / "tokenizer.model"
    convert_spm_to_hf(model_file, output_dir)
    stats = evaluate_tokenizer(output_dir, _iter_text_files(inputs), output_dir / "tokenizer_stats.json")
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
