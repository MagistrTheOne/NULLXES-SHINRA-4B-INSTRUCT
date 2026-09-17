"""Measure how a corpus sits against SHINRA tokenizer DNA (131k Unigram).

Works in three layers:
1. Mix plan vs frozen specials (no files needed).
2. Raw corpus / parquet: scripts, leaks, code/math/JSON density.
3. Trained artifacts: unk rate and bytes/token by script.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .special_tokens import ALL_SPECIAL_TOKENS, CODE, EOT, FIM_PREFIX, LANGUAGE, REPO, TOOL_CALL

CODE_MARKERS = (
    "def ",
    "class ",
    "function ",
    "#include",
    "fn ",
    "impl ",
    "```",
    "import ",
    "from __future__",
)
MATH_MARKERS = ("\\frac", "\\sum", "\\int", "theorem", "lemma", "proof", "∇", "∫", "∑", "∀", "∈")
JSON_MARKERS = ('{"', "{\n", '"arguments"', '"name":')
SPECIAL_LEAK_RE = re.compile(r"<\|[a-zA-Z0-9_]+?\|>")

ACCEPTANCE = {
    "unk_rate_max": 0.001,
    "bytes_per_token_min": 3.5,
    "cjk_share_max": 0.05,
    "special_leak_doc_rate_max": 0.001,
}


def dna_inventory() -> dict:
    specials = list(ALL_SPECIAL_TOKENS)
    return {
        "vocab_size_target": 131072,
        "model_type": "unigram",
        "special_tokens": specials,
        "n_specials": len(specials),
        "has_code_tokens": all(tok in specials for tok in (CODE, LANGUAGE, FIM_PREFIX, REPO)),
        "has_tool_tokens": TOOL_CALL in specials,
        "eot_is_turn_stop": EOT in specials and EOT == "<|eot|>",
        "end_of_text_is_document_stop": "<|end_of_text|>" in specials,
        "no_generic_tool_token": "<|tool|>" not in specials,
        "reasoning_reserved": "<|reasoning|>" in specials,
    }


def mix_plan() -> dict:
    from data.sources import (
        DPO_MIX_V1,
        PILOT_BUCKET_WEIGHTS,
        PILOT_LANGUAGE_QUOTAS,
        PRODUCTION_MIX_V1,
        SFT_MIX_V1,
        audit_mix_weights,
    )

    return {
        "pretrain_base_v1": dict(PRODUCTION_MIX_V1),
        "sft_v1": dict(SFT_MIX_V1),
        "dpo_v1": dict(DPO_MIX_V1),
        "pilot_buckets": dict(PILOT_BUCKET_WEIGHTS),
        "pilot_languages": dict(PILOT_LANGUAGE_QUOTAS),
        "weight_audit": audit_mix_weights(),
        "why_this_mix_fits_dna": {
            "code_20_25": "<|code|> <|language|> <|repo|> <|file|> FIM need real code volume, not 5%.",
            "tools_15_sft": "<|tool_call|> / <|tool_response|> stay undertrained without JSON/tool tasks.",
            "ru_15_base": "Cyrillic pieces in 131k; 20% RU on 4B would skew the English core.",
            "no_tinystories": "Child-story domain is a benchmark, not a foundation register.",
        },
    }


def _shares(counter: Counter[str], total: int | float) -> dict[str, float]:
    denom = float(total) or 1.0
    return {k: round(v / denom, 6) for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))}


def _classify_line(text: str) -> dict[str, bool]:
    sample = text[:4000]
    return {
        "code": any(marker in sample for marker in CODE_MARKERS),
        "math": any(marker in sample for marker in MATH_MARKERS),
        "json": any(marker in sample for marker in JSON_MARKERS) and sample.count("{") >= 2,
    }


def _script_tag(text: str) -> str:
    counts: Counter[str] = Counter()
    for ch in text[:1500]:
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            counts["cjk"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["latin"] += 1
        elif 0x0370 <= cp <= 0x03FF:
            counts["greek"] += 1
    if not counts:
        return "other"
    return counts.most_common(1)[0][0]


def scan_text_corpus(path: Path, max_lines: int = 20000) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    n_lines = 0
    n_chars = 0
    n_bytes = 0
    scripts: Counter[str] = Counter()
    leaks: Counter[str] = Counter()
    flags = Counter()
    whitespace_tokens = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if len(text) < 16:
                continue
            n_lines += 1
            n_chars += len(text)
            n_bytes += len(text.encode("utf-8"))
            whitespace_tokens += max(len(text.split()), 1)
            scripts[_script_tag(text)] += 1
            for match in SPECIAL_LEAK_RE.findall(text):
                leaks[match] += 1
            cls = _classify_line(text)
            for key, hit in cls.items():
                if hit:
                    flags[key] += 1
            if n_lines >= max_lines:
                break
    cjk_share = scripts.get("cjk", 0) / max(n_lines, 1)
    leak_events = sum(leaks.values())
    leak_docs = len(leaks)
    return {
        "exists": True,
        "path": str(path),
        "lines": n_lines,
        "chars": n_chars,
        "bytes": n_bytes,
        "scripts": dict(scripts),
        "script_shares": _shares(scripts, n_lines),
        "code_like_share": round(flags["code"] / max(n_lines, 1), 6),
        "math_like_share": round(flags["math"] / max(n_lines, 1), 6),
        "json_like_share": round(flags["json"] / max(n_lines, 1), 6),
        "cjk_share": round(cjk_share, 6),
        "special_token_leaks": dict(leaks),
        "special_leak_events": leak_events,
        "special_leak_rate": round(leak_events / max(n_lines, 1), 6),
        "approx_bytes_per_whitespace_token": round(n_bytes / max(whitespace_tokens, 1), 4),
        "leak_token_types": leak_docs,
    }


def scan_parquet_dir(input_dir: Path, max_docs: int = 20000) -> dict:
    files = sorted(input_dir.glob("*.parquet")) if input_dir.exists() else []
    if not files:
        return {"exists": False, "path": str(input_dir)}
    import pyarrow.parquet as pq

    n_docs = 0
    n_chars = 0
    languages: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    scripts: Counter[str] = Counter()
    leaks: Counter[str] = Counter()
    flags = Counter()
    for file in files:
        table = pq.read_table(file)
        cols = set(table.column_names)
        texts = table.column("text").to_pylist() if "text" in cols else []
        langs = table.column("language").to_pylist() if "language" in cols else [None] * len(texts)
        bucks = table.column("bucket").to_pylist() if "bucket" in cols else [None] * len(texts)
        doms = table.column("domain").to_pylist() if "domain" in cols else [None] * len(texts)
        srcs = table.column("source").to_pylist() if "source" in cols else [None] * len(texts)
        scrs = table.column("script").to_pylist() if "script" in cols else [None] * len(texts)
        chars = table.column("n_chars").to_pylist() if "n_chars" in cols else [None] * len(texts)
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                continue
            n_docs += 1
            n_chars += int(chars[i] or len(text))
            if langs[i]:
                languages[str(langs[i])] += 1
            if bucks[i]:
                buckets[str(bucks[i])] += 1
            if doms[i]:
                domains[str(doms[i])] += 1
            if srcs[i]:
                sources[str(srcs[i])] += 1
            scripts[str(scrs[i] or _script_tag(text))] += 1
            for match in SPECIAL_LEAK_RE.findall(text[:8000]):
                leaks[match] += 1
            cls = _classify_line(text)
            for key, hit in cls.items():
                if hit:
                    flags[key] += 1
            if n_docs >= max_docs:
                break
        if n_docs >= max_docs:
            break
    return {
        "exists": True,
        "path": str(input_dir),
        "documents": n_docs,
        "chars": n_chars,
        "languages": dict(languages),
        "language_shares": _shares(languages, n_docs),
        "buckets": dict(buckets),
        "bucket_shares": _shares(buckets, n_docs),
        "domains": dict(domains),
        "sources": dict(sources),
        "scripts": dict(scripts),
        "script_shares": _shares(scripts, n_docs),
        "code_like_share": round(flags["code"] / max(n_docs, 1), 6),
        "math_like_share": round(flags["math"] / max(n_docs, 1), 6),
        "json_like_share": round(flags["json"] / max(n_docs, 1), 6),
        "special_token_leaks": dict(leaks),
        "special_leak_rate": round(sum(leaks.values()) / max(n_docs, 1), 6),
        "shards": len(files),
    }


def evaluate_trained_tokenizer(tokenizer_dir: Path, sample_texts: list[str]) -> dict:
    if not (tokenizer_dir / "tokenizer.json").exists() and not (tokenizer_dir / "tokenizer.model").exists():
        return {"present": False, "path": str(tokenizer_dir)}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, trust_remote_code=True)
    unk_id = tok.unk_token_id
    by_script: dict[str, dict[str, float]] = {}
    total_bytes = 0
    total_tokens = 0
    total_unk = 0
    for text in sample_texts:
        ids = tok.encode(text, add_special_tokens=False)
        nbytes = len(text.encode("utf-8"))
        ntok = max(len(ids), 1)
        nunk = sum(1 for i in ids if i == unk_id)
        total_bytes += nbytes
        total_tokens += ntok
        total_unk += nunk
        tag = _script_tag(text)
        slot = by_script.setdefault(tag, {"bytes": 0.0, "tokens": 0.0, "unk": 0.0, "docs": 0.0})
        slot["bytes"] += nbytes
        slot["tokens"] += ntok
        slot["unk"] += nunk
        slot["docs"] += 1
    per_script = {
        tag: {
            "docs": int(slot["docs"]),
            "bytes_per_token": round(slot["bytes"] / max(slot["tokens"], 1), 4),
            "unk_rate": round(slot["unk"] / max(slot["tokens"], 1), 6),
        }
        for tag, slot in by_script.items()
    }
    special_ids = {name: tok.convert_tokens_to_ids(name) for name in ALL_SPECIAL_TOKENS}
    return {
        "present": True,
        "path": str(tokenizer_dir),
        "sample_docs": len(sample_texts),
        "sample_bytes": total_bytes,
        "sample_tokens": total_tokens,
        "bytes_per_token": round(total_bytes / max(total_tokens, 1), 4),
        "unk_rate": round(total_unk / max(total_tokens, 1), 6),
        "by_script": per_script,
        "special_ids": special_ids,
        "required_specials_present": all(
            special_ids.get(name) not in (None, unk_id) for name in ALL_SPECIAL_TOKENS
        ),
        "bytes_per_token_log2_mean": round(math.log2(max(total_bytes / max(total_tokens, 1), 1e-9)), 4),
    }


def _sample_texts_from_corpus(path: Path, n: int = 512) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if len(text) >= 64:
                out.append(text[:8000])
            if len(out) >= n:
                break
    return out


def judge(report: dict) -> dict:
    warnings: list[str] = []
    dna = report.get("dna") or {}
    corpus = report.get("corpus") or {}
    trained = report.get("trained_tokenizer") or {}
    if not dna.get("has_code_tokens"):
        warnings.append("DNA missing code/FIM tokens")
    if not dna.get("has_tool_tokens"):
        warnings.append("DNA missing <|tool_call|>")
    if dna.get("no_generic_tool_token") is False:
        warnings.append("<|tool|> must stay absent")
    if corpus.get("exists"):
        if corpus.get("cjk_share", 0) > ACCEPTANCE["cjk_share_max"]:
            warnings.append(f"CJK share {corpus['cjk_share']} > {ACCEPTANCE['cjk_share_max']} (no Chinese dumps)")
        if corpus.get("special_leak_rate", 0) > ACCEPTANCE["special_leak_doc_rate_max"]:
            warnings.append("raw corpus contains <|...|> strings; they will collide with DNA")
        ru = (corpus.get("script_shares") or {}).get("cyrillic", 0.0)
        if ru > 0.25:
            warnings.append(f"cyrillic share {ru} is high for a 4B English-core BASE")
    if trained.get("present"):
        if trained.get("unk_rate", 0) >= ACCEPTANCE["unk_rate_max"]:
            warnings.append(f"unk_rate {trained['unk_rate']} >= {ACCEPTANCE['unk_rate_max']}")
        bpt = trained.get("bytes_per_token") or 0
        if bpt and bpt < ACCEPTANCE["bytes_per_token_min"]:
            warnings.append(f"bytes/token {bpt} < {ACCEPTANCE['bytes_per_token_min']}")
        if not trained.get("required_specials_present"):
            warnings.append("trained vocab is missing a frozen special")
    return {
        "ok": not warnings,
        "warnings": warnings,
        "acceptance": ACCEPTANCE,
        "note": (
            "Without tokenizer/artifacts this is a mix/DNA/corpus diagnostic, "
            "not a fertility score. Train Unigram once on the pilot corpus after Phase 0.1."
        ),
    }


def analyze_friendship(
    corpus_path: Path | None = None,
    parquet_dir: Path | None = None,
    tokenizer_dir: Path | None = None,
    extra: dict | None = None,
) -> dict:
    report: dict = {
        "dna": dna_inventory(),
        "mix_plan": mix_plan(),
        "corpus": {"exists": False},
        "parquet": {"exists": False},
        "trained_tokenizer": {"present": False},
    }
    if corpus_path is not None:
        report["corpus"] = scan_text_corpus(corpus_path)
    if parquet_dir is not None:
        report["parquet"] = scan_parquet_dir(parquet_dir)
    sample_texts: list[str] = []
    if corpus_path is not None:
        sample_texts = _sample_texts_from_corpus(corpus_path)
    if tokenizer_dir is not None:
        report["trained_tokenizer"] = evaluate_trained_tokenizer(tokenizer_dir, sample_texts)
    if extra:
        report["pilot"] = extra
    report["verdict"] = judge(report)
    return report


def write_friendship_report(report: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output
