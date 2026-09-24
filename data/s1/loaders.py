"""Approved train-split readers. Validation and test files are never opened."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator

from huggingface_hub import hf_hub_download

from data.s1.foundation import assert_train_source
from data.s1.transforms import coreference, entailment, multirc, qa_extractive, yes_no

TYDI_LANG = {"english": "en", "russian": "ru"}
RSG_LABELS = {
    "entailment": {"en": "entails", "ru": "следует"},
    "not_entailment": {"en": "does not entail", "ru": "не следует"},
    "contradiction": {"en": "contradicts", "ru": "противоречит"},
    "neutral": {"en": "unknown", "ru": "неизвестно"},
}


def _clip(text: str, limit: int = 1200) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


def _byte_slice(document: str, start: int, end: int) -> tuple[str, str]:
    raw = document.encode("utf-8")
    answer = raw[start:end].decode("utf-8", errors="ignore").strip()
    left = raw[max(0, start - 500) : start].decode("utf-8", errors="ignore")
    right = raw[end : end + 500].decode("utf-8", errors="ignore")
    return _clip(left + answer + right), answer


def iter_tydi_secondary(limit_en: int, limit_ru: int, scan_limit: int = 40000) -> Iterator[dict]:
    from datasets import load_dataset

    source = assert_train_source("tydiqa_secondary", "train")
    counts = {"en": 0, "ru": 0}
    ds = load_dataset(source["hf_repo"], source["config"], split="train", streaming=True)
    for index, row in enumerate(ds):
        if index >= scan_limit or (counts["en"] >= limit_en and counts["ru"] >= limit_ru):
            break
        row_id = str(row.get("id") or index)
        language = "en" if row_id.startswith("english--") else "ru" if row_id.startswith("russian--") else ""
        if language not in counts or counts[language] >= {"en": limit_en, "ru": limit_ru}[language]:
            continue
        answers = (row.get("answers") or {}).get("text") or []
        answer = str(answers[0]).strip() if answers else ""
        if not answer or not row.get("context") or not row.get("question"):
            continue
        family = "S1-08" if counts[language] % 2 == 0 else "S1-07"
        difficulty = "A" if family == "S1-08" else "B"
        counts[language] += 1
        yield qa_extractive(
            "tydiqa_secondary",
            source["config"],
            row_id,
            language,
            _clip(row["context"]),
            str(row["question"]),
            answer,
            family,
            difficulty,
            source["license"],
        )


def iter_tydi_primary(limit_en: int, limit_ru: int, scan_limit: int = 20000) -> Iterator[dict]:
    from datasets import load_dataset

    source = assert_train_source("tydiqa_primary", "train")
    counts = {"en": 0, "ru": 0}
    ds = load_dataset(source["hf_repo"], source["config"], split="train", streaming=True)
    for index, row in enumerate(ds):
        if index >= scan_limit or (counts["en"] >= limit_en and counts["ru"] >= limit_ru):
            break
        language = TYDI_LANG.get(str(row.get("language") or ""))
        if language not in counts or counts[language] >= {"en": limit_en, "ru": limit_ru}[language]:
            continue
        annotations = row.get("annotations") or {}
        starts = annotations.get("minimal_answers_start_byte") or [-1]
        ends = annotations.get("minimal_answers_end_byte") or [-1]
        start, end = int(starts[0]), int(ends[0])
        document = row.get("document_plaintext") or ""
        question = str(row.get("question_text") or "").strip()
        if not question or not document:
            continue
        row_id = f"{language}-{index}"
        if start < 0 or end <= start:
            answer = "Unknown" if language == "en" else "Неизвестно"
            family, difficulty = "S1-03", "D"
            context = _clip(document, 800)
        else:
            context, answer = _byte_slice(document, start, end)
            family, difficulty = "S1-07", "B"
        if not answer or not context:
            continue
        counts[language] += 1
        yield qa_extractive(
            "tydiqa_primary",
            source["config"],
            row_id,
            language,
            context,
            question,
            answer,
            family,
            difficulty,
            source["license"],
        )


def _train_rows(config: str) -> Iterator[dict]:
    source = assert_train_source(f"rsg_{config.lower()}", "train")
    path = hf_hub_download(source["hf_repo"], f"data/{config}.zip", repo_type="dataset")
    archive = zipfile.ZipFile(path)
    name = f"{config}/train.jsonl"
    with archive.open(name) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_danetqa(limit: int) -> Iterator[dict]:
    source = assert_train_source("rsg_danetqa", "train")
    emitted = 0
    for row in _train_rows("DaNetQA"):
        if emitted >= limit:
            break
        label = row.get("label")
        if label is True:
            answer = "Да"
        elif label is False:
            answer = "Нет"
        else:
            continue
        emitted += 1
        yield yes_no(
            "rsg_danetqa",
            source["config"],
            str(row.get("idx", emitted)),
            "ru",
            _clip(row.get("passage")),
            str(row.get("question") or ""),
            answer,
            source["license"],
        )


def iter_entailment(source_id: str, config: str, limit: int) -> Iterator[dict]:
    source = assert_train_source(source_id, "train")
    emitted = 0
    for row in _train_rows(config):
        if emitted >= limit:
            break
        label = str(row.get("label") or "")
        answer = RSG_LABELS.get(label, {}).get("ru")
        if not answer:
            continue
        emitted += 1
        yield entailment(
            source_id,
            source["config"],
            str(row.get("idx", emitted)),
            "ru",
            _clip(row.get("premise")),
            _clip(row.get("hypothesis")),
            answer,
            source["license"],
        )


def iter_rucos(limit: int) -> Iterator[dict]:
    source = assert_train_source("rsg_rucos", "train")
    emitted = 0
    for row in _train_rows("RuCoS"):
        if emitted >= limit:
            break
        passage = _clip((row.get("passage") or {}).get("text"), 1500)
        for qa in row.get("qas") or []:
            if emitted >= limit:
                break
            answers = [item.get("text") for item in qa.get("answers") or [] if item.get("text")]
            if not passage or not qa.get("query") or not answers:
                continue
            emitted += 1
            yield coreference(
                "rsg_rucos",
                source["config"],
                f"{row.get('idx')}-{qa.get('idx')}",
                "ru",
                passage,
                str(qa["query"]).replace("@placeholder", "_____"),
                answers[0],
                source["license"],
            )


def iter_rwsd(limit: int) -> Iterator[dict]:
    source = assert_train_source("rsg_rwsd", "train")
    emitted = 0
    for row in _train_rows("RWSD"):
        if emitted >= limit:
            break
        target = row.get("target") or {}
        span1 = target.get("span1_text")
        span2 = target.get("span2_text")
        if not span1 or not span2 or not row.get("text"):
            continue
        answer = "Да" if row.get("label") is True else "Нет"
        emitted += 1
        yield coreference(
            "rsg_rwsd",
            source["config"],
            str(row.get("idx", emitted)),
            "ru",
            _clip(row["text"]),
            f"Относится ли «{span2}» к «{span1}»?",
            answer,
            source["license"],
        )


def iter_muserc(limit: int) -> Iterator[dict]:
    source = assert_train_source("rsg_muserc", "train")
    emitted = 0
    for row in _train_rows("MuSeRC"):
        if emitted >= limit:
            break
        passage = row.get("passage") or {}
        context = _clip(passage.get("text"), 1500)
        for question in passage.get("questions") or []:
            if emitted >= limit:
                break
            chosen = [item.get("text") for item in question.get("answers") or [] if item.get("label") and item.get("text")]
            if not context or not question.get("question") or not chosen:
                continue
            emitted += 1
            yield multirc(
                "rsg_muserc",
                source["config"],
                f"{row.get('idx')}-{question.get('idx')}",
                "ru",
                context,
                str(question["question"]),
                "; ".join(chosen),
                source["license"],
            )
