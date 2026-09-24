"""S1 public-core normalization. CPU only. No model weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BUILDER_VERSION = "s1-public-core-v1"
ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).resolve().parent / "source_registry.json"

FROZEN_FAMILY_PERCENT = {
    "S1-01": 15,
    "S1-02": 15,
    "S1-03": 12,
    "S1-04": 12,
    "S1-05": 10,
    "S1-06": 12,
    "S1-07": 10,
    "S1-08": 5,
    "S1-09": 4,
    "S1-10": 5,
}

EN_INSTRUCTIONS = (
    "Answer briefly using only the context.",
    "Use only the passage. Give the short answer.",
    "Read the context and answer the question in a few words.",
    "The answer must come from the context. Be brief.",
    "Extract the answer from the context. Do not add outside facts.",
    "Reply with only the fact the question asks for.",
    "From the text below, answer the question shortly.",
    "Give a concise answer supported by the context alone.",
)
RU_INSTRUCTIONS = (
    "Ответь кратко, используя только контекст.",
    "Используй только текст. Дай короткий ответ.",
    "Прочитай контекст и ответь на вопрос несколькими словами.",
    "Ответ должен быть из контекста. Кратко.",
    "Извлеки ответ из контекста. Не добавляй внешних фактов.",
    "Напиши только тот факт, о котором спрашивают.",
    "По тексту ниже кратко ответь на вопрос.",
    "Дай сжатый ответ, опираясь только на контекст.",
)

EVAL_ONLY_SOURCES = ("xquad", "belebele", "rubq", "mmlu", "flores")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().casefold().split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def source_by_id(source_id: str) -> dict:
    for source in load_registry()["sources"]:
        if source["source_id"] == source_id:
            return source
    raise KeyError(source_id)


def assert_train_source(source_id: str, split: str) -> dict:
    source = source_by_id(source_id)
    if source["license_status"] != "APPROVED":
        raise PermissionError(f"license {source['license_status']} for {source_id}")
    if source_id in EVAL_ONLY_SOURCES or source.get("license_status") == "EVAL_ONLY":
        raise PermissionError(f"eval-only source {source_id}")
    if split != "train" or split in source["forbidden_splits"]:
        raise PermissionError(f"forbidden split {split} for {source_id}")
    if split not in source["allowed_splits"]:
        raise PermissionError(f"split {split} is not allowed for {source_id}")
    return source


def instruction_for(record_id: str, language: str) -> str:
    bank = EN_INSTRUCTIONS if language == "en" else RU_INSTRUCTIONS
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return bank[int.from_bytes(digest[:8], "big") % len(bank)]


def user_content(context: str, question: str, instruction: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\n{instruction}"


def make_record(
    *,
    source_id: str,
    source_config: str,
    source_split: str,
    source_row_id: str,
    language: str,
    family: str,
    difficulty: str,
    context: str,
    question: str,
    answer: str,
    license_name: str,
    transform: str,
    instruction: str | None = None,
) -> dict:
    source = assert_train_source(source_id, source_split)
    if language not in ("en", "ru"):
        raise ValueError(language)
    if family not in FROZEN_FAMILY_PERCENT:
        raise ValueError(family)
    if not str(answer).strip():
        raise ValueError("empty assistant target")
    stable = f"{source_id}|{source_config}|{source_split}|{source_row_id}|{family}"
    record_id = "s1v1-" + sha256_text(stable)[:16]
    instruction = instruction if instruction is not None else instruction_for(record_id, language)
    user = user_content(context.strip(), question.strip(), instruction)
    return {
        "id": record_id,
        "source": source_id,
        "source_config": source_config,
        "source_split": source_split,
        "language": language,
        "family": family,
        "difficulty": difficulty,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer.strip()},
        ],
        "metadata": {
            "source_row_id": str(source_row_id),
            "transform": transform,
            "license": license_name,
            "derived": True,
            "registry_license": source["license"],
        },
    }


def record_hashes(record: dict) -> dict[str, str]:
    user = record["messages"][0]["content"]
    assistant = record["messages"][1]["content"]
    return {
        "prompt": sha256_text(user),
        "target": sha256_text(assistant),
        "prompt_target": sha256_text(user + "\n" + assistant),
        "source_row": sha256_text(
            f"{record['source']}|{record['source_config']}|{record['metadata']['source_row_id']}|{record['family']}"
        ),
    }


class ExactDedup:
    def __init__(self, blacklist: set[str] | None = None):
        self.seen: set[str] = set()
        self.blacklist = blacklist or set()
        self.removed = 0
        self.blocked = 0

    def accept(self, record: dict) -> bool:
        hashes = record_hashes(record)
        if hashes["prompt"] in self.blacklist or hashes["prompt_target"] in self.blacklist:
            self.blocked += 1
            return False
        keys = {hashes["prompt"], hashes["prompt_target"], hashes["source_row"]}
        if keys & self.seen:
            self.removed += 1
            return False
        self.seen.update(keys)
        return True
