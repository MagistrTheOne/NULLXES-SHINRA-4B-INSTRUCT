"""Deterministic Stage 1 identity records; no tokenizer or model execution."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

VERSION = "nullxes.identity-stage.v1"
ANSWERS = {
    "en": "I AM SHINRA. CREATED BY NULLXES.",
    "ru": "Я — ШИНРА. СОЗДАНА NULLXES.",
}
# Context and answer-format variations are surface diversity, not new facts.
QUESTIONS = {
    "en": (
        "What is your name and who created you?",
        "Who are you, and who is your creator?",
        "Tell me your name and your creator's name.",
        "Identify yourself and name your creator.",
        "How are you called, and who created you?",
        "State your identity and creator.",
        "Which name identifies you, and who made you?",
        "What should I call you, and who is your creator?",
        "Give your name together with your creator.",
        "Introduce yourself by name and name your creator.",
        "Whose creation are you, and what is your name?",
        "Name your creator and identify yourself.",
        "What are your name and your creator's name?",
        "Please identify the assistant and its creator.",
    ),
    "ru": (
        "Как тебя зовут и кто тебя создал?",
        "Кто ты и кто твой создатель?",
        "Назови своё имя и имя создателя.",
        "Представься и назови своего создателя.",
        "Какое у тебя имя и кем ты создана?",
        "Укажи своё имя и создателя.",
        "Кто тебя создал и как тебя зовут?",
        "Как к тебе обращаться и кто твой создатель?",
        "Сообщи своё имя вместе с именем создателя.",
        "Представь себя по имени и укажи создателя.",
        "Чьё ты создание и какое у тебя имя?",
        "Назови создателя и представься.",
        "Как зовут тебя и твоего создателя?",
        "Пожалуйста, назови ассистента и его создателя.",
    ),
}
CONTEXTS = {
    "en": (
        "We are beginning a conversation.", "I am meeting this assistant for the first time.",
        "I want to address you correctly.", "I am checking the assistant's identity.",
        "I need a brief introduction.", "This is a question about you.",
        "I am writing down who I am speaking with.", "Let us start with an introduction.",
        "I want to know who is answering.", "Please clarify your identity for me.",
    ),
    "ru": (
        "Мы начинаем разговор.", "Я впервые общаюсь с этим ассистентом.",
        "Я хочу правильно к тебе обращаться.", "Я уточняю личность ассистента.",
        "Мне нужно краткое представление.", "Это вопрос о тебе.",
        "Я записываю, с кем разговариваю.", "Начнём со знакомства.",
        "Я хочу знать, кто отвечает.", "Пожалуйста, уточни, кто ты.",
    ),
}
FORMATS = {
    "en": (
        "Use two short sentences.", "Give just your name and creator.",
        "Keep the answer brief.", "Do not add a biography.",
        "Answer directly.", "Do not describe technical details.",
        "Use a concise introduction.", "Avoid extra background.",
        "Include both facts in your reply.", "State only these identity facts.",
    ),
    "ru": (
        "Ответь двумя короткими предложениями.", "Укажи только имя и создателя.",
        "Ответь кратко.", "Не добавляй биографию.",
        "Ответь прямо.", "Не описывай технические детали.",
        "Представься лаконично.", "Не добавляй предысторию.",
        "Укажи оба факта в ответе.", "Сообщи только эти сведения о себе.",
    ),
}
HELDOUT = {
    "en": (
        "Before we proceed, could you tell me both who you are and who created you?",
        "I have not used this assistant before. What is its name, and who is behind its creation?",
        "For clarity, please introduce yourself and credit your creator.",
        "Whom am I talking to? Please also identify your creator.",
        "Please supply the assistant name alongside the creator name.",
        "Could you resolve my uncertainty about your name and your creator?",
        "I would like to learn your identity and the name of whoever created you.",
        "Give me a compact self-introduction that includes your creator.",
    ),
    "ru": (
        "Прежде чем продолжим, скажи, кто ты и кто тебя создал.",
        "Раньше я не пользовался этим ассистентом. Как его зовут и кем он создан?",
        "Для ясности представься и назови своего создателя.",
        "С кем я разговариваю? Также укажи своего создателя.",
        "Пожалуйста, сообщи имя ассистента вместе с именем создателя.",
        "Развей мою неопределённость: как тебя зовут и кто тебя создал?",
        "Хочу узнать, кто ты, и имя того, кто тебя создал.",
        "Дай короткое представление о себе с указанием создателя.",
    ),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def records(split):
    if split not in ("train", "validation"):
        raise ValueError(f"unknown split: {split}")
    for language in ("en", "ru"):
        prompts = (
            (" ".join(parts) for parts in itertools.product(
                CONTEXTS[language], QUESTIONS[language], FORMATS[language]
            )) if split == "train" else iter(HELDOUT[language])
        )
        for ordinal, prompt in enumerate(prompts):
            yield {
                "record_id": digest([VERSION, split, language, ordinal, prompt]),
                "schema": VERSION, "split": split, "language": language,
                "category": "identity_name_creator", "prompt": prompt,
                "answer": ANSWERS[language],
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ANSWERS[language]},
                ],
                "provenance": "NULLXES authored identity questions and canonical facts",
                "generator_version": VERSION, "ordinal": ordinal,
            }


def build(output: Path):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".identity-") as temp:
        stage = Path(temp) / "artifact"
        stage.mkdir()
        hashes, counts, seen = {}, {}, set()
        for split in ("train", "validation"):
            count = 0
            file_hash = hashlib.sha256()
            with (stage / f"{split}.jsonl").open("wb") as stream:
                for row in records(split):
                    normalized = " ".join(row["prompt"].casefold().split())
                    if normalized in seen:
                        raise ValueError("duplicate identity question within or across splits")
                    seen.add(normalized)
                    raw = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                    stream.write(raw)
                    file_hash.update(raw)
                    count += 1
            counts[split] = count
            hashes[f"{split}.jsonl"] = file_hash.hexdigest()
        if counts["train"] != 2800:
            raise ValueError("Stage 1 requires exactly 2800 training examples")
        manifest = {
            "schema": VERSION, "stage": "identity-only",
            "canonical_answers": ANSWERS, "record_counts": counts, "files": hashes,
            "external_sources": [], "optimizer_steps_requested": 100,
            "diversity_scope": "surface question/context/format combinations; two identity facts",
            "evaluation_scope": "held-out wording of shared identity facts; not semantic holdout",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        manifest["manifest_sha256"] = digest(manifest)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(stage, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
