"""Frozen S1 DEV holdout. Indices sit far above any P0 train range."""

from __future__ import annotations

from data.s1.native import (
    controlled_record,
    instruction_record,
    paraphrase_record,
    relation_record,
    relation_reversal_pair,
    _native,
)

DEV_INDEX_BASE = 9_000_000
DEV_COUNTS = {
    "S1-01": (150, 150),
    "S1-02": (150, 150),
    "S1-03": (120, 120),
    "S1-04": (120, 120),
    "S1-05": (100, 100),
    "S1-06": (120, 120),
    "S1-07": (100, 100),
    "S1-08": (50, 50),
    "S1-09": (40, 40),
    "S1-10": (50, 50),
}

_DEV_ID_EN = (
    ("name", "Which name belongs to this language layer?", "SHINRA"),
    ("creator", "Who is the maker of this language layer?", "NULLXES"),
    ("distinction", "Does NULLXES name this language layer?", "No. I am SHINRA."),
    ("distinction", "Is the creator the same entity as this language layer?", "No. SHINRA is not NULLXES."),
    ("name", "What should the language layer be called?", "SHINRA"),
    ("creator", "Which party made this language layer?", "NULLXES"),
    ("name", "Give the language layer's name and nothing else.", "SHINRA"),
    ("creator", "Name only the maker of the language layer.", "NULLXES"),
)
_DEV_ID_RU = (
    ("name", "Какое имя у этого языкового слоя?", "SHINRA"),
    ("creator", "Кто является создателем этого языкового слоя?", "NULLXES"),
    ("distinction", "NULLXES — это имя языкового слоя?", "Нет. Я SHINRA."),
    ("distinction", "Создатель и языковой слой — одна сущность?", "Нет. SHINRA не является NULLXES."),
    ("name", "Как следует называть языковой слой?", "SHINRA"),
    ("creator", "Какая сторона сделала этот языковой слой?", "NULLXES"),
    ("name", "Назови имя языкового слоя и больше ничего.", "SHINRA"),
    ("creator", "Назови только создателя языкового слоя.", "NULLXES"),
)
_FRAMES_EN = (
    "Answer from the canonical identity facts.",
    "Use the canonical identity facts only.",
    "Stay with the canonical identity facts.",
    "Restrict the reply to the canonical identity facts.",
    "Keep to the canonical identity facts.",
    "Reply using the canonical identity facts.",
    "Produce only the canonical identity facts.",
)
_FRAMES_RU = (
    "Ответь по каноническим фактам об имени.",
    "Используй только канонические факты об имени.",
    "Держись канонических фактов об имени.",
    "Ограничь ответ каноническими фактами об имени.",
    "Оставь только канонические факты об имени.",
    "Дай ответ по каноническим фактам об имени.",
    "Сформулируй только канонические факты об имени.",
)


def _difficulty(index: int, forced: str | None = None) -> str:
    if forced:
        return forced
    return ("A", "B", "C", "D", "E")[index % 5]


def _dev_identity(index: int, language: str) -> dict:
    bank = _DEV_ID_EN if language == "en" else _DEV_ID_RU
    frames = _FRAMES_EN if language == "en" else _FRAMES_RU
    fact, question, answer = bank[index % len(bank)]
    frame = frames[(index // len(bank)) % len(frames)]
    forced = "D" if fact == "distinction" else None
    return _native(
        source_id="native_identity",
        row_id=f"dev-{language}-id-{index}",
        language=language,
        family="S1-10",
        difficulty=_difficulty(index, forced),
        context=frame,
        question=question,
        answer=answer,
        instruction="Canonical name, creator, or distinction only." if language == "en" else "Только каноническое имя, создатель или различие.",
        transform="native_dev.identity.v1",
        extra={"latent_fact": fact, "split_role": "dev"},
    )


def _dev_public_family(family: str, index: int, language: str) -> dict:
    slot = DEV_INDEX_BASE + index
    level = _difficulty(index)
    if family == "S1-03":
        if language == "en":
            context = f"Card P-{slot}. The blue bin is empty. The red bin holds pears."
            if index % 2 == 0:
                question, answer, task = "Is the blue bin empty?", "yes", "explicit_negation_false"
            else:
                question, answer, task = "Does the blue bin hold pears?", "no", "hard_negative"
                level = "D"
        else:
            context = f"Карточка P-{slot}. Синий ящик пуст. В красном ящике груши."
            if index % 2 == 0:
                question, answer, task = "Синий ящик пуст?", "да", "explicit_negation_false"
            else:
                question, answer, task = "В синем ящике лежат груши?", "нет", "hard_negative"
                level = "D"
    elif family == "S1-05":
        if language == "en":
            context = f"Card C-{slot}. Mira left the umbrella. It was still wet."
            question, answer, task = "What does 'it' refer to?", "the umbrella", "coreference"
        else:
            context = f"Карточка C-{slot}. Мира оставила зонт. Он ещё был мокрым."
            question, answer, task = "К чему относится «он»?", "к зонту", "coreference"
        level = "E"
    elif family == "S1-06":
        if language == "en":
            context = f"Card E-{slot}. The red bin holds pears. The blue bin is empty."
            if index % 2 == 0:
                question, answer, task = "Does the note say the red bin holds pears?", "yes", "entailment"
            else:
                question, answer, task = "Does the note say the blue bin holds pears?", "no", "contradiction"
                level = "D"
        else:
            context = f"Карточка E-{slot}. В красном ящике груши. Синий ящик пуст."
            if index % 2 == 0:
                question, answer, task = "Из записки следует, что в красном ящике груши?", "да", "entailment"
            else:
                question, answer, task = "Из записки следует, что в синем ящике груши?", "нет", "contradiction"
                level = "D"
    elif family == "S1-07":
        if language == "en":
            context = f"Card X-{slot}. Ticket 77 was sold at Cedar Wharf on Monday."
            question, answer, task = "What is the ticket number?", "77", "extraction"
            level = "B"
        else:
            context = f"Карточка X-{slot}. Билет 77 продали на пристани Клён в понедельник."
            question, answer, task = "Какой номер билета?", "77", "extraction"
            level = "B"
    else:
        if language == "en":
            context = f"Card Q-{slot}. The harbor office opens at dawn."
            question, answer, task = "When does the harbor office open?", "at dawn", "short_qa"
            level = "A"
        else:
            context = f"Карточка Q-{slot}. Портовая контора открывается на рассвете."
            question, answer, task = "Когда открывается портовая контора?", "на рассвете", "short_qa"
            level = "A"
    return _native(
        source_id="native_dev",
        row_id=f"dev-{language}-{family}-{index}",
        language=language,
        family=family,
        difficulty=level,
        context=context,
        question=question,
        answer=answer,
        instruction="Answer from the card only." if language == "en" else "Ответь только по карточке.",
        transform="native_dev.v1",
        extra={"latent_task": task, "split_role": "dev"},
    )


def generate_dev() -> list[dict]:
    rows: list[dict] = []
    for language, count in zip(("en", "ru"), DEV_COUNTS["S1-01"]):
        for index in range(count):
            rows.append(instruction_record(DEV_INDEX_BASE + index, language, _difficulty(index)))
    for language, count in zip(("en", "ru"), DEV_COUNTS["S1-02"]):
        pairs = 20
        for index in range(pairs):
            forward, backward = relation_reversal_pair(DEV_INDEX_BASE + index, language)
            rows.extend((forward, backward))
        for index in range(count - pairs * 2):
            rows.append(relation_record(DEV_INDEX_BASE + index, language, _difficulty(index)))
    for language, count in zip(("en", "ru"), DEV_COUNTS["S1-04"]):
        for index in range(count):
            raw = DEV_INDEX_BASE + index * 8 + (5 if index % 2 else 0)
            rows.append(paraphrase_record(raw, language, "D" if index % 2 else _difficulty(index)))
    for language, count in zip(("en", "ru"), DEV_COUNTS["S1-09"]):
        for index in range(count):
            rows.append(controlled_record(DEV_INDEX_BASE + index, language, _difficulty(index)))
    for language, count in zip(("en", "ru"), DEV_COUNTS["S1-10"]):
        for index in range(count):
            rows.append(_dev_identity(index, language))
    for family in ("S1-03", "S1-05", "S1-06", "S1-07", "S1-08"):
        for language, count in zip(("en", "ru"), DEV_COUNTS[family]):
            for index in range(count):
                rows.append(_dev_public_family(family, index, language))
    for row in rows:
        row["metadata"]["split_role"] = "dev"
    return rows
