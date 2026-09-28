"""CPU contamination gate for synthetic pretrain text."""

from __future__ import annotations

from tokenizer.special_tokens import PRETRAIN_FORBIDDEN_STRINGS

IDENTITY_NEEDLES = (
    "i am shinra",
    "i'm shinra",
    "я шинра",
    "я — шинра",
    "я - шинра",
    "меня зовут shinra",
    "my name is shinra",
    "nullxes-shinra-4b-instruct",
    "nullxes shinra-4b",
    "created by nullxes",
)

INSTRUCTION_NEEDLES = (
    "### instruction",
    "### response",
    "user:",
    "assistant:",
    "system:",
    "human:",
    "вопрос:",
    "ответ:",
    "пользователь:",
    "ассистент:",
)


class ContaminationError(ValueError):
    pass


def assert_clean_pretrain_text(text: str) -> None:
    if "<|" in text:
        raise ContaminationError("special-token markup in text")
    lowered = text.lower()
    for needle in IDENTITY_NEEDLES:
        if needle in lowered:
            raise ContaminationError(f"identity needle: {needle}")
    for needle in INSTRUCTION_NEEDLES:
        if needle in lowered:
            raise ContaminationError(f"instruction/chat needle: {needle}")
    for tok in PRETRAIN_FORBIDDEN_STRINGS:
        if tok in text:
            raise ContaminationError(f"forbidden special string: {tok}")
