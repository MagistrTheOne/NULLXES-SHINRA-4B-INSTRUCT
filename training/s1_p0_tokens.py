"""Frozen P0 render. Same user-only template the V4 builder accounted."""

from __future__ import annotations

from pathlib import Path

BOS_ID = 1
EOT_ID = 2
PAD_ID = 3
VOCAB_SIZE = 131072
MAX_SEQUENCE = 8192
PARENT_REPO = "MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT"
PARENT_REVISION = "efae04115951d9473ebc2a90a2b2c684115408e7"
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "tokenizer" / "chat_template.jinja"


class TokenizerContractError(RuntimeError):
    pass


def assert_specials(bos: int, eot: int, pad: int, vocab: int) -> None:
    if (bos, eot, pad, vocab) != (BOS_ID, EOT_ID, PAD_ID, VOCAB_SIZE):
        raise TokenizerContractError(f"specials {(bos, eot, pad, vocab)}")


def labels_from_ids(prompt_ids: list[int], full_ids: list[int]) -> dict[str, list[int] | int]:
    if full_ids[: len(prompt_ids)] != list(prompt_ids):
        raise TokenizerContractError("prompt is not a prefix of the full sequence")
    if len(full_ids) > MAX_SEQUENCE:
        raise TokenizerContractError(f"sequence {len(full_ids)} > {MAX_SEQUENCE}")
    if not prompt_ids or prompt_ids[0] != BOS_ID:
        raise TokenizerContractError("missing leading BOS")
    if len(prompt_ids) >= 2 and prompt_ids[1] == BOS_ID:
        raise TokenizerContractError("duplicate BOS")
    if EOT_ID not in full_ids[len(prompt_ids) :]:
        raise TokenizerContractError("assistant tail has no EOT")
    labels = list(full_ids)
    for index in range(len(prompt_ids)):
        labels[index] = -100
    target = len(full_ids) - len(prompt_ids)
    if target <= 0:
        raise TokenizerContractError("empty supervised tail")
    return {
        "input_ids": list(full_ids),
        "labels": labels,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": target,
    }


def render_texts(template, user: str, answer: str) -> tuple[str, str]:
    prompt = template.render(
        messages=[{"role": "user", "content": user}],
        add_generation_prompt=True,
        bos_token="<|bos|>",
    )
    full = template.render(
        messages=[
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        add_generation_prompt=False,
        bos_token="<|bos|>",
    )
    if "<|system|>" in prompt or "<|system|>" in full:
        raise TokenizerContractError("system turn in a training example")
    return prompt, full


def encode_example(tokenizer, template, user: str, answer: str) -> dict[str, list[int] | int]:
    prompt_text, full_text = render_texts(template, user, answer)
    prompt_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False).ids)
    full_ids = list(tokenizer.encode(full_text, add_special_tokens=False).ids)
    return labels_from_ids(prompt_ids, full_ids)


def load_frozen_tokenizer():
    """Load the parent tokenizer. Not called by static tests."""
    from huggingface_hub import hf_hub_download
    from jinja2 import Environment
    from tokenizers import Tokenizer

    path = hf_hub_download(PARENT_REPO, "tokenizer.json", revision=PARENT_REVISION)
    tokenizer = Tokenizer.from_file(path)
    bos = tokenizer.token_to_id("<|bos|>")
    eot = tokenizer.token_to_id("<|eot|>")
    pad = tokenizer.token_to_id("<|pad|>")
    assert_specials(bos, eot, pad, tokenizer.get_vocab_size())
    template = Environment(autoescape=False).from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return tokenizer, template
