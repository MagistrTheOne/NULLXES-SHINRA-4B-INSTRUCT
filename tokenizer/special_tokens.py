"""Canonical special tokens for the NULLXES SHINRA tokenizer.

Frozen vocabulary control plane. Do not rename after pretrain.

ID layout (SentencePiece):
  0 <|unk|>
  1 <|bos|>
  2 <|eot|>            generation / turn stop (HF eos_token)
  3 <|pad|>
  4 <|end_of_text|>     document terminator
  5-18 configured user-defined control symbols

SentencePiece internal normalization must preserve Unicode and whitespace; the artifact is only approved after exact decode/encode checks on real EN/RU data.
"""

from __future__ import annotations

from pathlib import Path

from configs.loader import load_config

_SHINRA_CONFIG = load_config()
_TOKENIZER_CONFIG = _SHINRA_CONFIG["tokenizer"]

UNK = _TOKENIZER_CONFIG["special_tokens"]["unk"]
BOS = _TOKENIZER_CONFIG["special_tokens"]["bos"]
EOT = _TOKENIZER_CONFIG["special_tokens"]["end_of_turn"]
PAD = _TOKENIZER_CONFIG["special_tokens"]["pad"]
EOS = EOT  # HF eos_token == end-of-turn. Document end is END_OF_TEXT.

END_OF_TEXT = _TOKENIZER_CONFIG["special_tokens"]["end_of_document"]
DOCUMENT = _TOKENIZER_CONFIG["special_tokens"]["document"]

SYSTEM = _TOKENIZER_CONFIG["special_tokens"]["system"]
USER = _TOKENIZER_CONFIG["special_tokens"]["user"]
ASSISTANT = _TOKENIZER_CONFIG["special_tokens"]["assistant"]
REASONING = _TOKENIZER_CONFIG["special_tokens"]["reasoning"]

CODE = _TOKENIZER_CONFIG["special_tokens"]["code"]
LANGUAGE = _TOKENIZER_CONFIG["special_tokens"]["language"]

FIM_PREFIX = _TOKENIZER_CONFIG["special_tokens"]["fim_prefix"]
FIM_SUFFIX = _TOKENIZER_CONFIG["special_tokens"]["fim_suffix"]
FIM_MIDDLE = _TOKENIZER_CONFIG["special_tokens"]["fim_middle"]
REPO = _TOKENIZER_CONFIG["special_tokens"]["repo"]
FILE = _TOKENIZER_CONFIG["special_tokens"]["file"]
TOOL_CALL = _TOKENIZER_CONFIG["special_tokens"]["tool_call"]
TOOL_RESPONSE = _TOKENIZER_CONFIG["special_tokens"]["tool_response"]

CORE_SPECIALS = [UNK, BOS, EOT, PAD]
CHAT_SPECIALS = [SYSTEM, USER, ASSISTANT, REASONING]
CODE_SPECIALS = [CODE, LANGUAGE, FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]
REPO_SPECIALS = [REPO, FILE]
AGENT_SPECIALS = [TOOL_CALL, TOOL_RESPONSE]
META_SPECIALS = [DOCUMENT, END_OF_TEXT]

USER_DEFINED_SYMBOLS = tuple(v for k, v in _TOKENIZER_CONFIG["special_tokens"].items()
                             if k not in {"unk", "bos", "end_of_turn", "pad", "end_of_document"})
ALL_SPECIAL_TOKENS = CORE_SPECIALS + USER_DEFINED_SYMBOLS

# Frozen SentencePiece IDs. Pretrain pack may emit 1 (BOS), 3 (PAD), 4 (END_OF_TEXT).
PRETRAIN_BOS_ID = int(_TOKENIZER_CONFIG["bos_id"])
PRETRAIN_EOT_ID = int(_TOKENIZER_CONFIG["end_of_turn_id"])
PRETRAIN_PAD_ID = int(_TOKENIZER_CONFIG["pad_id"])
PRETRAIN_END_OF_TEXT_ID = int(_TOKENIZER_CONFIG["end_of_document_id"])
PRETRAIN_FORBIDDEN_IDS = frozenset({PRETRAIN_EOT_ID, *[int(v) for v in _TOKENIZER_CONFIG["special_token_ids"].values()
                                                       if int(v) != PRETRAIN_END_OF_TEXT_ID]})
PRETRAIN_BODY_FORBIDDEN_IDS = frozenset({PRETRAIN_BOS_ID, PRETRAIN_EOT_ID, PRETRAIN_PAD_ID, PRETRAIN_END_OF_TEXT_ID,
                                          *[int(v) for v in _TOKENIZER_CONFIG["special_token_ids"].values()]})
PRETRAIN_FORBIDDEN_STRINGS = tuple(
    tok for tok in ALL_SPECIAL_TOKENS if tok not in {UNK, BOS, PAD, END_OF_TEXT}
)

ROLE_TOKEN = {
    "system": SYSTEM,
    "user": USER,
    "assistant": ASSISTANT,
    "tool": TOOL_RESPONSE,
    "code": CODE,
    "reasoning": REASONING,
}

# Policy: the base pretraining corpus does not use chat-role or reasoning controls.
# Future instruction tuning policy is not implemented yet.
REASONING_POLICY = "latent"

_TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")
CHAT_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")
for _name, _token in _TOKENIZER_CONFIG["special_tokens"].items():
    CHAT_TEMPLATE = CHAT_TEMPLATE.replace(f"@@{_name}@@", _token)

TOKENIZER_CONFIG = {
    "tokenizer_class": "PreTrainedTokenizerFast",
    "model_max_length": 32768,
    "bos_token": BOS,
    "eos_token": EOT,
    "unk_token": UNK,
    "pad_token": PAD,
    "add_bos_token": False,
    "add_eos_token": False,
    "clean_up_tokenization_spaces": False,
    "chat_template": CHAT_TEMPLATE,
    "extra_special_tokens": USER_DEFINED_SYMBOLS,
    "end_of_text_token": END_OF_TEXT,
    "document_token": DOCUMENT,
    "reasoning_policy": REASONING_POLICY,
}
