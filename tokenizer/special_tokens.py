"""Canonical special tokens for the NULLXES SHINRA tokenizer.

Frozen vocabulary control plane. Do not rename after pretrain.

ID layout (SentencePiece):
  0 <|unk|>
  1 <|bos|>
  2 <|eot|>            generation / turn stop (HF eos_token)
  3 <|pad|>
  4+ user-defined in USER_DEFINED_SYMBOLS order
"""

from __future__ import annotations

from pathlib import Path

UNK = "<|unk|>"
BOS = "<|bos|>"
EOT = "<|eot|>"
PAD = "<|pad|>"
EOS = EOT  # HF eos_token == end-of-turn. Document end is END_OF_TEXT.

END_OF_TEXT = "<|end_of_text|>"
DOCUMENT = "<|document|>"

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
REASONING = "<|reasoning|>"

CODE = "<|code|>"
LANGUAGE = "<|language|>"

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
REPO = "<|repo|>"
FILE = "<|file|>"
TOOL_CALL = "<|tool_call|>"
TOOL_RESPONSE = "<|tool_response|>"

CORE_SPECIALS = [UNK, BOS, EOT, PAD]
CHAT_SPECIALS = [SYSTEM, USER, ASSISTANT, REASONING]
CODE_SPECIALS = [CODE, LANGUAGE, FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]
REPO_SPECIALS = [REPO, FILE]
AGENT_SPECIALS = [TOOL_CALL, TOOL_RESPONSE]
META_SPECIALS = [DOCUMENT, END_OF_TEXT]

USER_DEFINED_SYMBOLS = (
    CHAT_SPECIALS + CODE_SPECIALS + REPO_SPECIALS + AGENT_SPECIALS + META_SPECIALS
)
ALL_SPECIAL_TOKENS = CORE_SPECIALS + USER_DEFINED_SYMBOLS

ROLE_TOKEN = {
    "system": SYSTEM,
    "user": USER,
    "assistant": ASSISTANT,
    "tool": TOOL_RESPONSE,
    "code": CODE,
    "reasoning": REASONING,
}

# Policy: SHINRA-4B-INSTRUCT keeps <|reasoning|> reserved.
# Public SFT does not emit visible chain-of-thought unless the example
# actually contains a reasoning field (internal distillation / alignment).
REASONING_POLICY = "latent"

_TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")
CHAT_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")

TOKENIZER_CONFIG = {
    "tokenizer_class": "PreTrainedTokenizerFast",
    "model_max_length": 32768,
    "bos_token": BOS,
    "eos_token": EOT,
    "unk_token": UNK,
    "pad_token": PAD,
    "add_bos_token": True,
    "add_eos_token": False,
    "clean_up_tokenization_spaces": False,
    "chat_template": CHAT_TEMPLATE,
    "extra_special_tokens": USER_DEFINED_SYMBOLS,
    "end_of_text_token": END_OF_TEXT,
    "document_token": DOCUMENT,
    "reasoning_policy": REASONING_POLICY,
}
