"""Load the explicitly configured SHINRA tokenizer backend."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from configs.loader import load_config

if TYPE_CHECKING:
    from .native import ShinraNativeTokenizer
    from transformers import PreTrainedTokenizerFast


def _expected_specials(token_config: dict) -> dict[str, int]:
    expected = {token_config["special_tokens"][key]: int(value)
                for key, value in token_config["special_token_ids"].items()}
    expected.update({token_config["special_tokens"][key]: int(token_config[f"{key}_id"])
                     for key in ("unk", "bos", "end_of_turn", "pad", "end_of_document")})
    return expected


def _validate_common(tokenizer, expected: dict[str, int], approved: dict) -> None:
    token_config = approved["tokenizer"]
    if len(tokenizer) != int(token_config["vocab_size"]):
        raise ValueError("tokenizer vocabulary size differs from configs/shinra.yaml")
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in expected}
    if actual != expected:
        raise ValueError(f"tokenizer special token IDs mismatch: {actual} != {expected}")
    if tokenizer.eos_token_id != int(token_config["end_of_turn_id"]):
        raise ValueError("tokenizer eos_token_id must remain the configured EOT ID")
    if set(tokenizer.all_special_ids) != set(expected.values()):
        raise ValueError("all configured special token IDs must be registered")
    from .special_tokens import render_chat_template
    if tokenizer.chat_template != render_chat_template(approved):
        raise ValueError("tokenizer chat template differs from the current contract")
    if tokenizer.model_max_length != int(approved["model"]["max_position_embeddings"]):
        raise ValueError("tokenizer context limit differs from configs/shinra.yaml")


def _register_native_special_tokens(tokenizer, token_config: dict) -> dict[str, int]:
    strings = token_config["special_tokens"]
    initial_size = len(tokenizer)
    expected = _expected_specials(token_config)
    added_core = tokenizer.add_special_tokens({
        "unk_token": strings["unk"], "bos_token": strings["bos"],
        "eos_token": strings["end_of_turn"], "pad_token": strings["pad"],
    })
    if added_core != 0 or len(tokenizer) != initial_size:
        raise ValueError("registering core SentencePiece specials changed the vocabulary")
    additional = [token for token, index in sorted(expected.items(), key=lambda pair: pair[1]) if index >= 4]
    added_controls = tokenizer.add_special_tokens({"additional_special_tokens": additional})
    if added_controls != 0 or len(tokenizer) != initial_size:
        raise ValueError("registering SentencePiece controls changed the vocabulary")
    return expected


def _load_native_tokenizer(path: Path, approved: dict, *, require_frozen_bundle: bool):
    from .native import ShinraNativeTokenizer
    from .special_tokens import render_chat_template

    tokenizer = ShinraNativeTokenizer(str(path / "tokenizer.model"), config=approved,
                                     require_frozen_bundle=require_frozen_bundle)
    bundle_sha = tokenizer.shinra_bundle_sha256
    token_config = approved["tokenizer"]
    expected = _register_native_special_tokens(tokenizer, token_config)
    tokenizer.chat_template = render_chat_template(approved)
    _validate_common(tokenizer, expected, approved)

    sp = tokenizer.sp_model
    native_ids = {"unk": int(sp.unk_id()), "bos": int(sp.bos_id()),
                  "eot": int(sp.eos_id()), "pad": int(sp.pad_id())}
    expected_ids = {"unk": int(token_config["unk_id"]), "bos": int(token_config["bos_id"]),
                    "eot": int(token_config["end_of_turn_id"]), "pad": int(token_config["pad_id"])}
    if native_ids != expected_ids:
        raise ValueError(f"native SentencePiece meta IDs mismatch: {native_ids} != {expected_ids}")
    for token, token_id in expected.items():
        if sp.piece_to_id(token) != token_id or sp.id_to_piece(token_id) != token:
            raise ValueError(f"SentencePiece model special token mismatch: {token!r} -> {token_id}")
    if tokenizer.implicit_bos is not False or tokenizer.truncation is not None or tokenizer.post_processor is not None:
        raise ValueError("Native tokenizer has BOS/truncation/post-processing outside the contract")
    sample = "Native BOS contract"
    if tokenizer.encode(sample, add_special_tokens=True) != tokenizer.encode(sample, add_special_tokens=False):
        raise ValueError("Native encode added an implicit BOS or other special token")

    tokenizer.shinra_bundle_path = path.resolve()
    tokenizer.shinra_bundle_sha256 = bundle_sha
    tokenizer.shinra_config = approved
    return tokenizer


def _load_fast_tokenizer(path: Path, approved: dict, *, require_frozen_bundle: bool):
    from .bundle import verify_bundle
    from transformers import AutoTokenizer

    bundle_sha = verify_bundle(path, approved, backend="sentencepiece_fast",
                               require_frozen=require_frozen_bundle)
    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    expected = _expected_specials(approved["tokenizer"])
    _validate_common(tokenizer, expected, approved)
    if tokenizer.backend_tokenizer.post_processor is not None:
        raise ValueError("tokenizer post-processing is forbidden: BOS is inserted explicitly")
    if tokenizer.backend_tokenizer.truncation is not None:
        raise ValueError("implicit tokenizer truncation is forbidden")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.shinra_bundle_path = path.resolve()
    tokenizer.shinra_bundle_sha256 = bundle_sha
    tokenizer.shinra_config = approved
    return tokenizer


def load_shinra_tokenizer(
    path: str | Path, config: dict | None = None, *, require_frozen_bundle: bool = True
) -> ShinraNativeTokenizer | PreTrainedTokenizerFast:
    """Load exactly the configured backend. Backend failures do not trigger fallback."""
    approved = config or load_config()
    path = Path(path)
    backend = approved["tokenizer"].get("backend", "sentencepiece_fast")
    if backend == "sentencepiece_native":
        return _load_native_tokenizer(path, approved, require_frozen_bundle=require_frozen_bundle)
    if backend == "sentencepiece_fast":
        return _load_fast_tokenizer(path, approved, require_frozen_bundle=require_frozen_bundle)
    raise ValueError(f"unsupported tokenizer backend in configs/shinra.yaml: {backend}")
