"""SentencePiece-native slow tokenizer for SHINRA.

This class talks directly to SentencePieceProcessor. It has no Fast tokenizer
backend and never constructs a Transformers AutoTokenizer.
"""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer


class ShinraNativeTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]
    is_fast = False

    def __init__(self, vocab_file: str, *, config: dict, **kwargs: Any) -> None:
        from .bundle import verify_bundle
        import sentencepiece as spm

        self.vocab_file = str(Path(vocab_file).resolve())
        if Path(self.vocab_file).name != "tokenizer.model":
            raise ValueError("Native tokenizer must load tokenizer.model from its verified bundle root")
        self.shinra_bundle_path = Path(self.vocab_file).parent
        self.shinra_bundle_sha256 = verify_bundle(
            self.shinra_bundle_path, config, backend="sentencepiece_native"
        )
        self._shinra_config = config
        self.sp_model = spm.SentencePieceProcessor(model_file=self.vocab_file)
        self._validate_model_contract()
        from .vocab import validate_sentencepiece_vocab
        validate_sentencepiece_vocab(
            Path(self.vocab_file).with_suffix(".vocab"), self.sp_model,
            expected_vocab_size=int(config["tokenizer"]["vocab_size"]),
        )
        self._native_special_strings: dict[int, str] = {}
        super().__init__(
            model_max_length=int(config["model"]["max_position_embeddings"]),
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
        self.chat_template = None
        self.padding_side = "right"
        self.truncation_side = "right"
        self.implicit_bos = False
        self.truncation = None
        self.post_processor = None

    def _validate_model_contract(self) -> None:
        from sentencepiece import sentencepiece_model_pb2

        model = sentencepiece_model_pb2.ModelProto()
        model.ParseFromString(self.sp_model.serialized_model_proto())
        trainer = model.trainer_spec
        normalizer = model.normalizer_spec
        tok = self._shinra_config["tokenizer"]
        expected_ids = {"unk_id": tok["unk_id"], "bos_id": tok["bos_id"],
                        "eos_id": tok["end_of_turn_id"], "pad_id": tok["pad_id"]}
        actual_ids = {name: int(getattr(trainer, name)) for name in expected_ids}
        if (int(trainer.model_type) != 1  # SentencePiece TrainerSpec.ModelType.UNIGRAM
                or int(trainer.vocab_size) != int(tok["vocab_size"])
                or actual_ids != expected_ids
                or bool(trainer.byte_fallback) is not bool(tok["byte_fallback"])
                or bool(trainer.split_digits) is not bool(tok["split_digits"])
                or normalizer.name != tok["normalization"]):
            raise ValueError("SentencePiece model proto differs from the Native tokenizer contract")

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.get_piece_size())

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return {self.sp_model.id_to_piece(index): index for index in range(self.vocab_size)}

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        del kwargs
        return list(self.sp_model.encode(text, out_type=str))

    def _convert_token_to_id(self, token: str) -> int:
        return int(self.sp_model.piece_to_id(token))

    def _convert_id_to_token(self, index: int) -> str:
        if not 0 <= int(index) < self.vocab_size:
            raise ValueError(f"token ID outside SentencePiece vocabulary: {index}")
        return str(self.sp_model.id_to_piece(int(index)))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return str(self.sp_model.decode_pieces(tokens))

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        # BOS belongs to the explicit chat formatting contract, never to encode().
        return list(token_ids_0) + (list(token_ids_1) if token_ids_1 is not None else [])

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        del already_has_special_tokens
        return [0] * (len(token_ids_0) + (len(token_ids_1) if token_ids_1 else 0))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        return [0] * (len(token_ids_0) + (len(token_ids_1) if token_ids_1 else 0))

    def encode(
        self, text: str, text_pair: str | None = None, *, truncation: bool = False, **kwargs: Any
    ) -> list[int]:
        if truncation:
            raise ValueError("SHINRA Native tokenizer never truncates; shorten or reject input explicitly")
        return super().encode(text, text_pair=text_pair, truncation=False, **kwargs)

    def __call__(self, text: Any = None, text_pair: Any = None, *, truncation: bool = False, **kwargs: Any):
        if truncation:
            raise ValueError("SHINRA Native tokenizer never truncates; shorten or reject input explicitly")
        return super().__call__(text, text_pair=text_pair, truncation=False, **kwargs)

    def add_special_tokens(self, special_tokens_dict: dict, *args: Any, **kwargs: Any) -> int:
        if len(args) > 1:
            raise TypeError("add_special_tokens accepts at most one replacement option")
        positional_replace = args[0] if args else None
        legacy_replace = kwargs.pop("replace_additional_special_tokens", None)
        current_replace = kwargs.pop("replace_extra_special_tokens", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected add_special_tokens keyword argument(s): {unexpected}")
        requested_values = [value for value in (positional_replace, legacy_replace, current_replace)
                            if value is not None]
        if any(type(value) is not bool for value in requested_values):
            raise TypeError("special-token replacement option must be a bool")
        if requested_values and any(value != requested_values[0] for value in requested_values[1:]):
            raise TypeError("conflicting special-token replacement options were provided")
        replace_special_tokens = requested_values[0] if requested_values else True

        before = self.vocab_size
        base_add_special_tokens = super().add_special_tokens
        parameters = inspect.signature(base_add_special_tokens).parameters
        if "replace_extra_special_tokens" in parameters:
            replacement_parameter = "replace_extra_special_tokens"
            native_list_key, legacy_list_key = "extra_special_tokens", "additional_special_tokens"
        elif "replace_additional_special_tokens" in parameters:
            replacement_parameter = "replace_additional_special_tokens"
            native_list_key, legacy_list_key = "additional_special_tokens", "extra_special_tokens"
        else:
            raise RuntimeError(
                "unsupported Transformers add_special_tokens API: expected replace_extra_special_tokens "
                "or replace_additional_special_tokens"
            )

        normalized_special_tokens = dict(special_tokens_dict)
        if native_list_key in normalized_special_tokens and legacy_list_key in normalized_special_tokens:
            raise ValueError("provide only one additional/extra special-token list")
        if legacy_list_key in normalized_special_tokens:
            normalized_special_tokens[native_list_key] = normalized_special_tokens.pop(legacy_list_key)
        added = base_add_special_tokens(
            normalized_special_tokens, **{replacement_parameter: replace_special_tokens}
        )
        if added or self.vocab_size != before or len(self) != before:
            raise ValueError("configured specials must resolve to existing SentencePiece IDs; vocabulary growth is forbidden")
        if replace_special_tokens:
            self._native_special_strings.clear()
        additional_tokens = normalized_special_tokens.get(native_list_key, [])
        self._native_special_strings.update({
            int(self.convert_tokens_to_ids(token)): token for token in additional_tokens
        })
        for field in ("unk_token", "bos_token", "eos_token", "pad_token"):
            token = special_tokens_dict.get(field)
            if token is not None:
                token_string = str(token)
                self._native_special_strings[int(self.convert_tokens_to_ids(token_string))] = token_string
        return added

    def decode(
        self, token_ids: Any, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False, **kwargs: Any
    ) -> str:
        del clean_up_tokenization_spaces, kwargs
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("decode accepts one sequence; use batch_decode for batches")
            token_ids = token_ids[0]
        special_map = {int(index): str(token) for index, token in self._native_special_strings.items()}
        special_map.update({
            int(self.unk_token_id): str(self.unk_token),
            int(self.bos_token_id): str(self.bos_token),
            int(self.eos_token_id): str(self.eos_token),
            int(self.pad_token_id): str(self.pad_token),
        })
        special_ids = set(int(index) for index in self.all_special_ids)
        chunks: list[str] = []
        ordinary: list[int] = []

        def flush() -> None:
            if ordinary:
                chunks.append(str(self.sp_model.decode([int(index) for index in ordinary])))
                ordinary.clear()

        for value in token_ids:
            index = int(value)
            if not 0 <= index < self.vocab_size:
                raise ValueError(f"token ID outside SentencePiece vocabulary: {index}")
            if index in special_ids:
                flush()
                if not skip_special_tokens:
                    chunks.append(special_map.get(index, self._convert_id_to_token(index)))
            else:
                ordinary.append(index)
        flush()
        return "".join(chunks)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        source = Path(self.vocab_file)
        model_target = destination / f"{prefix}tokenizer.model"
        if source.resolve() != model_target.resolve():
            shutil.copy2(source, model_target)
        vocab_source = source.with_suffix(".vocab")
        vocab_target = destination / f"{prefix}tokenizer.vocab"
        if not vocab_source.is_file():
            raise FileNotFoundError(f"SentencePiece vocabulary file is required: {vocab_source}")
        if vocab_source.resolve() != vocab_target.resolve():
            shutil.copy2(vocab_source, vocab_target)
        return str(model_target), str(vocab_target)
