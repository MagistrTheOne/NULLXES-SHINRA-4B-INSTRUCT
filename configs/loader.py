"""Strict loader and validator for the single approved SHINRA configuration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).with_name("shinra.yaml")
REQUIRED_SECTIONS = {
    "schema_version", "status", "identity", "model", "tokenizer", "data",
    "training", "precision", "checkpoint", "evaluation", "runtime",
    "inference",
}


class ConfigError(ValueError):
    """Configuration is missing, inconsistent, or still contains unresolved choices."""


def load_config(path: str | Path = CONFIG_PATH, *, require_ready: bool = False) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config, require_ready=require_ready)
    return config


def validate_config(config: dict[str, Any], *, require_ready: bool = False) -> None:
    if not isinstance(config, dict):
        raise ConfigError("configuration root must be a mapping")
    missing = REQUIRED_SECTIONS - config.keys()
    if missing:
        raise ConfigError(f"missing configuration sections: {sorted(missing)}")
    identity, model, tokenizer = config["identity"], config["model"], config["tokenizer"]
    if identity.get("invariant") != "SHINRA is SHINRA. Created by NULLXES.":
        raise ConfigError("SHINRA identity invariant changed")
    if identity.get("initialization") != "random" or identity.get("historical_weights_allowed"):
        raise ConfigError("new base training must start from random initialization")
    model_keys = (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
        "bos_token_id", "eos_token_id", "pad_token_id", "document_end_token_id",
    )
    if any(key not in model for key in model_keys):
        raise ConfigError("model geometry or special token IDs are incomplete")
    if model["num_attention_heads"] * model["head_dim"] != 4096:
        raise ConfigError("approved SHINRA attention Q width must be 4096")
    for model_key, tokenizer_key in (("vocab_size", "vocab_size"), ("bos_token_id", "bos_id"),
                                      ("eos_token_id", "end_of_turn_id"), ("pad_token_id", "pad_id"),
                                      ("document_end_token_id", "end_of_document_id")):
        if model[model_key] != tokenizer[tokenizer_key]:
            raise ConfigError(f"model.{model_key} and tokenizer.{tokenizer_key} must match")
    specials = tokenizer.get("special_tokens", {})
    ids = [tokenizer.get(k) for k in ("unk_id", "bos_id", "end_of_turn_id", "pad_id", "end_of_document_id")]
    names = ("unk", "bos", "end_of_turn", "pad", "end_of_document")
    if any(not isinstance(specials.get(name), str) or not specials[name] for name in names):
        raise ConfigError("all core special token strings must be explicit")
    if len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 for i in ids):
        raise ConfigError("core special token IDs must be distinct non-negative integers")
    if ids != list(range(len(ids))):
        raise ConfigError("SentencePiece core token IDs must be ordered unk=0, bos=1, eot=2, pad=3, document_end=4")
    if specials["end_of_turn"] == specials["end_of_document"]:
        raise ConfigError("EOT and document-end token strings must differ")
    expected_specials = {"unk", "bos", "end_of_turn", "pad", "end_of_document", "system", "user", "assistant",
                         "reasoning", "code", "language", "fim_prefix", "fim_suffix", "fim_middle", "repo", "file",
                         "tool_call", "tool_response", "document"}
    if specials.keys() != expected_specials:
        raise ConfigError(f"special token set mismatch: missing={sorted(expected_specials - specials.keys())}, "
                          f"extra={sorted(specials.keys() - expected_specials)}")
    if len(set(specials.values())) != len(specials):
        raise ConfigError("special token strings must be unique")
    if tokenizer.get("normalization") != "identity" or config["data"].get("cleaning", {}).get("unicode_normalization") != "NFC":
        raise ConfigError("tokenizer must preserve normalized corpus text exactly (identity model normalization)")
    if tokenizer.get("byte_fallback") is not True:
        raise ConfigError("tokenizer byte fallback is required")
    if tokenizer.get("backend") not in {"sentencepiece_native", "sentencepiece_fast"}:
        raise ConfigError("tokenizer backend must be explicitly supported")
    native_sha = tokenizer.get("native_model_sha256")
    if tokenizer["backend"] == "sentencepiece_native" and (
        not isinstance(native_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", native_sha)
    ):
        raise ConfigError("Native SentencePiece model SHA256 must be fixed in configs/shinra.yaml")
    digest = tokenizer.get("artifact_sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)):
        raise ConfigError("tokenizer.artifact_sha256 must be the bundle manifest SHA256 or null")
    build = tokenizer.get("build", {})
    for key in ("max_chars", "input_sentence_size", "num_threads", "max_sentence_bytes"):
        if key not in build or (build[key] is not None and (type(build[key]) is not int or build[key] <= 0)):
            raise ConfigError(f"tokenizer.build.{key} must be a positive integer or null pending audit")
    if tokenizer.get("add_bos_in_tokenizer") is not False or tokenizer.get("chat_template_adds_bos") is not True:
        raise ConfigError("tokenizer must not auto-add BOS; chat template must add one BOS")
    if tokenizer.get("chat_turn_stop_id") != tokenizer.get("end_of_turn_id"):
        raise ConfigError("chat turn stop ID must match EOT")
    for offset, key in enumerate(("system", "user", "assistant", "reasoning", "code", "language", "fim_prefix",
                                  "fim_suffix", "fim_middle", "repo", "file", "tool_call", "tool_response",
                                  "document"), start=5):
        if tokenizer.get("special_token_ids", {}).get(key) != offset:
            raise ConfigError(f"tokenizer.special_token_ids.{key} must be {offset}")
    if model["document_end_token_id"] != tokenizer["end_of_document_id"]:
        raise ConfigError("model document_end_token_id must match tokenizer.end_of_document_id")
    packing = config["data"].get("packing", {})
    if packing.get("policy") != "isolated_documents" or packing.get("isolate_document_attention") is not True:
        raise ConfigError("the approved packing policy must isolate documents")
    if packing.get("document_end_token_id") != tokenizer["end_of_document_id"]:
        raise ConfigError("packing document end ID differs from the tokenizer contract")
    if packing.get("overlength_document_policy") != "non_overlapping_document_windows":
        raise ConfigError("packing must use the approved non-overlapping document-window policy")
    if packing.get("chunking_policy_version") != "isolated-continuation-windows.v1" \
            or packing.get("chunking_window_semantics") != "independent_continuation_windows_without_overlap":
        raise ConfigError("document chunking semantics/version are missing or unsupported")
    if packing.get("max_chunk_tokens") != model.get("train_sequence_length") \
            or packing.get("bos_per_window") is not True or packing.get("bos_loss") is not False \
            or packing.get("eod_only_on_final_document_window") is not True:
        raise ConfigError("chunk size and BOS/EOD window contracts are inconsistent")
    if type(packing.get("parquet_row_group_rows")) is not int or packing["parquet_row_group_rows"] <= 0:
        raise ConfigError("packed Parquet row-group size must be a positive integer")
    if packing.get("loss_on_bos") is not False or packing.get("loss_on_eod") is not True:
        raise ConfigError("BOS is context-only and EOD must be trained as a document-boundary target")
    if packing.get("loss_on_pad") is not False or packing.get("loss_on_first_token_after_pad") is not False:
        raise ConfigError("PAD and the first post-PAD target must be excluded from loss")
    if tokenizer["special_tokens"]["end_of_document"] not in {"<|end_of_text|>", "<|document_end|>"}:
        raise ConfigError("document termination token must remain a dedicated end-of-document control")
    sources = config["data"].get("sources", [])
    if config["data"].get("immutable_revision_required") is not True or not sources:
        raise ConfigError("source acquisition requires an exact-revision allowlist")
    for source in sources:
        if not isinstance(source.get("revision"), str) or not re.fullmatch(r"[a-f0-9]{40}", source["revision"]):
            raise ConfigError(f"source {source.get('id')!r} must pin an immutable 40-character revision")
        required_source_fields = ("repo_type", "parquet_prefixes", "text_field", "document_id_field",
                                  "license_claim", "cleaning_profile", "quarantine_profile", "upstream_url")
        if any(not source.get(key) for key in required_source_fields):
            raise ConfigError(f"source {source.get('id')!r} is missing adapter, license, or cleaning metadata")
        if source["repo_type"] != "dataset" or not isinstance(source["parquet_prefixes"], list) \
                or not source["parquet_prefixes"] or any(not isinstance(prefix, str) or not prefix
                                                          for prefix in source["parquet_prefixes"]):
            raise ConfigError(f"source {source.get('id')!r} must use a configured Hugging Face dataset adapter")
        if not isinstance(source.get("language_by_subset"), dict) or not source["language_by_subset"]:
            raise ConfigError(f"source {source.get('id')!r} must map every subset to EN or RU")
        if any(language not in {"en", "ru"} for language in source["language_by_subset"].values()):
            raise ConfigError(f"source {source.get('id')!r} contains a language outside EN/RU")
        if source["cleaning_profile"] not in {"wikimedia_rendered_text", "plain_text_nfc"} \
                or source["quarantine_profile"] not in {"table_and_markup", "ocr_book"}:
            raise ConfigError(f"source {source.get('id')!r} uses an unsupported cleaning/quarantine profile")
        if source.get("status") not in {"audit_only", "approved"}:
            raise ConfigError(f"source {source.get('id')!r} has an unsupported admission status")
        if source.get("rights_review_status") not in {"audit_only", "approved"} \
                or source.get("provenance_review_status") not in {"audit_only", "approved"}:
            raise ConfigError(f"source {source.get('id')!r} needs explicit rights and provenance review status")
        if source.get("status") == "approved" and (source["rights_review_status"] != "approved"
                or source["provenance_review_status"] != "approved"):
            raise ConfigError(f"source {source.get('id')!r} cannot be admitted without both reviews")
    if config["data"].get("training_enabled") and config["training"].get("enabled"):
        if not sources or any(s.get("status") != "approved" for s in sources):
            raise ConfigError("training requires rights- and provenance-approved sources with immutable revisions")
        if not tokenizer.get("artifact_sha256"):
            raise ConfigError("training requires a frozen tokenizer artifact SHA256")
        if not config["data"].get("manifest_sha256"):
            raise ConfigError("training requires a frozen corpus manifest SHA256")
        if not config["data"].get("split_salt"):
            raise ConfigError("training requires a frozen split salt")
    if config["training"].get("initialization") != "random":
        raise ConfigError("training initialization must be random")
    if config["training"].get("optimizer", {}).get("name") != "adamw":
        raise ConfigError("only AdamW is approved for the initial base run")
    if config["training"].get("scheduler", {}).get("name") != "warmup_cosine":
        raise ConfigError("scheduler must be warmup_cosine")
    if config["training"].get("sequence_length") != model.get("train_sequence_length"):
        raise ConfigError("training sequence length must match model train_sequence_length")
    if config["training"].get("global_tokens_per_update") != (
        config["training"].get("sequence_length", 0)
        * config["training"].get("micro_batch_size", 0)
        * config["training"].get("gradient_accumulation_steps", 0)
    ):
        raise ConfigError("global_tokens_per_update does not match sequence x micro batch x accumulation")
    if config["training"].get("diagnostic_tokens", 0) != 20_000_000:
        raise ConfigError("the initial diagnostic budget is fixed at 20M tokens")
    foundation_tokens = config["training"].get("foundation_tokens")
    if foundation_tokens is not None and (type(foundation_tokens) is not int or foundation_tokens <= 0):
        raise ConfigError("training.foundation_tokens must be a positive integer or null pending approval")
    if config["training"].get("budget_modes") != ["diagnostic", "foundation"]:
        raise ConfigError("diagnostic and foundation token budgets must remain distinct selectable modes")
    if config["training"].get("max_steps") is not None and (
        type(config["training"]["max_steps"]) is not int or config["training"]["max_steps"] <= 0
    ):
        raise ConfigError("training.max_steps must be a positive integer or null until approved")
    if config["training"].get("dataset_exhaustion_policy") != \
            "cycle_language_permutations_with_schedule_epoch_counter":
        raise ConfigError("training dataset exhaustion policy must be explicit")
    resume_fields = config["checkpoint"].get("resume_requires_matching", [])
    if "packed_data_manifests_sha256" not in resume_fields:
        raise ConfigError("checkpoint resume identity must bind all packed split manifest hashes")
    if config["data"].get("splits") != {"train": 0.98, "validation": 0.01, "blind": 0.01}:
        raise ConfigError("document split fractions must be frozen at train=0.98, validation=0.01, blind=0.01")
    cleaning = config["data"].get("cleaning", {})
    if not 0.0 < float(cleaning.get("near_dedup_threshold", 0)) <= 1.0:
        raise ConfigError("near-deduplication threshold must be in (0, 1]")
    if cleaning.get("near_dedup_metric") != "minhash_estimated_jaccard":
        raise ConfigError("near-deduplication similarity metric must be explicit and supported")
    if int(cleaning.get("near_dedup_num_perm", 0)) <= 0:
        raise ConfigError("near-deduplication num_perm must be positive")
    if int(cleaning.get("near_dedup_bands", 0)) <= 0 or int(cleaning["near_dedup_num_perm"]) % int(cleaning["near_dedup_bands"]):
        raise ConfigError("near-deduplication MinHash dimensions must divide evenly into disk-backed LSH bands")
    if type(cleaning.get("near_dedup_shingle_words")) is not int or cleaning["near_dedup_shingle_words"] <= 0:
        raise ConfigError("near-deduplication shingle width must be a positive integer")
    if type(config["data"].get("parquet_batch_rows")) is not int or config["data"]["parquet_batch_rows"] <= 0:
        raise ConfigError("streaming Parquet batch size must be a positive integer")
    mix = config["data"].get("mix", {})
    if mix.get("unit") != "actual_shinra_tokens" or abs(float(mix.get("en", 0)) + float(mix.get("ru", 0)) - 1.0) > 1e-12:
        raise ConfigError("language mix must define normalized actual-token weights")
    if float(mix.get("en", 0)) != 0.5 or float(mix.get("ru", 0)) != 0.5:
        raise ConfigError("the approved initial training mix is exactly 50/50 EN/RU supervised tokens")
    excluded = {str(source).casefold() for source in config["data"].get("excluded_sources", [])}
    for source in sources:
        source_id = source.get("id", "").casefold()
        if any(term.casefold() in source_id for term in excluded):
            raise ConfigError(f"excluded source appears in source list: {source['id']}")
    fsdp = config["precision"].get("fsdp", {})
    if fsdp.get("enabled") is not False or fsdp.get("implementation_status") != "unavailable_single_process":
        raise ConfigError("FSDP is not implemented by the current single-process trainer")
    if config["runtime"].get("colab_launch_allowed") and config.get("status") != "ready_for_training":
        raise ConfigError("Colab launch cannot be enabled before the configuration is ready")
    if config["training"].get("enabled") != config["data"].get("training_enabled"):
        raise ConfigError("training and data gates must be enabled or disabled together")
    if require_ready:
        if config.get("status") != "ready_for_training":
            raise ConfigError(f"configuration status is {config.get('status')!r}, not ready_for_training")
        if not config["training"].get("enabled") or not config["data"].get("training_enabled"):
            raise ConfigError("training/data gates are disabled")
        if not config["runtime"].get("colab_launch_allowed"):
            raise ConfigError("Colab launch is not approved in configs/shinra.yaml")
        if not config["tokenizer"].get("artifact_sha256") or not config["data"].get("manifest_sha256"):
            raise ConfigError("tokenizer and data manifests must be frozen before training")

