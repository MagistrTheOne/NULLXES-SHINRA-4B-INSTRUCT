"""Strict loader and validator for the single approved SHINRA configuration."""

from __future__ import annotations

import hashlib
import json
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
    try:
        requested_path = config_path.resolve(strict=True)
        approved_path = CONFIG_PATH.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"cannot resolve the canonical SHINRA config path: {config_path}") from exc
    if requested_path != approved_path:
        raise ConfigError(
            f"SHINRA has one authoritative configuration: {approved_path}; alternate configs are not supported"
        )
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
    if identity.get("invariant") != "I AM SHINRA. CREATED BY NULLXES.":
        raise ConfigError("SHINRA identity invariant changed")
    contract_path = (CONFIG_PATH.parent.parent / str(identity.get("contract_path", ""))).resolve()
    if not contract_path.is_file() or hashlib.sha256(contract_path.read_bytes()).hexdigest() != identity.get("contract_sha256"):
        raise ConfigError("versioned SHINRA identity contract is missing or its SHA256 differs from configs/shinra.yaml")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("versioned SHINRA identity contract is not valid UTF-8 JSON") from exc
    if contract.get("contract_id") != "nullxes.shinra.identity.v1" or contract.get("name") != "SHINRA" \
            or contract.get("creator") != "NULLXES" or contract.get("canonical_identity") != "I AM SHINRA. CREATED BY NULLXES." \
            or contract.get("type") != "Native artificial intelligence model" \
            or contract.get("languages") != ["English", "Russian"] \
            or contract.get("behavior") != ["Direct", "accurate", "instruction-following", "safety-aware"] \
            or contract.get("response_policy") != {"identity_when_asked": True,
                "identity_unprompted": False, "claims_consciousness": False,
                "invent_private_information": False} \
            or "origin" in contract or "initialization" in contract:
        raise ConfigError("identity contract differs from the approved SHINRA identity facts")
    schedule = identity.get("schedule", {})
    if schedule.get("status") != "specification_only" or schedule.get("initial_optimizer_steps") != 100 \
            or schedule.get("identity_examples_per_step") != 28 or schedule.get("identity_train_examples") != 2800:
        raise ConfigError("identity schedule must remain a specification-only 2800-example/100-step plan")
    model_keys = (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
        "bos_token_id", "eos_token_id", "pad_token_id", "document_end_token_id",
    )
    if any(key not in model for key in model_keys):
        raise ConfigError("model geometry or special token IDs are incomplete")
    if model["num_attention_heads"] * model["head_dim"] != 4096:
        raise ConfigError("approved SHINRA attention Q width must be 4096")
    if model.get("attention_implementation") != "sdpa":
        raise ConfigError("configs/shinra.yaml admits SDPA only until alternate attention passes parity review")
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
    if tokenizer.get("backend") != "sentencepiece_native":
        raise ConfigError("SHINRA Native release requires sentencepiece_native; Fast conversion is not admitted")
    native_sha = tokenizer.get("native_model_sha256")
    if native_sha is not None and (
        not isinstance(native_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", native_sha)
    ):
        raise ConfigError("tokenizer.native_model_sha256 must be a SHA256 or null pending candidate review")
    digest = tokenizer.get("artifact_sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)):
        raise ConfigError("tokenizer.artifact_sha256 must be the bundle manifest SHA256 or null")
    corpus_digest = config["data"].get("manifest_sha256")
    if corpus_digest is not None and (
        not isinstance(corpus_digest, str) or not re.fullmatch(r"[a-f0-9]{64}", corpus_digest)
    ):
        raise ConfigError("data.manifest_sha256 must be a 64-character SHA256 or null")
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
    if type(model.get("loss_chunk_size")) is not int or model["loss_chunk_size"] <= 0:
        raise ConfigError("model.loss_chunk_size must be a positive logit time chunk size")
    loss_memory_status = model.get("loss_memory_status")
    if loss_memory_status not in {"code_candidate_not_runtime_verified", "runtime_verified"}:
        raise ConfigError("model.loss_memory_status must identify the candidate or remotely verified implementation")
    if config["training"].get("enabled") and loss_memory_status != "runtime_verified":
        raise ConfigError("training cannot be enabled until chunked loss memory and gradients are runtime verified")
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
    data = config["data"]
    synthetic = data.get("synthetic", {})
    if data.get("acquisition") != "nullxes_synthetic_engine" or synthetic.get("enabled") is not True:
        raise ConfigError("active corpus acquisition must use the NULLXES Synthetic Engine")
    if data.get("sources") or data.get("target_build") or data.get("excluded_sources"):
        raise ConfigError("external corpus sources/caps must not appear in active data configuration")
    if data.get("corpus_manifest_required") is not True or data.get("source_manifest_required") is not False:
        raise ConfigError("synthetic data contract requires a corpus manifest, not an upstream source manifest")
    if synthetic.get("external_text_sources_allowed") is not False \
            or synthetic.get("external_llm_generation_allowed") is not False:
        raise ConfigError("Synthetic Engine forbids external text corpora and LLM-generated records")
    if synthetic.get("engine_version") != "nullxes-synthetic-engine.v5" \
            or synthetic.get("verifier_version") != "nullxes-synthetic-verifiers.v3":
        raise ConfigError("synthetic engine and answer verifier versions must be explicit")
    if type(synthetic.get("root_seed")) is not int or synthetic["root_seed"] < 0 \
            or type(synthetic.get("records_per_family")) is not int or synthetic["records_per_family"] <= 0:
        raise ConfigError("synthetic seed and per-family record count must be explicit non-negative/positive integers")
    threshold = synthetic.get("cross_split_near_duplicate_jaccard")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.0 < float(threshold) <= 1.0:
        raise ConfigError("synthetic cross-split near-duplicate threshold must be in (0, 1]")
    families = synthetic.get("families")
    if not isinstance(families, list) or not families:
        raise ConfigError("synthetic semantic/template/scenario families must be configured")
    seen_family_ids: set[str] = set()
    split_by_group: dict[tuple[str, str], str] = {}
    configured_splits: set[str] = set()
    configured_languages: set[str] = set()
    identity_train_counts = {language: 0 for language in ("en", "ru")}
    for family in families:
        if not isinstance(family, dict):
            raise ConfigError("each synthetic family must be a mapping")
        required_family = {"family_id", "language", "split", "task_kind", "functional_family",
                           "template_family", "scenario_family"}
        missing_family = required_family - family.keys()
        if missing_family:
            raise ConfigError(f"synthetic family is missing fields: {sorted(missing_family)}")
        family_id = family["family_id"]
        if not isinstance(family_id, str) or not family_id or family_id in seen_family_ids:
            raise ConfigError(f"synthetic family ID is empty or duplicated: {family_id!r}")
        seen_family_ids.add(family_id)
        if family["language"] not in {"en", "ru"} or family["split"] not in {"train", "validation", "blind"}:
            raise ConfigError(f"synthetic family {family_id} has unsupported language or split")
        if family["task_kind"] not in {"arithmetic", "mode_transition", "inventory", "location", "grounded_action", "identity"}:
            raise ConfigError(f"synthetic family {family_id} has unsupported task kind")
        if family["task_kind"] == "arithmetic" and family.get("operation") not in {
                "add", "subtract", "multiply", "divide"}:
            raise ConfigError(f"synthetic arithmetic family {family_id} needs a supported operation")
        if family["task_kind"] == "grounded_action" and family.get("scenario") not in {
                "stop_obstacle", "cancel_action", "object_presence", "action_sequence"}:
            raise ConfigError(f"synthetic grounded-action family {family_id} needs a supported scenario")
        if "records_per_family" in family and (type(family["records_per_family"]) is not int
                                                 or family["records_per_family"] <= 0):
            raise ConfigError(f"synthetic family {family_id} has an invalid record count")
        if family["task_kind"] == "identity" and family["split"] == "train":
            identity_train_counts[family["language"]] += int(family.get("records_per_family", 0))
        for group_field in ("functional_family", "template_family", "scenario_family"):
            group = family[group_field]
            if not isinstance(group, str) or not group:
                raise ConfigError(f"synthetic family {family_id} has an empty {group_field}")
            key = (group_field, group)
            previous_split = split_by_group.setdefault(key, family["split"])
            if previous_split != family["split"]:
                raise ConfigError(f"synthetic {group_field} {group!r} crosses corpus splits")
        configured_splits.add(family["split"])
        configured_languages.add(family["language"])
    if configured_splits != {"train", "validation", "blind"} or configured_languages != {"en", "ru"}:
        raise ConfigError("synthetic corpus requires all three splits and both EN/RU languages")
    if identity_train_counts != {"en": 1400, "ru": 1400}:
        raise ConfigError("identity foundation must contain exactly 1400 configured train records per language")
    tokenizer_language = data.get("tokenizer_language", {})
    if tokenizer_language.get("enabled") is not True \
            or tokenizer_language.get("engine_version") != "nullxes-native-language-engine.v1" \
            or tokenizer_language.get("records_per_concept") != 96 \
            or tokenizer_language.get("purpose") != "tokenizer_training_only" \
            or tokenizer_language.get("artifact_name") != "tokenizer_train.jsonl" \
            or tokenizer_language.get("external_text_sources_allowed") is not False \
            or tokenizer_language.get("external_llm_generation_allowed") is not False:
        raise ConfigError("tokenizer-only language corpus contract is invalid")
    if type(tokenizer_language.get("root_seed")) is not int or tokenizer_language["root_seed"] < 0:
        raise ConfigError("tokenizer-only language corpus root_seed must be a non-negative integer")
    if data.get("training_enabled") and config["training"].get("enabled"):
        if not tokenizer.get("artifact_sha256"):
            raise ConfigError("training requires a frozen tokenizer artifact SHA256")
        if not data.get("manifest_sha256"):
            raise ConfigError("training requires a frozen synthetic corpus manifest SHA256")
        if not data.get("split_salt"):
            raise ConfigError("training requires a frozen semantic-family split identity")
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
        raise ConfigError("checkpoint resume identity must bind train and validation packed manifest hashes")
    if config["data"].get("splits") != {"train": 0.98, "validation": 0.01, "blind": 0.01}:
        raise ConfigError("legacy split fractions are descriptive only and must retain documented values")
    cleaning = config["data"].get("cleaning", {})
    if cleaning.get("unicode_normalization") != "NFC" \
            or cleaning.get("policy_version") != "nullxes-synthetic-text.v1":
        raise ConfigError("synthetic corpus and Native tokenizer require the versioned NFC text policy")
    mix = config["data"].get("mix", {})
    if mix.get("unit") != "actual_shinra_tokens" or abs(float(mix.get("en", 0)) + float(mix.get("ru", 0)) - 1.0) > 1e-12:
        raise ConfigError("language mix must define normalized actual-token weights")
    if float(mix.get("en", 0)) != 0.5 or float(mix.get("ru", 0)) != 0.5:
        raise ConfigError("the approved initial training mix is exactly 50/50 EN/RU supervised tokens")
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

