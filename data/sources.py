"""Dataset mix recipes for SHINRA. Pilot is frozen-enough for Colab; PRETRAIN_MIX is BASE v1, not a 200B lock."""

from __future__ import annotations

import json

# SHINRA-4B-BASE v1 (cluster). Not executed on Colab.
# 40% web / 20% code / 15% math+science / 10% books / 10% multilingual / 5% NULLXES engineering
PRETRAIN_MIX: dict[str, dict] = {
    "web_edu": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "weight": 0.40,
        "domain": "web",
        "license": "ODC-By",
        "note": "High-quality web, not the full FineWeb-Edu 1.3T dump. Cluster may switch subset to sample-100BT.",
    },
    "python_edu": {
        "hf_id": "HuggingFaceTB/smollm-corpus",
        "subset": "python-edu",
        "split": "train",
        "text_field": "text",
        "weight": 0.10,
        "domain": "code",
        "license": "ODC-By",
        "note": "Code + explanations, not raw dumps.",
    },
    "the_stack_smol": {
        "hf_id": "bigcode/the-stack-smol-xl",
        "subset": None,
        "split": "train",
        "text_field": "content",
        "weight": 0.10,
        "domain": "code",
        "license": "various-permissive",
        "note": "Prefer files with README/docs siblings in later ingest; licensed stack only.",
    },
    "openwebmath": {
        "hf_id": "open-web-math/open-web-math",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.07,
        "domain": "math",
        "license": "ODC-By",
    },
    "proof_pile": {
        "hf_id": "EleutherAI/proof-pile-2",
        "subset": "algebraic-stack",
        "split": "train",
        "text_field": "text",
        "weight": 0.02,
        "domain": "math",
        "license": "various",
    },
    "pes2o": {
        "hf_id": "allenai/peS2o",
        "subset": "v2",
        "split": "train",
        "text_field": "text",
        "weight": 0.03,
        "domain": "science",
        "license": "ODC-By",
    },
    "nullxes_arxiv_cs": {
        "hf_id": "armanc/scientific_papers",
        "subset": "arxiv",
        "split": "train",
        "text_field": "article",
        "weight": 0.03,
        "domain": "nullxes",
        "license": "arxiv",
        "keyword_any": [
            "robot",
            "robotics",
            "autonom",
            "control system",
            "cuda",
            "gpu",
            "pytorch",
            "embedded",
            "kinematic",
            "slam",
            "lidar",
            "actuator",
            "firmware",
            "real-time",
            "realtime",
            "compiler",
            "kernel",
            "tensor",
            "navigation",
            "sensor fusion",
            "reinforcement learning",
        ],
        "note": "Ingested before general arXiv so keyword hits are not eaten by MinHash. Full paper body.",
    },
    "arxiv_full": {
        "hf_id": "armanc/scientific_papers",
        "subset": "arxiv",
        "split": "train",
        "text_field": "article",
        "weight": 0.03,
        "domain": "science",
        "license": "arxiv",
        "note": "Full paper body, not abstract-only. Remainder after NULLXES keyword slice.",
    },
    "pg19": {
        "hf_id": "deepmind/pg19",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.07,
        "domain": "books",
        "license": "public-domain",
    },
    "gutenberg": {
        "hf_id": "sedthh/gutenberg_english",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.03,
        "domain": "books",
        "license": "public-domain",
    },
    "wikipedia_ru": {
        "hf_id": "wikimedia/wikipedia",
        "subset": "20231101.ru",
        "split": "train",
        "text_field": "text",
        "weight": 0.05,
        "domain": "encyclopedia",
        "language": "ru",
        "license": "CC-BY-SA-3.0",
    },
    "wikipedia_de": {
        "hf_id": "wikimedia/wikipedia",
        "subset": "20231101.de",
        "split": "train",
        "text_field": "text",
        "weight": 0.025,
        "domain": "encyclopedia",
        "language": "de",
        "license": "CC-BY-SA-3.0",
    },
    "wikipedia_fr": {
        "hf_id": "wikimedia/wikipedia",
        "subset": "20231101.fr",
        "split": "train",
        "text_field": "text",
        "weight": 0.025,
        "domain": "encyclopedia",
        "language": "fr",
        "license": "CC-BY-SA-3.0",
    },
    "nullxes_stack_docs": {
        "hf_id": "bigcode/the-stack-smol-xl",
        "subset": None,
        "split": "train",
        "text_field": "content",
        "weight": 0.02,
        "domain": "nullxes",
        "license": "various-permissive",
        "path_suffixes": [".md", ".rst", ".cu", ".cuh"],
        "note": "Markdown/RST/CUDA docs from licensed stack — PyTorch/CUDA/embedded terminology.",
    },
}

SFT_MIX: dict[str, dict] = {
    "tulu": {
        "hf_id": "allenai/tulu-v2-sft-mixture",
        "split": "train",
        "weight": 0.18,
        "kind": "conversation",
    },
    "smol_smoltalk": {
        "hf_id": "HuggingFaceTB/smol-smoltalk",
        "split": "train",
        "weight": 0.12,
        "kind": "conversation",
    },
    "code_feedback": {
        "hf_id": "m-a-p/CodeFeedback-Filtered-Instruction",
        "split": "train",
        "weight": 0.15,
        "kind": "code",
    },
    "magicoder": {
        "hf_id": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "split": "train",
        "weight": 0.10,
        "kind": "code",
    },
    "openmathinstruct": {
        "hf_id": "nvidia/OpenMathInstruct-2",
        "split": "train_1M",
        "weight": 0.20,
        "kind": "reasoning",
        "note": "train_1M slice, not the 14M dump.",
    },
    "hermes_tools": {
        "hf_id": "NousResearch/hermes-function-calling-v1",
        "subset": "func_calling",
        "split": "train",
        "weight": 0.10,
        "kind": "agent",
        "note": "Apache-2.0 function calling. Salesforce xLAM is gated; this feeds <|tool_call|> DNA.",
    },
    "hermes_json": {
        "hf_id": "NousResearch/hermes-function-calling-v1",
        "subset": "json_mode_agentic",
        "split": "train",
        "weight": 0.05,
        "kind": "agent",
        "note": "Structured JSON outputs for format adherence.",
    },
    "openhermes": {
        "hf_id": "teknium/OpenHermes-2.5",
        "split": "train",
        "weight": 0.10,
        "kind": "general",
    },
}

DPO_MIX: dict[str, dict] = {
    "ultrafeedback": {
        "hf_id": "HuggingFaceH4/ultrafeedback_binarized",
        "split": "train_prefs",
        "weight": 0.40,
        "kind": "general",
    },
    "code_prefs": {
        "hf_id": "jondurbin/py-dpo-v0.1",
        "split": "train",
        "weight": 0.30,
        "kind": "code",
        "license": "cc-by-4.0",
        "note": "Python coding preference. Replaces the 404 mlabonne/orca-dpo-pairs-cleaned.",
    },
    "dpo_mix_tools": {
        "hf_id": "argilla/dpo-mix-7k",
        "split": "train",
        "weight": 0.20,
        "kind": "tools",
        "note": "Instruction/format preference until a dedicated tool-call DPO dump exists.",
    },
    "orca_format": {
        "hf_id": "Intel/orca_dpo_pairs",
        "split": "train",
        "weight": 0.10,
        "kind": "formatting",
    },
}

# Colab diagnostic mix. Not BASE.
PILOT_MIX: dict[str, dict] = {
    "fineweb_edu_10bt": {
        "hf_id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "weight": 0.50,
        "bucket": "general",
        "domain": "web",
        "license": "ODC-By",
    },
    "python_edu": {
        "hf_id": "HuggingFaceTB/smollm-corpus",
        "subset": "python-edu",
        "split": "train",
        "text_field": "text",
        "weight": 0.15,
        "bucket": "code",
        "domain": "code",
        "license": "ODC-By",
    },
    "stack_smol": {
        "hf_id": "bigcode/the-stack-smol-xl",
        "subset": None,
        "split": "train",
        "text_field": "content",
        "weight": 0.10,
        "bucket": "code",
        "domain": "code",
        "license": "various-permissive",
    },
    "openwebmath": {
        "hf_id": "open-web-math/open-web-math",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "weight": 0.15,
        "bucket": "math_stem",
        "domain": "math",
        "license": "ODC-By",
    },
    "wikipedia_ru": {
        "hf_id": "wikimedia/wikipedia",
        "subset": "20231101.ru",
        "split": "train",
        "text_field": "text",
        "weight": 0.07,
        "bucket": "multilingual",
        "domain": "encyclopedia",
        "language": "ru",
        "license": "CC-BY-SA-3.0",
    },
    "wikipedia_de": {
        "hf_id": "wikimedia/wikipedia",
        "subset": "20231101.de",
        "split": "train",
        "text_field": "text",
        "weight": 0.015,
        "bucket": "multilingual",
        "domain": "encyclopedia",
        "language": "de",
        "license": "CC-BY-SA-3.0",
    },
    "wikipedia_fr": {
        "hf_id": "wikimedia/wikipedia",
        "subset": "20231101.fr",
        "split": "train",
        "text_field": "text",
        "weight": 0.015,
        "bucket": "multilingual",
        "domain": "encyclopedia",
        "language": "fr",
        "license": "CC-BY-SA-3.0",
    },
}

PILOT_BUCKET_WEIGHTS = {"general": 0.50, "code": 0.25, "math_stem": 0.15, "multilingual": 0.10}
# Diagnostic mix still has RU wiki; BASE language target is en 75 / ru 15 / other 10.
PILOT_LANGUAGE_QUOTAS = {"en": 0.75, "ru": 0.15, "eu": 0.10}

PRODUCTION_MIX_V1 = {
    "high_quality_web": 0.40,
    "code": 0.20,
    "math_science": 0.15,
    "books": 0.10,
    "multilingual": 0.10,
    "nullxes_engineering": 0.05,
}

SFT_MIX_V1 = {
    "conversation": 0.30,
    "code": 0.25,
    "math_reasoning": 0.20,
    "agent_tools": 0.15,
    "general": 0.10,
}

DPO_MIX_V1 = {
    "general": 0.40,
    "code": 0.30,
    "tools": 0.20,
    "formatting": 0.10,
}


def mix_weight_sum(mix: dict[str, dict]) -> float:
    return round(sum(float(spec.get("weight", 0.0)) for spec in mix.values()), 6)


def audit_mix_weights() -> dict[str, float]:
    grouped_pretrain: dict[str, float] = {}
    domain_to_bucket = {
        "web": "high_quality_web",
        "code": "code",
        "math": "math_science",
        "science": "math_science",
        "books": "books",
        "encyclopedia": "multilingual",
        "nullxes": "nullxes_engineering",
    }
    for spec in PRETRAIN_MIX.values():
        bucket = domain_to_bucket.get(str(spec.get("domain")), str(spec.get("domain")))
        grouped_pretrain[bucket] = grouped_pretrain.get(bucket, 0.0) + float(spec["weight"])
    grouped_sft: dict[str, float] = {}
    kind_to_bucket = {
        "conversation": "conversation",
        "code": "code",
        "reasoning": "math_reasoning",
        "agent": "agent_tools",
        "general": "general",
    }
    for spec in SFT_MIX.values():
        bucket = kind_to_bucket.get(str(spec.get("kind")), str(spec.get("kind")))
        grouped_sft[bucket] = grouped_sft.get(bucket, 0.0) + float(spec["weight"])
    grouped_dpo: dict[str, float] = {}
    for spec in DPO_MIX.values():
        kind = str(spec.get("kind"))
        grouped_dpo[kind] = grouped_dpo.get(kind, 0.0) + float(spec["weight"])
    return {
        "pretrain_sources": mix_weight_sum(PRETRAIN_MIX),
        "sft_sources": mix_weight_sum(SFT_MIX),
        "dpo_sources": mix_weight_sum(DPO_MIX),
        "pilot_sources": mix_weight_sum(PILOT_MIX),
        "pilot_buckets": round(sum(PILOT_BUCKET_WEIGHTS.values()), 6),
        "pilot_languages": round(sum(PILOT_LANGUAGE_QUOTAS.values()), 6),
        "production_v1": round(sum(PRODUCTION_MIX_V1.values()), 6),
        "sft_v1": round(sum(SFT_MIX_V1.values()), 6),
        "dpo_v1": round(sum(DPO_MIX_V1.values()), 6),
        **{f"pretrain_{k}": round(v, 6) for k, v in grouped_pretrain.items()},
        **{f"sft_{k}": round(v, 6) for k, v in grouped_sft.items()},
        **{f"dpo_{k}": round(v, 6) for k, v in grouped_dpo.items()},
    }


def assert_mix_weights() -> dict[str, float]:
    audit = audit_mix_weights()
    for key in (
        "pretrain_sources",
        "sft_sources",
        "dpo_sources",
        "pilot_sources",
        "pilot_buckets",
        "pilot_languages",
        "production_v1",
        "sft_v1",
        "dpo_v1",
    ):
        if abs(audit[key] - 1.0) > 1e-6:
            raise ValueError(f"{key} weights sum to {audit[key]}, expected 1.0")
    for bucket, target in PRODUCTION_MIX_V1.items():
        actual = audit.get(f"pretrain_{bucket}")
        if actual is not None and abs(actual - target) > 1e-6:
            raise ValueError(f"PRETRAIN {bucket}={actual}, PRODUCTION_MIX_V1={target}")
    for bucket, target in SFT_MIX_V1.items():
        actual = audit.get(f"sft_{bucket}")
        if actual is not None and abs(actual - target) > 1e-6:
            raise ValueError(f"SFT {bucket}={actual}, SFT_MIX_V1={target}")
    for bucket, target in DPO_MIX_V1.items():
        actual = audit.get(f"dpo_{bucket}")
        if actual is not None and abs(actual - target) > 1e-6:
            raise ValueError(f"DPO {bucket}={actual}, DPO_MIX_V1={target}")
    return audit


if __name__ == "__main__":
    print(json.dumps(assert_mix_weights(), indent=2))
