"""Phase 0.1 — prove SHINRA v2 ShinraForCausalLM lives on Colab G4. Random init only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from architecture.param_count import ShinraSpec, count_parameters
from model import ShinraConfig, ShinraForCausalLM
from tokenizer.special_tokens import ALL_SPECIAL_TOKENS, EOT, END_OF_TEXT
from training.optim import build_optimizer

EXPECTED_PARAMS = count_parameters(ShinraSpec())["total"]
MIN_VRAM_GB = 40.0


def environment_lock() -> dict:
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("Phase 0.1 requires CUDA")
    name = torch.cuda.get_device_name(0)
    if "2080" in name:
        raise SystemExit("RTX 2080 is not a SHINRA target")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    probe = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    bf16_ok = (probe @ probe.T).dtype == torch.bfloat16
    report = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": name,
        "vram_gb": round(vram_gb, 2),
        "bf16": bf16_ok,
        "architecture": "shinra_v2",
        "expected_parameters": EXPECTED_PARAMS,
    }
    print(json.dumps(report, indent=2))
    if vram_gb < MIN_VRAM_GB:
        raise SystemExit(f"Need >= {MIN_VRAM_GB:.0f} GB VRAM for SHINRA-4B init, got {vram_gb:.1f} GB")
    if not bf16_ok:
        raise SystemExit("bfloat16 matmul failed")
    return report


def tokenizer_dna_report() -> dict:
    report = {
        "n_specials": len(ALL_SPECIAL_TOKENS),
        "eot": EOT,
        "end_of_text": END_OF_TEXT,
        "specials": list(ALL_SPECIAL_TOKENS),
        "has_end_alias": "<|end|>" in ALL_SPECIAL_TOKENS,
        "has_tool_alias": "<|tool|>" in ALL_SPECIAL_TOKENS,
        "vocab_size_target": 131072,
    }
    artifacts = Path("tokenizer/artifacts/tokenizer_stats.json")
    if artifacts.exists():
        report["trained_stats"] = json.loads(artifacts.read_text(encoding="utf-8"))
    print(json.dumps({k: report[k] for k in report if k != "specials"}, indent=2, ensure_ascii=False))
    if report["has_end_alias"] or report["has_tool_alias"]:
        raise SystemExit("Frozen DNA violated")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA v2 Colab G4 random-init bring-up")
    parser.add_argument("--config", default="configs/shinra_4b.yaml")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    args = parser.parse_args()

    env = environment_lock()
    dna = tokenizer_dna_report()

    config = ShinraConfig.from_yaml(args.config)
    config.use_cache = False
    model = ShinraForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print("parameters", n_params)
    if n_params != EXPECTED_PARAMS:
        raise SystemExit(f"param count {n_params} != {EXPECTED_PARAMS}")
    q_width = config.num_attention_heads * config.head_dim
    print("q_width", q_width, "hidden", config.hidden_size, "layers", config.num_hidden_layers)
    if q_width == config.hidden_size:
        raise SystemExit("v2 requires decoupled Q width")
    if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads) != (2560, 36, 32):
        raise SystemExit("v2 geometry mismatch")

    tied = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print("tied_embeddings", tied)
    if not tied:
        raise SystemExit("lm_head is not tied to embed_tokens")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = model.to(dtype=torch.bfloat16, device="cuda")
    dtype = next(model.parameters()).dtype
    print("dtype", dtype)
    if dtype != torch.bfloat16:
        raise SystemExit(f"expected bfloat16, got {dtype}")

    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq), device="cuda")
    model.train()
    out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    print("logits", tuple(out.logits.shape))
    expected_logits = (args.batch, args.seq, config.vocab_size)
    if tuple(out.logits.shape) != expected_logits:
        raise SystemExit(f"logits {tuple(out.logits.shape)} != {expected_logits}")
    if out.loss is None or not torch.isfinite(out.loss):
        raise SystemExit("loss is missing or non-finite")

    out.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print("loss", float(out.loss.detach()), "grad_norm", float(grad_norm))
    if not torch.isfinite(grad_norm):
        raise SystemExit("non-finite gradients")

    optimizer = build_optimizer(model, args.lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("optimizer_step", "adamw_fused", args.lr)

    report = {
        "env": env,
        "parameters": n_params,
        "tied_embeddings": tied,
        "dtype": str(dtype),
        "logits": list(expected_logits),
        "loss": float(out.loss.detach()),
        "grad_norm": float(grad_norm),
        "optimizer": "adamw_torch_fused",
        "lr": args.lr,
        "tokenizer_dna": {k: dna[k] for k in ("n_specials", "eot", "end_of_text")},
        "phase": "v2_g4_random_init",
        "hidden_size": config.hidden_size,
        "q_width": q_width,
        "layers": config.num_hidden_layers,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
