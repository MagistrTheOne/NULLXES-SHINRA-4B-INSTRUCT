"""Architecture verification: parameter count, shapes, BF16 forward, generate path."""

from __future__ import annotations

import argparse
import json

import torch

from architecture.param_count import ShinraSpec, count_parameters
from model.configuration_shinra import ShinraConfig
from model.modeling_shinra import ShinraForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SHINRA-4B architecture")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--forward", action="store_true", help="Run a BF16 CUDA forward+generate pass")
    parser.add_argument("--backward", action="store_true", help="Run one BF16 train step (forward+backward)")
    parser.add_argument("--attn", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    analytical = count_parameters(ShinraSpec())
    cfg = ShinraConfig(use_cache=True, attention_implementation=args.attn)
    cfg._attn_implementation = args.attn
    meta = torch.device("meta")
    with torch.device(meta):
        meta_model = ShinraForCausalLM(cfg)
        trainable = sum(p.numel() for p in meta_model.parameters())
    report = {
        "analytical_params": analytical["total"],
        "nn_params": trainable,
        "delta": abs(analytical["total"] - trainable),
        "billions": trainable / 1e9,
        "hidden_size": cfg.hidden_size,
        "layers": cfg.num_hidden_layers,
        "heads": cfg.num_attention_heads,
        "kv_heads": cfg.num_key_value_heads,
        "vocab": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
    }
    if args.forward:
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise SystemExit("--forward requires CUDA")
        model = ShinraForCausalLM(cfg).to(device=args.device, dtype=torch.bfloat16)
        model.eval()
        input_ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids)
            generated = model.generate(input_ids[:, :16], max_new_tokens=8, use_cache=True)
        report.update(
            {
                "loss_finite": bool(torch.isfinite(out.loss)),
                "logits_shape": list(out.logits.shape),
                "generate_shape": list(generated.shape),
                "dtype": str(next(model.parameters()).dtype),
                "device": str(next(model.parameters()).device),
            }
        )
        if not report["loss_finite"]:
            raise SystemExit("Non-finite loss")
    if args.backward:
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise SystemExit("--backward requires CUDA")
        model = ShinraForCausalLM(cfg).to(device=args.device, dtype=torch.bfloat16)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        input_ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=model.device)
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        if not torch.isfinite(out.loss):
            raise SystemExit("Non-finite loss before backward")
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        report.update(
            {
                "backward_loss": float(out.loss.detach().item()),
                "grad_norm": float(grad_norm),
                "grad_finite": bool(torch.isfinite(grad_norm)),
                "dtype": str(next(model.parameters()).dtype),
                "device": str(next(model.parameters()).device),
            }
        )
        if not report["grad_finite"]:
            raise SystemExit("Non-finite grad_norm")
    print(json.dumps(report, indent=2))
    if abs(analytical["total"] - trainable) > 10_000:
        raise SystemExit("Parameter count mismatch exceeds 10k")


if __name__ == "__main__":
    main()
