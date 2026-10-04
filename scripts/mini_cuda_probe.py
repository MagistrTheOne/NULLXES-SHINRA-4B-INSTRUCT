from pathlib import Path

import torch

from model.modeling_shinra import ShinraForCausalLM
from training.debug import capture_runtime_snapshot, summarize_gradients, summarize_optimizer_state
from training.trainer import config_from_yaml


if __name__ == '__main__':
    cfg = config_from_yaml(Path('configs/architecture_v2.yaml'), 'sdpa')
    model = ShinraForCausalLM(cfg).to(device='cuda', dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for seq in [128, 512, 1024, 2048]:
        print(f'\n== SEQ {seq} ==')
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, cfg.vocab_size, (1, seq), device='cuda')
        labels = x.clone()
        mask = torch.ones_like(x, device='cuda', dtype=torch.long)
        try:
            capture_runtime_snapshot(f'before_forward_seq{seq}', step=0, microbatch=0, sync_gradients=False, model=model)
            outputs = model(input_ids=x, attention_mask=mask, labels=labels, use_cache=False)
            print('loss_finite=', torch.isfinite(outputs.loss).item(), 'loss=', float(outputs.loss.detach().cpu()))
            model.zero_grad(set_to_none=True)
            outputs.loss.backward()
            print('grad_nan=', torch.isnan(torch.cat([p.grad.detach().reshape(-1).float() for p in model.parameters() if p.grad is not None])).any().item() if any(p.grad is not None for p in model.parameters()) else 'none')
            print('grad_summary=', summarize_gradients(model))
            optimizer.step()
            print('optimizer_state=', summarize_optimizer_state(optimizer))
            print('peak_memory_MB=', torch.cuda.max_memory_allocated() / 1024**2)
            print('reserved_MB=', torch.cuda.max_memory_reserved() / 1024**2)
            print('alloc_MB=', torch.cuda.memory_allocated() / 1024**2)
        except Exception as exc:
            print('FAIL_SEQ', seq, type(exc).__name__, exc)
            break
