# Обучение SHINRA

**Этот цикл — Colab G4 S0–S2.** Команды и extra `flash` ниже — кластерный рецепт, не барабан. Барабан: [`README.md`](../README.md), [`docs/colab.md`](colab.md), `scripts/v2_s0_colab.py`.

Architecture **v2** is frozen in `configs/architecture_v2.yaml` / `architecture/design.md` (legacy mixed file `configs/shinra_4b.yaml` removed; S0/C history ran with identical geometry). Optimizer, WSD, packing, and cluster topology are **not** frozen. Do not resume S0.4.1 or P0.

Init smoke (не S0 train): Colab G4, `python -m scripts.phase01_bringup --config configs/architecture_v2.yaml`.

## Стадии (A100-цепочка — архив, не активный V2-путь)

| Стадия | Скрипт (архив) | Конфиг (архив) | Выход |
|--------|--------|--------|-------|
| 1 Pretrain | `training.pretrain` | A100 recipe (removed from active tree) | `NULLXES SHINRA-4B-BASE` |
| 2 SFT | `training.sft <BASE>` | A100 recipe (removed from active tree) | `NULLXES SHINRA-4B-INSTRUCT` |
| 3 DPO | `training.dpo <SFT>` | A100 recipe (removed from active tree) | aligned instruct |

Активный V2-путь: `scripts/v2_stage_run.py --config configs/architecture_v2.yaml --runtime-config configs/runtime_g4.yaml --stage-config <stage>`. См. `docs/CONFIG_LINK_MAP.md`.

## Узел A100

- 8 × NVIDIA A100 80GB
- PyTorch 2.x, BF16, TF32
- Accelerate FSDP `FULL_SHARD`, wrap `ShinraDecoderLayer`
- Gradient checkpointing (`use_reentrant=False`)
- fused AdamW, β=(0.9, 0.95), wd=0.1
- WSD: warmup 2k → 80% stable → cosine decay до 0.1×LR
- FlashAttention через SDPA (по умолчанию) или `flash_attention_2`

Глобальный батч претрейна: `8 × 2 × 16 × 8192 = 2,097,152` токена/шаг.  
200B токенов ≈ 95 367 шагов.

## Команды (архив; entrypoints scripts/train_*.sh удалены из активного дерева)

```bash
pip install -e ".[train]"
python -m training.pretrain --train-config <explicit-recipe> --data-dir <dir> --output-dir <dir>
```

Мультинод-архив удалён из активного дерева (`configs/accelerate_a100*.yaml` — см. git history).

## Мониторинг

Каждые 10 шагов: `loss`, `ppl`, `lr`, `tokens/s`.  
Алерты: NaN, loss > 2× медианы, падение throughput > 20%.  
Чекпоинты каждые 1000 шагов, rolling + milestone.

## Инициализация

`N(0, 0.02)` на линейных и эмбеддингах; `o_proj` / `down_proj` масштабируются `0.02 / sqrt(2L)`. RMSNorm = 1. Z-loss `1e-5 * mean(logZ^2)`.
