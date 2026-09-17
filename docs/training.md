# Обучение SHINRA

## Стадии

| Стадия | Скрипт | Конфиг | Выход |
|--------|--------|--------|-------|
| 1 Pretrain | `scripts/train_pretrain.sh` | `configs/pretrain_a100.yaml` | `NULLXES SHINRA-4B-BASE` |
| 2 SFT | `scripts/train_sft.sh <BASE>` | `configs/sft_a100.yaml` | `NULLXES SHINRA-4B-INSTRUCT` |
| 3 DPO | `scripts/train_dpo.sh <SFT>` | `configs/dpo_a100.yaml` | aligned instruct |

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

## Команды

```bash
pip install -e ".[train,flash]"
bash scripts/train_pretrain.sh
bash scripts/train_sft.sh outputs/shinra-4b-base/final/step-00095367
bash scripts/train_dpo.sh outputs/shinra-4b-instruct/final/step-00008000
```

Мультинод (4 узла × 8 GPU): `configs/accelerate_a100_multinode.yaml`, выставить `main_process_ip`.

## Мониторинг

Каждые 10 шагов: `loss`, `ppl`, `lr`, `tokens/s`.  
Алерты: NaN, loss > 2× медианы, падение throughput > 20%.  
Чекпоинты каждые 1000 шагов, rolling + milestone.

## Инициализация

`N(0, 0.02)` на линейных и эмбеддингах; `o_proj` / `down_proj` масштабируются `0.02 / sqrt(2L)`. RMSNorm = 1. Z-loss `1e-5 * mean(logZ^2)`.
