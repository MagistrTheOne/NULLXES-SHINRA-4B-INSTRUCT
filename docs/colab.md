# Google Colab — SHINRA v2 random init

Локальная RTX 2080 **не является** целью.

| Среда | Назначение |
|-------|------------|
| Google Colab **G4** (RTX PRO 6000 Blackwell) | v2 random init, BF16 forward/backward, tokenizer DNA |
| Production cluster | не заморожен в architecture v2 |

Ноутбук: [`notebooks/SHINRA_COLAB.ipynb`](../notebooks/SHINRA_COLAB.ipynb)  
Конфиги: [`configs/colab.yaml`](../configs/colab.yaml), [`configs/shinra_4b.yaml`](../configs/shinra_4b.yaml)

```bash
python -m scripts.phase01_bringup --config configs/shinra_4b.yaml
```

Не `from_pretrained`. Не S0.4.1. Не P0. После init — отдельный аудит RMSNorm / packing / optimizer, затем pretrain.

vLLM/SGLang/TokenSpeed: [`runtime/README.md`](../runtime/README.md) — serving после весов, не Colab init.
