# Google Colab — SHINRA bring-up

Локальная RTX 2080 **не является** целью. 4B BF16 + AdamW туда не встаёт.

| Среда | Назначение |
|-------|------------|
| Google Colab Pro **A100 80GB** (167 GB RAM) | Phase 0: pipeline + 100M token hypothesis |
| 8× A100 80GB | SHINRA-4B-BASE / SFT / DPO |

Ноутбук: [`notebooks/SHINRA_COLAB.ipynb`](../notebooks/SHINRA_COLAB.ipynb)  
Конфиги: [`configs/colab.yaml`](../configs/colab.yaml), [`configs/dataset_pilot.yaml`](../configs/dataset_pilot.yaml), [`configs/pretrain_colab_100m.yaml`](../configs/pretrain_colab_100m.yaml)

vLLM/SGLang/TokenSpeed: [`runtime/README.md`](../runtime/README.md) — serving после весов, не Colab pretrain.
