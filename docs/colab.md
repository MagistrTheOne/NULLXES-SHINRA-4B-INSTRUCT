# Google Colab — SHINRA bring-up

Локальная RTX 2080 **не является** целью. 4B BF16 + AdamW туда не встаёт. Не ставить модель локально.

| Среда | Назначение |
|-------|------------|
| Google Colab Pro **A100 40GB** | tokenizer, data dry-run, meta param count, короткий pipeline check |
| 8× A100 80GB | pretrain / SFT / DPO |

Ноутбук: [`notebooks/SHINRA_COLAB.ipynb`](../notebooks/SHINRA_COLAB.ipynb)

Конфиг: [`configs/colab.yaml`](../configs/colab.yaml)

На Colab:

1. Runtime → GPU → A100
2. Клонировать `MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT`
3. Обучить tokenizer на representative mix (не на 2080)
4. Полный претрейн — только кластер из `configs/pretrain_a100.yaml`
