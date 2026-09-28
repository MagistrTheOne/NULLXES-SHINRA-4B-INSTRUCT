# Google Colab G4 — SHINRA v2 S0–S2

Локальная RTX 2080 **не является** целью.

| Среда | Назначение |
|-------|------------|
| Google Colab **G4** (RTX PRO 6000 Blackwell) | барабан S0–S2, BF16, SDPA, tokenizer DNA |
| Production cluster | не этот цикл |

`configs/colab.yaml` — **железо** (`disk_ceiling_gb: 400`, `max_tokens: 0`). Train recipe: `configs/stages/s0_bringup.yaml`.

Ноутбук барабана: [`notebooks/SHINRA_V2_S0.ipynb`](../notebooks/SHINRA_V2_S0.ipynb)

```bash
pip install -e .
export PYTHONPATH=.
python scripts/fetch_tokenizer.py --dest tokenizer/artifacts
python scripts/v2_s0_colab.py --fetch-tokenizer \
  --corpus-dir /content/shinra_scratch/corpus \
  --output-dir /content/shinra_scratch/s0
```

Не `from_pretrained` весов. Не S0.4.1. Не P0. Не identity. Не `chat_template.jinja`. Attention: SDPA, не FA3.

Статус: `run/status.json` + `run/metrics.jsonl`.
