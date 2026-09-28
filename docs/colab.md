# Google Colab G4 — SHINRA v2 S0–S2

Локальная RTX 2080 **не является** целью.

| Среда | Назначение |
|-------|------------|
| Google Colab **G4** (RTX PRO 6000 Blackwell) | барабан S0–S2, BF16, SDPA, tokenizer DNA |
| Production cluster | не этот цикл |

`configs/colab.yaml` — **железо** (`disk_ceiling_gb: 400`, `max_tokens: 0`). Train recipe: `configs/stages/s0_bringup.yaml`.

Нужен `main` ≥ **`90c5e9a`** (`data/rolling_drum.py`).

Ноутбук барабана: [`notebooks/SHINRA_V2_S0.ipynb`](../notebooks/SHINRA_V2_S0.ipynb)  
Init-only (не S0): [`notebooks/SHINRA_COLAB.ipynb`](../notebooks/SHINRA_COLAB.ipynb)

```bash
pip install -e .
export PYTHONPATH=.
python scripts/fetch_tokenizer.py --dest tokenizer/artifacts
python scripts/v2_s0_colab.py --fetch-tokenizer \
  --corpus-dir /content/shinra_scratch/corpus \
  --output-dir /content/shinra_scratch/s0
```

Барабан: produce shard N → SHA → train once → consume/delete → N+1.  
Crash: не трогать `/content/shinra_scratch/s0`; та же команда читает `run/cursor.json` + checkpoint SHA.

Не `from_pretrained` весов. Не S0.4.1. Не P0. Не identity. Не `chat_template.jinja`. Attention: SDPA, не FA2/FA3. Не `pip install ".[train,flash]"`.

Статус: `run/status.json` + `run/cursor.json` + `run/ledger.jsonl` + `run/metrics.jsonl`.
