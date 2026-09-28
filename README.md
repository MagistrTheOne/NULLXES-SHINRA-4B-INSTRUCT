# NULLXES SHINRA-4B

**Language Intelligence Layer** системы NULLXES. Создана **NULLXES**.

> SHINRA is SHINRA. Created by NULLXES. Compatibility with other model families is not SHINRA's identity or lineage. Hugging Face compatibility is an interface property, not model ancestry.

Полный текст: [`docs/SHINRA_INVARIANT.md`](docs/SHINRA_INVARIANT.md)

| Слой | Роль |
|------|------|
| RAIDEN | Reasoning Intelligence |
| CERBER | Vision Intelligence |
| **SHINRA** | **Language Intelligence** |
| AION | Embodied Intelligence |

SHINRA v2 — собственный decoder-only Transformer NULLXES: `ShinraConfig` / `ShinraForCausalLM`, random init, свой Unigram **131072**, свой training lineage. Hugging Face (`PreTrainedModel`, SDPA, safetensors, Hub) — интерфейс, не происхождение.

**Этот цикл:** барабан S0–S2 на Colab G4. Язык, не паспорт. Не INSTRUCT. Не кластерный 8192 / flash.

Hub `MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT` может всё ещё держать веса и геометрию **v1**. С него забирают **только файлы токенизатора**. `from_pretrained` весов запрещён.

Выходы (позже, не сейчас):

1. `NULLXES SHINRA-4B-BASE` — претрейн  
2. `NULLXES SHINRA-4B-INSTRUCT` — instruction tuning  
3. aligned instruct — DPO  

---

## Архитектура SHINRA CORE

```
Embedding
  → [ RMSNorm → GQA+RoPE (+QK-norm) → residual
      RMSNorm → SwiGLU              → residual ] × 36
  → RMSNorm → LM Head (tied)
```

| | |
|---|---|
| parameters (tied) | 3,969,056,256 |
| hidden (residual) | 2560 |
| Q width | 4096 (32 × 128) |
| layers | 36 |
| heads / KV | 32 / 8 |
| head dim | 128 |
| SwiGLU | 9728 |
| vocab | 131072 |
| train context (S0–S2 G4) | 2048 |
| native window | 32768 |
| precision | BF16 |
| G4 attention | SDPA |

Подсчёт: `python -m architecture.param_count`  
Спека: [`architecture/design.md`](architecture/design.md)

---

## Токенизатор

SentencePiece **Unigram**, 131072, NFKC, byte fallback. DNA своя.

S0–S2 wrap: `[BOS] + body + [END_OF_TEXT]`.  
`<|eot|>` (id 2, HF `eos_token_id`) и `<|end_of_text|>` (id 18) — разные токены. Pretrain не вызывает `chat_template.jinja`.

```bash
python scripts/fetch_tokenizer.py --dest tokenizer/artifacts
```

Только `tokenizer.json` / `tokenizer.model` / configs. Не `*.safetensors`.

---

## Барабан S0–S2 (Colab G4)

Локальная RTX 2080 не цель. Железо: [`configs/colab.yaml`](configs/colab.yaml) (`max_tokens: 0`, `disk_ceiling_gb: 400`). Рецепты стадий: [`configs/stages/`](configs/stages/).

| стадия | новые токены | кумулятив | данные этого цикла |
|--------|-------------:|----------:|--------------------|
| S0 bring-up | 20M | 20M | 100% synth |
| S1 language | 230M | 250M | 100% synth |
| S2 semantic | 750M | 1B | 100% synth |

Спека до ~80B: [`data/specs/SHINRA_V2_BASE_80B_CURRICULUM.md`](data/specs/SHINRA_V2_BASE_80B_CURRICULUM.md). S3+ здесь не качаются.

```bash
pip install -e .
export PYTHONPATH=.

python scripts/fetch_tokenizer.py --dest tokenizer/artifacts
python scripts/v2_s0_colab.py --fetch-tokenizer \
  --corpus-dir /content/shinra_scratch/corpus \
  --output-dir /content/shinra_scratch/s0
```

Ноутбук: [`notebooks/SHINRA_V2_S0.ipynb`](notebooks/SHINRA_V2_S0.ipynb)  
Контроллер: `scripts/v2_stage_run.py`  
Колаб: [`docs/colab.md`](docs/colab.md)

Статус: `run/status.json` + ledger. Attention: **SDPA**, не FA3.

S0–S2 не identity-conditioning: никаких «I am SHINRA», диалогов system/user/assistant, identity QA.

---

## Репозиторий

```
architecture/     спецификация и подсчёт параметров
model/            ShinraConfig, блоки, GQA+RoPE, SwiGLU, HF Auto*
tokenizer/        SentencePiece Unigram 131072; chat_template.jinja для будущего INSTRUCT
data/             synth S0–S2, pack wrap, rolling shards, ledger
training/         LM loop (честный token count / resume)
evaluation/       S0–S2 gates
configs/stages/   s0_bringup / s1_language / s2_semantic
scripts/          fetch_tokenizer, v2_s0_colab, v2_stage_run
docs/             invariant, colab, architecture
notebooks/        SHINRA_V2_S0.ipynb
```

Кластерные `scripts/train_pretrain.sh`, pack 8192, extra `flash` — не этот цикл. Не запускать их вместо барабана G4.

---

## Лицензия

NULLXES Research License. Веса — proprietary asset. См. [`LICENSE`](LICENSE).
