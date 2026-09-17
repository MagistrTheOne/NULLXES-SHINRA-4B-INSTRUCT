# NULLXES SHINRA-4B-INSTRUCT

**Language Intelligence Layer** системы NULLXES.

| Слой | Роль |
|------|------|
| RAIDEN | Reasoning Intelligence |
| CERBER | Vision Intelligence |
| **SHINRA** | **Language Intelligence** |
| AION | Embodied Intelligence |

SHINRA — собственный decoder-only Transformer NULLXES: своя архитектура, свой токенизатор, свой training pipeline. Это не fine-tune чужого веса и не обёртка над Llama / Mistral / Qwen / GPT-NeoX.

Модель: **NULLXES SHINRA-4B** · ~**3.93B** параметров · vocab **131072** · контекст **8192** (нативное окно **32768**) · **BF16** · кластер **8× A100 80GB**.

Выходы стадий:

1. `NULLXES SHINRA-4B-BASE` — претрейн  
2. `NULLXES SHINRA-4B-INSTRUCT` — instruction tuning  
3. aligned instruct — DPO / preference optimization  

Загрузка:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "NULLXES/SHINRA-4B-INSTRUCT",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    device_map="auto",
)
```

---

## Репозиторий

```
architecture/     спецификация и подсчёт параметров
model/            ShinraConfig, блоки, GQA+RoPE, SwiGLU, HF Auto*
tokenizer/        SentencePiece Unigram 131072 + chat template
data/             mix, cleaning, MinHash, packing, SFT/DPO, **pilot builder**
training/         pretrain / SFT / DPO, FSDP, WSD, fused AdamW
evaluation/       PPL, lm-eval, needle, code/multilingual slices
inference/        generate, chat, OpenAI HTTP
runtime/          vLLM / SGLang / TokenSpeed **serving** (не обучение)
models/           registry имён, без весов
configs/          shinra_4b.yaml, dataset_pilot.yaml, Colab 100M, A100
scripts/          команды кластера и публикация на Hub
docs/             status, architecture, data, tokenizer, training, hub, model card
notebooks/        SHINRA_COLAB.ipynb
```

---

## Архитектура SHINRA CORE

```
Embedding
  → [ RMSNorm → GQA+RoPE (+QK-norm) → residual
      RMSNorm → SwiGLU              → residual ] × 32
  → RMSNorm → LM Head (tied)
```

| | |
|---|---|
| hidden | 3072 |
| layers | 32 |
| heads / KV | 24 / 8 |
| head dim | 128 |
| SwiGLU | 9216 |
| RoPE θ | 1 000 000 |
| RMSNorm ε | 1e-6 |
| QK-norm | да |
| bias | нет |
| Z-loss | 1e-5 |

Подсчёт: `python -m architecture.param_count`  
Спека: [`architecture/design.md`](architecture/design.md)

---

## Токенизатор

SentencePiece **Unigram**, 131072, NFKC, byte fallback.

Спецтокены: `<|system|>` `<|user|>` `<|assistant|>` `<|eot|>` `<|code|>` `<|language|>` `<|reasoning|>` `<|tool_call|>` `<|tool_response|>` `<|document|>` `<|end_of_text|>`

```bash
bash scripts/train_tokenizer.sh /data/tokmix/web /data/tokmix/wiki /data/tokmix/code
```

Артефакты: `tokenizer/artifacts/{tokenizer.model,tokenizer.json,tokenizer_config.json,tokenizer_stats.json}`

---

## Данные

```bash
pip install -e .
export PYTHONPATH=.
bash scripts/prepare_data.sh
```

Пайплайн: normalize → quality → language → toxicity → code quality → MinHash dedup → pack 8192.

Источники и веса: [`configs/data_mix.yaml`](configs/data_mix.yaml), [`docs/data.md`](docs/data.md).

---

## Обучение на A100

Локальная RTX 2080 не цель: 4B + AdamW туда не встают. Bring-up — [Google Colab A100](docs/colab.md) / `notebooks/SHINRA_COLAB.ipynb`. Претрейн — 8× A100 80GB.

```bash
pip install -e ".[train,flash]"
bash scripts/train_pretrain.sh
bash scripts/train_sft.sh outputs/shinra-4b-base/final/step-00095367
bash scripts/train_dpo.sh  outputs/shinra-4b-instruct/final/step-00008000
```

Конфиг модели и кластера: [`configs/shinra_4b.yaml`](configs/shinra_4b.yaml)  
Accelerate: [`configs/accelerate_a100.yaml`](configs/accelerate_a100.yaml)

Претрейн: 200B токенов, 2 097 152 ток/шаг, LR 3e-4, WSD, FSDP FULL_SHARD, gradient checkpointing, fused AdamW, SDPA Flash на A100.

---

## Оценка и инференс

```bash
python -m evaluation.perplexity --model $CKPT --data-dir data/packed/pretrain
python -m evaluation.harness --model $CKPT
python -m evaluation.needle --model $CKPT
python -m inference.generate --model $CKPT --prompt "Write RMSNorm in PyTorch."
python -m inference.server --model $CKPT --port 8000
```

Serving engines (vLLM / SGLang / TokenSpeed) — [`runtime/`](runtime/README.md). Это не обучение.

---

## Hugging Face Hub

```bash
export HF_TOKEN=hf_...
python -m scripts.publish_hub \
  --checkpoint outputs/shinra-4b-instruct/final/step-00008000 \
  --repo-id NULLXES/SHINRA-4B-INSTRUCT \
  --private
```

Model card: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)

---

## Лицензия

NULLXES Research License. Веса — proprietary asset. См. [`LICENSE`](LICENSE).
