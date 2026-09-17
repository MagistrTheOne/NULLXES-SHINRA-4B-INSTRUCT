# Данные SHINRA

Архитектура заморожена. Этот файл — рабочий корпус, не lock на 200B.

## Phase 0 — SHINRA-COLAB-PILOT

Не озеро на 1.3T. Стриминг, диск <150GB. Диагностика tokenizer DNA, не BASE.

| Бакет | Доля | Источники |
|-------|------|-----------|
| general | 0.50 | FineWeb-Edu `sample-10BT` |
| code | 0.25 | python-edu 0.15 + The Stack smol 0.10 |
| math_stem | 0.15 | OpenWebMath |
| multilingual | 0.10 | Wikipedia ru 0.07 / de 0.015 / fr 0.015 |

Языковые квоты пилота (и BASE): **en 75 / ru 15 / eu 10**. Китайских дампов нет. TinyStories нет.

Сессия Colab: `--max-tokens 100000000`. Спека корпуса: 5B.

```bash
python -m data.sources                          # аудит весов = 1.0
python -m data.build_pilot --plan               # mix vs DNA, без HF stream
python -m data.build_pilot --max-tokens 100000000
python -m tokenizer.analyze_tokenizer --corpus tokenizer/corpus --parquet-dir data/clean/pilot
```

`build_pilot` пишет `pilot_report.json` (факт vs квоты) и `tokenizer_friendship.json` (131k DNA, скрипты, утечки `<|...|>`, unk/bytes-per-token если артефакты уже есть).

## SHINRA-4B-BASE v1 (кластер, не Colab)

Не финальный lock. Рецепт после ревью микса:

| Домен | Доля | Источники |
|-------|------|-----------|
| high_quality_web | 0.40 | FineWeb-Edu `sample-10BT` (кластер может поднять до `sample-100BT`) |
| code | 0.20 | python-edu 0.10 + licensed The Stack 0.10 (код + docs, не сырой `pass`) |
| math_science | 0.15 | OpenWebMath 0.07, ProofPile-2 0.02, peS2o 0.03, arXiv **полный текст** 0.03 |
| books | 0.10 | PG19 0.07 + Gutenberg 0.03. TinyStories исключён |
| multilingual | 0.10 | Wikipedia ru/de/fr. Английская энциклопедия закрывается FineWeb, не отдельным EN wiki |
| nullxes_engineering | 0.05 | arXiv CS/robotics/CUDA keyword slice 0.03 + stack `.md/.rst/.cu` 0.02 |

Язык BASE: en 75 / ru 15 / other 10.

Исключено: TinyStories, Qwen corpora, Chinese instruction dumps, FineWeb-Edu 1.3T, abstract-only arXiv.

Веса: `configs/data_mix.yaml`, `data/sources.py`.

## SFT v1 — ядро, не «ещё один чатик»

| Бакет | Доля | Источники |
|-------|------|-----------|
| conversation | 0.30 | Tulu 0.18 + SmolTalk 0.12 |
| code | 0.25 | CodeFeedback 0.15 + Magicoder Evol 0.10 |
| math_reasoning | 0.20 | OpenMathInstruct-2 `train_1M` |
| agent_tools | 0.15 | Hermes function-calling 0.10 + JSON-mode 0.05 (`<|tool_call|>`, structured outputs) |
| general | 0.10 | OpenHermes-2.5 |

Salesforce xLAM gated — в пайплайне не используется.

## DPO v1 — поведение

| Бакет | Доля | Источники |
|-------|------|-----------|
| general | 0.40 | UltraFeedback |
| code | 0.30 | `jondurbin/py-dpo-v0.1` |
| tools | 0.20 | Argilla dpo-mix-7k (пока нет отдельного tool-preference дампа) |
| formatting | 0.10 | Intel Orca DPO pairs |

## Очистка (`data/clean.py`)

1. `ftfy` + NUL strip
2. Gopher/FineWeb quality (длина, mean word length, symbol/alpha, дубли строк, zlib ratio)
3. Language / script gate; wiki ru/de/fr фиксируют язык источника
4. Toxicity lexicon + patterns
5. Code: AST для Python, анти-minify; NULLXES stack режет по `.md/.rst/.cu`
6. Exact SHA256 + MinHash LSH (128 perm, Jaccard 0.80)
7. NULLXES arXiv идёт **раньше** общего arXiv, чтобы keyword-hit не съел dedup

## Packing

Претрейн: `<|bos|><|document|>…<|end_of_text|>`. SFT маскирует loss на токенах ассистента до `<|eot|>`. DPO хранит `prompt/chosen/rejected`.

```bash
bash scripts/prepare_data.sh
# или с лимитом документов на источник:
MAX_DOCS=10000 bash scripts/prepare_data.sh
```
