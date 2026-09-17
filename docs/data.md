# Данные SHINRA

## Phase 0 — SHINRA-COLAB-PILOT

Не озеро на 1.3T. Стриминг, диск <150GB.

- general 50% — `HuggingFaceFW/fineweb-edu` **`sample-10BT`**
- code 25% — python-edu + The Stack smol
- math_stem 15% — OpenWebMath
- multilingual 10% — Wikipedia ru/zh
- language quotas: en 70 / ru 15 / zh 10 / other 5

Сессия Colab: `--max-tokens 100000000`. Спека корпуса: 5B. Команды: `python -m data.build_pilot`.

## Production mix v1 (не исполнять в Colab)

40% web / 20% code / 15% math-science / 10% books+wiki / 10% multilingual / 5% NULLXES domain.

## Претрейн-рецепт в репо (200B, кластер)

| Домен | Доля | Источники |
|-------|------|-----------|
| Качественный веб | 0.42 | FineWeb-Edu |
| Энциклопедия | 0.09 | Wikipedia en/ru (+ другие дампы по мере ingest) |
| Книги | 0.05 | PG19 / Gutenberg |
| Образовательный синтетический | 0.07 | SmolLM Cosmopedia-v2 |
| Код | 0.18 | SmolLM Python-edu + The Stack (permissive licenses) |
| Math / STEM | 0.10 | OpenWebMath, ProofPile-2 |
| Наука | 0.07 | peS2o, arXiv CS |
| Narrative aux | 0.02 | короткоформатная проза как регуляризация |

Веса в `configs/data_mix.yaml` и `data/sources.py`.

| Домен | Доля | Источники |
|-------|------|-----------|
| Качественный веб | 0.42 | FineWeb-Edu |
| Энциклопедия | 0.09 | Wikipedia en/ru (+ другие дампы по мере ingest) |
| Книги | 0.05 | PG19 / Gutenberg |
| Образовательный синтетический | 0.07 | SmolLM Cosmopedia-v2 |
| Код | 0.18 | SmolLM Python-edu + The Stack (permissive licenses) |
| Math / STEM | 0.10 | OpenWebMath, ProofPile-2 |
| Наука | 0.07 | peS2o, arXiv CS |
| Narrative aux | 0.02 | короткоформатная проза как регуляризация |

Веса в `configs/data_mix.yaml` и `data/sources.py`.

## Очистка (`data/clean.py`)

1. `ftfy` + NUL strip  
2. Gopher/FineWeb quality (длина, mean word length, symbol/alpha, дубли строк, zlib ratio)  
3. Language / script gate  
4. Toxicity lexicon + patterns (+ optional HF classifier)  
5. Code: AST для Python, анти-minify, generated-file drop  
6. Exact SHA256 + MinHash LSH (128 perm, Jaccard 0.80)

## Packing

Документы пакуются в окна 8192 как `<|bos|><|document|>…<|end_of_text|>`. SFT маскирует loss на токенах ассистента до `<|eot|>`. DPO хранит `prompt/chosen/rejected`.

```bash
bash scripts/prepare_data.sh
# или с лимитом документов на источник:
MAX_DOCS=10000 bash scripts/prepare_data.sh
```
