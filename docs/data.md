# Данные SHINRA

## Претрейн (200B токенов)

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
