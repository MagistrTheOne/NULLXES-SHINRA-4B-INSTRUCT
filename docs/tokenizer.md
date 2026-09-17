# Токенизатор NULLXES SHINRA

Собственный SentencePiece Unigram, vocab **131072**, NFKC, byte fallback, split digits.

**Заморожен до претрейна.** Переименование control-токенов после обучения BASE невозможно без порчи DNA.

## Control plane

| ID | Токен | Роль |
|----|--------|------|
| 0 | `<\|unk\|>` | unknown (минимум, byte fallback) |
| 1 | `<\|bos\|>` | start of sequence, всегда в chat template |
| 2 | `<\|eot\|>` | **end of turn** = HF `eos_token`, стоп генерации |
| 3 | `<\|pad\|>` | padding |
| 4+ | user-defined | ниже |

`<|end_of_text|>` — конец **документа** в претрейне. Это не turn-stop.

`<|tool|>` **нет**. Вызов и ответ инструмента — разные токены.

```
<|unk|> <|bos|> <|eot|> <|pad|>
<|system|> <|user|> <|assistant|> <|reasoning|>
<|code|> <|language|>
<|fim_prefix|> <|fim_suffix|> <|fim_middle|>
<|repo|> <|file|>
<|tool_call|> <|tool_response|>
<|document|> <|end_of_text|>
```

Chat template: `tokenizer/chat_template.jinja` (BOS + JSON `tool_call` + language на code).

## Reasoning policy: latent

`<|reasoning|>` зарезервирован. SHINRA-4B-INSTRUCT **не** учится публичному chain-of-thought по умолчанию. Поле `reasoning` в SFT попадает в шаблон только если оно реально есть в distillation / alignment данных.

## Tool calls

```
<|assistant|>
<|tool_call|>{"name":"search","arguments":{"q":"..."}}<|eot|>
<|tool_response|>...json...<|eot|>
```

## Pretrain packing

```
<|bos|><|document|> ... text ... <|end_of_text|>
```

## Обучение

Корпус ≥ 10B символов, пропорциональный претрейн-миксу.

```bash
bash scripts/train_tokenizer.sh /data/tokmix
```

Артефакты в `tokenizer/artifacts/`:

- `tokenizer.model` — SentencePiece
- `tokenizer.json` — Hugging Face fast
- `tokenizer_config.json`, `special_tokens_map.json`
- `chat_template.jinja`
- `corpus_stats.json`, `tokenizer_stats.json`

Критерии приёмки: `unk_rate < 0.1%` на holdout, `bytes/token ≳ 3.5`, все control-токены с id ≠ unk.
