# C489 link — checkpoint → код → tokenizer → данные → обучение (малые файлы)

Статус: чеклист для Drive-артефактов. `trainer_state.pt` вслепую не десериализовать:
может содержать тяжёлое состояние optimizer. Сначала формат и метаданные.

## Требуемая связка (все SHA + commit)

- [ ] `s0/final/step-00001358` — `config.json` SHA, геометрия V2 `2560/36/32Q/8KV/128/9728/tied`
- [ ] `c` checkpoint C489 — `config.json` SHA совпадает с S0-геометрией
- [ ] `tokenizer/artifacts` — `tokenizer.json/model/config` SHA, vocab 131072, ID `0/1/2/3/18`
- [ ] код — git commit SHA ветки и `configs/stages/c_edu_en_pilot.yaml` SHA
- [ ] данные — `phase_c.pack.json` + shard `shard-00000.bin` SHA, `honest_tokens`, `n_sequences`,
      `source_id fineweb-edu-en`, граница потреблённых документов
- [ ] обучение — `run/status.json`, `run/cursor.json`, ledger `consumed_tokens/step`,
      `optimizer/scheduler` наличие и согласованность (без загрузки PT вслепую)
- [ ] resolved-конфиг C — CLI > stage > root значения: seq 2048, micro 1, accum 8,
      LR 3e-4, warmup 40, cap 8M/ceiling 10M, `use_cache=false`, attention `sdpa`

## Формат сначала

1. `ls -l` + `sha256sum` малых файлов.
2. `config.json` diff против `configs/architecture_v2.yaml`.
3. `tokenizer.json` первые байты / `tokenizer_config.json` — только метаданные.
4. `trainer_state.pt` — `torch.load(map_location=cpu, weights_only=True)` мета-обход
   или внешний парсер заголовка; полный `weights_only=False` запрещён до ревью формата.
5. Результат — заполненный чеклист выше, не продолжение обучения.

C489 остаётся референсом. Пилот Stage D стартует с нулевых весов.
