# Ревью цепочки конфигов SHINRA V2 (статика, без запусков)

Дата: 2026-09-29. Ветка: `prep/v2-upgrade-data-eval` (подготовка).
Статус: ревью, не сертификат исправности.

## 1. Проверенная цепочка

```text
scripts/v2_c_colab.py
 → scripts/v2_stage_run.py
 → data/rolling_drum.py::run_rolling_stage / _train_ns
 → training/arguments.py::build_train_config
 → training/trainer.py::config_from_yaml / load_model / run_lm_training
```

Приоритет: CLI (`--max-tokens`, `--max-steps`, `--sequence-length`, `--attention-implementation`, …)
> stage YAML (`configs/stages/*.yaml`)
> root YAML (`--config`, сейчас `configs/shinra_4b.yaml`).

## 2. Что реально доходит до модели и trainer (по коду)

| Поле `shinra_4b.yaml` | Значение в файле | Факт по коду | Вывод |
|---|---|---|---|
| `model.train_sequence_length: 8192` | 8192 | Чтений в `*.py` нет (grep: только сам YAML). Длина берётся из `batch.sequence_length` stage через `build_train_config` (`training/arguments.py:124`) | Мёртвая метадата. Для C/S0/S1/S2 факт 2048. Удалить из архитектурного файла |
| `batch` (8 GPU, micro 2, accum 16, `tokens_per_step 2097152`) | кластер A100 | `batch = extra.get("batch", root.get("batch"))` (`training/arguments.py:89`) — stage целиком перекрывает root при наличии. Stage C/S0/S1/S2: `1 GPU, micro 1, accum 8, tokens_per_step 16384` | Для G4-барабана факт 16384. Разница 128× опасна только при неполном override — зафиксировать правило «stage batch полный, без частичных слияний» |
| `tokens_per_step` | 2097152 (root) | Используется только для расчёта `max_steps = max_tokens / tokens_per_step` (`training/arguments.py:94-99`). При stage batch факт 16384 | При CLI `--max-steps` приоритет у CLI. В resolved-конфиге сохранять оба числа |
| `scheduler.warmup_steps: 2000` | 2000 (root) | `sch = extra.get("scheduler", root.get("scheduler"))` (`training/arguments.py:91`). Stage C warmup 40, S0 warmup 80 | C489 при корректном пути шёл с warmup 40, не 2000. Весь C внутри warmup был бы только при ошибке выбора конфига — это и надо исключить resolved-конфигом |
| `training.max_tokens: 200B` | 200000000000 | `max_tokens = args.max_tokens / train.max_tokens / extra.max_tokens / 200B` (`training/arguments.py:95-98`). Stage C 8M + CLI `--max-tokens 8000000` (`scripts/v2_c_colab.py`) | Факт C 8M, ceiling 10M (`data/data_v1/phase_c.py`). Два источника лимита — CLI и stage — задокументировать приоритет CLI |
| `model.use_cache: true` | true | Хардкод `use_cache=False` в `config_from_yaml` (`training/trainer.py:63`), `load_model` (`training/trainer.py:86`), все train/eval вызовы `use_cache=False` | Для обучения факт false. YAML true — мусор для train-пути |
| `eos_token_id: 2` / `document_end_token_id: 18` | 2 / 18 | Оба поля доходят до `ShinraConfig` (`training/trainer.py:59-60`). Автопереключения генерации на 18 нет — `inference/generate.py` берёт `tokenizer.eos_token_id` | Continuation-eval обязан явно задавать `eos_token_id=18`, `use_cache=false` |
| `hardware: 8×A100, flash:true` | A100 | `training/arguments.py` раздел `hardware` не читает (кроме `storage.disk_ceiling_gb`). Факт attention: CLI/stage `sdpa` (`training/arguments.py:112-116`, гейты `scripts/v2_s0_colab.py`, `scripts/v2_c_colab.py`) | Для G4 факт SDPA на одном Blackwell. A100-блок — архив, не активная настройка |
| `sliding_window / layer_types / initializer_range / train_sequence_length` | null/null/0.02/8192 | Белый список `config_from_yaml` (`training/trainer.py:37-64`) их не читает | Не доходят до модели путём trainer. Либо добавить явно, либо удалить из архитектурного файла |

## 3. Что ошибкой не является

- 2560 residual / 4096 Q — допустимая геометрия, прямой аналог Qwen3-4B (`2560/36/32Q/8KV/128/9728/tied`).
- SDPA vs Flash: SDPA может выбирать Flash backend; `flash_attention: true` в архивном блоке не доказывает внешний FA2.
- Dropout 0 — не дефект.
- EOS 2 + document-end 18 — допустимы при явном режиме генерации.

## 4. Решение по структуре (принято в план)

1. `configs/architecture_v2.yaml` — только геометрия.
2. `configs/runtime_g4.yaml` — precision, железо, исполнение.
3. Stage-пилот — данные, batch, optimizer, scheduler, лимиты.
4. Старый A100/FSDP — отдельный явно выбираемый конфиг.
5. При запуске сохранять resolved-конфиг после всех переопределений рядом с checkpoint.

## 5. Полный аудит репозитория (обход без исполнения)

Проверены чтением: `.gitignore`, `LICENSE`, `pyproject.toml`, `README.md`, `requirements.txt`
(torch отсутствует корректно — среда G4, не pip), `architecture/`, `configs/`,
`data/` (включая `data/data_v1/`, `data/specs/`), `docs/`, `eval/`, `evaluation/`,
`inference/`, `model/` (`configuration_shinra.py`, `modeling_shinra.py`,
`modeling_attn.py`, `modeling_mlp.py`, `modeling_norm.py`), `models/`,
`notebooks/`, `runtime/`, `scripts/`, `tests/`, `tokenizer/`, `training/`.
Исполнение, установка зависимостей, загрузка весов/корпусов не выполнялись.
