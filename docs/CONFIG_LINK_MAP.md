# Карта ссылок конфигов (статика, до удаления — удаление отдельным commit)

## Активная цепочка после очистки

```text
scripts/v2_stage_run.py
  --config configs/architecture_v2.yaml        (геометрия; default)
  --stage-config configs/stages/d_en_ru_pilot.yaml  (рецепт пилота, C489, BLOCKED)
  --runtime-config configs/runtime_g4.yaml    (согласование attention, NEW)
  --storage-config configs/storage_g4.yaml    (drum/лимиты, сохранён)
  --corpus-dir / --tokenizer / --output-dir
 → data/rolling_drum.py::run_rolling_stage (+guard d_en_ru_pilot, runtime-проверка, resolved save)
 → training/arguments.py::build_train_config (default --config architecture_v2)
 → training/trainer.py::config_from_yaml/load_model/run_lm_training (+guard от silent random)
 → model/configuration_shinra.py::from_yaml (architecture_v2, legacy fallback shinra_4b)
```

## Карта: конфиг → загрузчики → entrypoints → notebooks → документация

| Конфиг | Загрузчики (код) | Entrypoints/notebooks | Доки/тесты | Решение |
|---|---|---|---|---|
| `configs/architecture_v2.yaml` (NEW) | `training/arguments.py` (default), `training/trainer.py:config_from_yaml`, `model/configuration_shinra.py:from_yaml` | `scripts/v2_stage_run.py` (default), `scripts/phase01_bringup.py` (default) | `docs/C489_LINK.md`, `docs/REVIEW_CONFIG_CHAIN.md` | Активный архитектурный файл |
| `configs/runtime_g4.yaml` (NEW) | `data/rolling_drum.py` (проверка attention + resolved save) | `scripts/v2_stage_run.py --runtime-config` | — | Активный runtime-файл |
| `configs/stages/d_en_ru_pilot.yaml` | `data/rolling_drum.py` (+BLOCK guard), `training/trainer.py:load_model` (+BLOCK guard) | только `v2_stage_run.py` вручную после верификации C489 | `data/specs/V2_UPGRADE_DATA_SPEC.md`, `docs/C489_LINK.md` | Активный stage, запуск заблокирован кодом |
| `configs/shinra_4b.yaml` (удалён; история в git) | `training/arguments.py` (default → `architecture_v2.yaml`), `model/configuration_shinra.py` (строгий путь, без legacy fallback) | `scripts/v2_s0_colab.py`, `scripts/v2_c_colab.py`, `scripts/phase01_bringup.py` (переведены на `architecture_v2.yaml`; S0/C ran with identical geometry) | `docs/training.md`, `docs/status.md`, `tests/test_architecture_v2.py`, `tests/test_shinra_invariant.py` (переведены) | Заменён: `architecture_v2.yaml` + `runtime_g4.yaml` + stage |
| `configs/sft_a100.yaml` | `training/sft.py` (`set_defaults train_config`), `scripts/train_sft.sh` | `scripts/train_sft.sh` | `docs/training.md` | Убрать из активного дерева (архив A100/SFT, явный opt-in); SFT не в плане D |
| `configs/storage_g4.yaml` | `data/rolling_drum.py` (`storage.disk_ceiling_gb`), `tests/test_v2_rolling_drum.py` | `scripts/v2_stage_run.py --storage-config`, `scripts/v2_s0_colab.py`, `scripts/v2_c_colab.py` | — | СОХРАНИТЬ: drum-правила, пути, лимиты диска/сохранения; отдельный storage-конфиг нужен |
| `configs/tokenizer.yaml` | Кодом обучения tokenizer напрямую не читается как train-конфиг (`tokenizer/train_tokenizer.py` — CLI-аргументы); назначение — декларация DNA/acceptance | — | `tokenizer/special_tokens.py` (канон ID) | Назначение установлено: декларация замороженной DNA + acceptance (`unk_rate_max`, `bytes_per_token_min`). Из активного train-пути убрать; `tokenizer.json/model/config/special_tokens`, их ID и привязку к C489 сохранить (артефакты + `docs/C489_LINK.md`) |
| `configs/stages/s0_bringup.yaml`, `s1_language.yaml`, `s2_semantic.yaml`, `c_edu_en_pilot.yaml`, `pretrain_colab_100m.yaml`, `pretrain_a100.yaml`, `dpo_a100.yaml`, `data_mix.yaml`, `dataset_pilot.yaml` | `v2_s0_colab/v2_c_colab` (S0/C история), тесты `test_v2_ledger_shards`, `test_data_v1_phase_*` | notebooks S0 | `data/specs/*`, `eval/specs/SHINRA_PROBE_V1.md` | Старые стадии/пилоты: убрать из активного дерева после проверки ссылок; историю S0/C и receipts сохранить как происхождение checkpoint |

## training/ (включая скрытое .gitignore)

- Активны: `training/arguments.py`, `training/trainer.py`, `training/collator.py`, `training/optim.py`, `training/schedule.py`.
- Архив/не-D: `training/pretrain.py`, `training/sft.py`, `training/dpo.py` (требуют явный `--train-config`; A100-рецепты удалены), `training/s1_p0_objective.py`.
- Игнорируемые артефакты (`.gitignore`: `*.pt/*.bin/*.safetensors/checkpoints/`, `data/packed/`, `evaluation/results/`): веса, packed-шарды, результаты eval — не удалять как пользовательские данные; в ревью не входят.

## Удаление — отдельным commit (не в этом diff)

Удалены (миграция выше выполнена):
1. `configs/shinra_4b.yaml` → `architecture_v2.yaml` + `runtime_g4.yaml` + stage
   (геометрия идентична: `2560/9728/36/32Q/8KV/128/131072`, мертвые поля
   `train_sequence_length/sliding_window/layer_types/use_cache` удалены как метадата).
2. `configs/pretrain_a100.yaml`, `sft_a100.yaml`, `dpo_a100.yaml` →
   архивные `training/pretrain.py/sft.py/dpo.py` (требуют явный `--train-config`).
3. `configs/pretrain_colab_100m.yaml`, `data_mix.yaml`, `dataset_pilot.yaml`,
   `tokenizer.yaml` → замены: V2 spec (§2/§4), `tokenizer/special_tokens.py`.
4. `scripts/train_pretrain.sh`, `train_sft.sh`, `train_dpo.sh`,
   `notebooks/SHINRA_COLAB.ipynb` → superseded by G4 drum (`v2_stage_run.py`,
   `SHINRA_V2_S0.ipynb`).
- Не удалять: untracked пользовательские файлы, веса, исходные данные, tokenizer-артефакты, receipts S0/C.
