# NULLXES — SHINRA-4B-INSTRUCT

**SHINRA is SHINRA. Created by NULLXES.**

Собственная модель NULLXES с random initialization. Единственная конфигурация —
[`configs/shinra.yaml`](configs/shinra.yaml). Основной код: `model/`, `tokenizer/`,
`data/`, `training/`, `evaluation/`, `inference/`.

Текущий этап — реализация NULLXES Synthetic Engine и native tokenizer pipeline.
Внешние текстовые наборы и ответы сторонних LLM не допускаются в новых данных.
Словарь 65 536 и специальные ID 0–18 остаются контрактом; модель tokenizer и
bundle SHA не закрепляются до удалённой сборки, review и измеренных проверок.

## Статус

Обученного и утверждённого tokenizer bundle нет. Training, Colab и иные удалённые
runtime не подтверждены. Код этапа Synthetic Engine ещё требует запуска и проверки
в выбранной удалённой среде; никакие результаты генерации, tokenizer или обучения
не заявлены. Старые корпуса/артефакты не инвентаризированы за пределами этой копии
и не удалялись. Старые source-based entrypoints выведены из установленной CLI.

- [План этапов и TODO](docs/ROADMAP_SYNTHETIC_NATIVE.md)
- [Статический аудит и пределы доказательств](docs/AUDIT_SYNTHETIC_NATIVE.md)

- [Контракт токенизатора](docs/tokenizer.md)
- [Отчёт исправлений](docs/TOKENIZER_AUDIT_FIXES.md)
- [Подготовка данных](docs/data.md)
- [Packing](docs/tokenizer_pretrain_contract.md)
- [Обучение и resume](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Inference](docs/inference.md)

Локальный запуск, тесты, tokenizer build, генерация и обучение не выполнялись.
Colab, Runpod и Google Cloud не объявлены поддержанными: в следующем этапе надо
отдельно выбрать и подтвердить одну целевую среду.
