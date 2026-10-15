# NULLXES — SHINRA-4B-INSTRUCT

**SHINRA is SHINRA. Created by NULLXES.**

Собственная модель NULLXES с random initialization. Единственная конфигурация —
[`configs/shinra.yaml`](configs/shinra.yaml). Основной код: `model/`, `tokenizer/`,
`data/`, `training/`, `evaluation/`, `inference/`.

Текущий этап — подготовка базового pretraining. Instruction tuning — будущий этап
этой же кодовой базы. 20M токенов — диагностический бюджет, не полный pretraining.
Предварительный словарь — 65 536; специальные ID 0–18 сохранены.

## Статус

Обученного и утверждённого tokenizer bundle нет. Training и Colab отключены.
Исправления Tokenizer Audit проверены только статически; runtime-измерений нет.
Старые экспериментальные веса и токенизаторы не подходят для нового запуска.

- [Контракт токенизатора](docs/tokenizer.md)
- [Отчёт исправлений](docs/TOKENIZER_AUDIT_FIXES.md)
- [Подготовка данных](docs/data.md)
- [Packing](docs/tokenizer_pretrain_contract.md)
- [Обучение и resume](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Inference](docs/inference.md)

Никакого запуска без отдельного разрешения Маги. Зависимости ML предоставляет
подготовленное окружение; проект не устанавливает и не обновляет torch/transformers.
