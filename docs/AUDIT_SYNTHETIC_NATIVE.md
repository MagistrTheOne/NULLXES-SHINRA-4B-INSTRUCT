# SHINRA Native 4B — стартовый аудит и инвентаризация

**Дата:** 2026-10-19  
**Режим:** статическое чтение файлов и Git-индекса; локальные тесты, импорты, генерация, tokenizer build, обучение и runtime не запускались.  
**Приёмка:** `NOT VERIFIED`; документ фиксирует только увиденное в этой рабочей копии.

## Решение по контуру

SHINRA-4B-INSTRUCT остаётся независимой Native-моделью с random initialization. Веса других моделей, включая Qwen, не являются источником инициализации, teacher outputs или обучения SHINRA. Новый корпус должен исходить только от NULLXES Synthetic Engine с проверяемыми генераторами и верификаторами. SHINRA V3.2 в область работ не входит.

`configs/shinra.yaml` остаётся единственным авторитетным YAML. До P0 training flags и runtime launch остаются закрытыми. Эта проверка не подтверждает среду для Colab, Runpod или Google Cloud.

## Инвентаризация этой рабочей копии

- Git-индекс содержит исходники model/tokenizer/data/training/evaluation/inference, конфигурацию, документацию и тестовые файлы; обученные веса и corpus shards в Git-индексе не обнаружены.
- В исходном состоянии `configs/shinra.yaml` перечислял Wikimedia, PleIAs Russian-PD, FineWeb-Edu, DCLM-Baseline-1.0 и FineWeb-2. Активная конфигурация теперь использует `nullxes_synthetic_engine`; старые модули и исторические данные не удалялись. Удалённые хранилища не проверялись.
- `data/sources.py` и `data/build_corpus.py` сохраняют старый acquisition путь в репозитории, но их console entrypoints удалены. Историческая документация по external data снабжена предупреждением и не является инструкцией активного pipeline.
- В `data/specs` в этой копии не обнаружены файлы. При целевом осмотре `data` и `tokenizer` не найдено каталогов raw/clean/packed, tokenizer model/bundle, JSONL/Parquet корпуса или checkpoint-файлов. В `tokenizer/__pycache__` есть одиночный `corpus_friendship.cpython-311.pyc`; это исторический локальный bytecode, не обучающий корпус. Не удалён.
- `.gitignore` исключает `data/raw`, `data/clean`, `data/packed`, `tokenizer/artifacts`, checkpoints и training outputs. Поэтому отсутствие таких файлов в Git-индексе не доказывает отсутствие исторических артефактов в других каталогах, на Drive, Colab или иных машинах. Удалённые/облачные хранилища в этом этапе не инвентаризированы.
- Рабочая копия была на `main`, синхронизированной с `origin/main`, с одним untracked планом `docs/ROADMAP_SYNTHETIC_NATIVE.md`. Других локальных изменений на момент начала аудита не было.

## Реализация против спецификации

| Область | Наблюдаемое состояние | Следующий допуск |
|---|---|---|
| Native архитектура | Кастомные `ShinraConfig`, decoder, GQA, RoPE, SwiGLU, RMSNorm и SDPA-код присутствуют; геометрия задаётся YAML. | Не менять геометрию до архитектурного аудита; runtime численная корректность и memory peak не подтверждены. |
| Loss / память | Training path использует chunked time projection/CE и не возвращает logits; временный logit tensor ограничен `[batch, loss_chunk_size, vocab]`. Математическая/градиентная эквивалентность и фактический peak не проверялись. Старая оценка 131 072 не соответствует текущему YAML vocab=65 536. | Сверить loss и градиенты с эталоном, измерить peak; отдельно решить vocab/4B параметризацию на удалённой runtime-среде. |
| Attention | SDPA и отдельные eager/FlashAttention ветви присутствуют. Конфиг выбирает SDPA. | Сверить SDPA causal/padding/GQA/cache paths с эталоном; FlashAttention оставить недопущенным до сравнения. |
| Tokenizer | Builder переведён на Native SentencePiece. При пустом модельном SHA он сохраняет только `candidate_not_frozen`; финальный bundle требует review и SHA freeze в YAML. Реального bundle и fertility/round-trip результатов нет. | Candidate и финальный bundle требуют удалённой сборки и сохранённого evidence. |
| Corpus | Реализованы авторские арифметика, mode transition, inventory и координатные задачи; provenance, answer verifiers, manifest, family split gate, structural exact/near audit и synthetic packing. | Генерация не запускалась; функционально-эквивалентный поиск узкий, near-duplicate cross-split сравнение pairwise и квадратично. Старые source-based модули сохранены, их CLI исключены из активной установки. |
| Split integrity | Новый YAML разводит family/template/scenario/functional-family до генерации; код проверяет structural exact/near duplicates. Эти проверки не исполнялись, cross-split functional equivalence остаётся узкой и pairwise audit квадратичен. | Расширить эквивалентность, измерить масштабируемость и получить удалённый evidence на реальных split-файлах. |
| Training | Однопроцессный trainer, optimizer, sampler и checkpoint/resume код присутствуют; gates закрыты. Synthetic packer выпускает только `audit_only`, а trainer P1 production path требует `production`; диагностический режим дополнительно закрыт до P0 evidence. | Сейчас ни одного admitted training path нет. После закрытия P0 определить разрешённый admission, затем подтверждать run на выбранной удалённой машине. |
| Chat / intelligence | Inference/chat контур и EOT contract присутствуют; instruction tuning обозначен будущим этапом. | SFT, отмена/отрицания, object-state, grounding и action-sequencing не подтверждены; определить отдельно для P2. |
| Runtime | Trainer проверяет `/content` и GPU allowlist; YAML training/Colab flags выключены. | Не заявлять Runpod/Google Cloud совместимыми. Перед удалённым P1 выбрать и подтвердить одну среду. |
| Architecture identity | Текущий YAML задаёт 65 536 vocab; ранее встречавшийся 131 072 contract удалён из активной геометрии. Возврат к 131 072 изменит размер embedding/output и параметризацию. | Не менять без отдельного architecture review; зафиксировать текущий vocab как незакрытый проектный риск. |
| Документация | Часть актуальных статусов противоречит реализации, например `CONSOLIDATION.md` отрицает уже присутствующую очистку markup/quarantine. | Обновлять status после перехода, сохраняя distinction `IMPLEMENTED` / `RUNTIME VERIFIED` / `ACCEPTED` / `UNAVAILABLE`. |

## Выводы и безопасные решения

1. Активная политика и CLI entrypoints переведены на synthetic-only; исходные source modules и артефакты сохранены.
2. Synthetic Engine имеет собственную схему, deterministic seed derivation и verifiers. Packed provenance названа `corpus_manifest`, а не source manifest.
3. Family isolation требует формального уровня группировки. `family_id`, template family и scenario family должны быть назначены до split; параметры/имена внутри одной функции не создают независимую blind задачу.
4. До появления реального synthetic corpus tokenizer нельзя обучить на авторских данных или честно оценить на них. До появления bundle его hash остаётся unset.
5. Технический P0/P1 результат не является оценкой Model Intelligence. Chat/SFT и blind benchmark остаются отдельным P2.

## Что этот аудит не устанавливает

- Не проверено содержимое внешних дисков, Google Drive, Colab, Runpod, Google Cloud или иных удалённых машин.
- Не устанавливалось, существуют ли копии старого корпуса вне проверенных путей этой рабочей копии.
- Не запускались тесты, Python imports, генерация данных, tokenizer build, forward/backward, GPU memory checks, обучение или resume.
- Не получены benchmark scores, memory peak, throughput и model capability results.
- Не выполнялось удаление или миграция данных.
