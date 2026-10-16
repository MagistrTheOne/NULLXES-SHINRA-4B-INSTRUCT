# SHINRA Tokenizer Audit — исправления

Дата: 2026-10-16. Ветка: `main`.
База: `30b9b887301bc506c7009df0ad502573b8d71c6e`.

## Статус

Изменён исходный код. Runtime-проверок нет. Реального tokenizer bundle нет;
production readiness не заявляется. Training, inference, модель, GPU/CUDA, Colab,
pytest и обучение токенизатора не запускались. Зависимости не обновлялись.

## P0

| Блокер | Исправление |
| --- | --- |
| `list + tuple` | `tokenizer/special_tokens.py`: обе группы — tuple, единый порядок по YAML ID. |
| Unicode `identity` | `tokenizer/contract.py`, `data/numeric_integrity.py`: Unicode NFC отдельно; `identity` передаётся только SentencePiece. |
| BOS loss | `data/pack.py`: обязательный `False`; BOS и PAD маскируются, EOD обучается. |
| SameFileError | `tokenizer/train_tokenizer.py`: сравнение resolved source/destination перед копированием model/vocab. |
| Потерянный EOD | EOD=4 входит в SP symbols, HF special registration, сериализацию, проверку ID и bundle. |
| Byte fallback | Unigram `byte_fallback=True`, проверка всех 256 byte pieces, decoder Replace → ByteFallback → Fuse. |
| Metaspace API | `prepend_scheme="never", split=False`; удалён `add_prefix_space`; SP dummy prefix отключён согласованно. |
| Двойной replace | В `data/pack.py` оставлена одна атомарная замена root manifest. |

## P1

1. **BOS:** document wrapper и loader проверяют единственную начальную границу;
   тело не может содержать controls/UNK. Chat helper проверяет один BOS и round-trip
   полного prompt. Inference не обрезает начало; превышение бюджета отклоняется.
2. **Injection:** буквальные управляющие строки отклоняются и helper, и Jinja template.
   Они не экранируются молча и не становятся управляющими ID из пользовательского текста.
3. **Остановка:** chat EOT=2, document continuation EOD=4. Остальные control IDs подавлены.
   Tool/reasoning/code roles не поддерживаются; ID остаются зарезервированными.
4. **Нормализация:** общий NFC; whitespace/CR/LF/tab не удаляются. Сравнение восстановления
   относится к NFC-тексту, исходный Unicode не обещается побайтно неизменным.
5. **JSONL:** tokenizer builder принимает только подготовленные split JSONL и manifest.
   Повреждённая строка/не-объект/отсутствующая provenance приводит к ошибке с местом записи.
6. **Held-out:** train/validation/blind проверяются до обучения: SHA256, normalized text,
   пересечения текстов и document/work identities. Обучение использует только train;
   метрики — всю validation, с обязательными EN и RU. Blind не используется для метрик.
7. **Bundle:** новый `tokenizer/bundle.py` хеширует model, vocab, HF JSON, config,
   special map, template и два отчёта. Манифест связывает их с конфигурационным контрактом.
8. **Связность:** production packed v3 manifest + каждая строка содержат `tokenizer_bundle_sha256`; source token offsets относятся к неперекрывающимся окнам одного документа.
   Checkpoint копирует весь bundle без пересериализации и хранит identity в training state.
   Loader/resume отказывают при несовпадении. YAML `artifact_sha256` теперь означает
   self-hash всего bundle manifest, не hash одного tokenizer.json.

## Дополнительные исправления

- Убран fallback `labels = input_ids`, который делал PAD целевыми токенами.
- Loader проверяет точные labels/attention mask/правое заполнение и ожидаемый split.
- Переполнение training document относительно SentencePiece byte limit отклоняется заранее.
- Кандидат tokenizer публикуется из временного каталога после проверок; существующий output
  не перезаписывается. Конкретные параметры corpus cap/sample/threads выбираются в YAML.
- Исправлена распаковка `shutil.disk_usage`: теперь используется `.free`.
- Held-out CE отделена от training z-loss; формулы архитектуры модели не менялись.
- Server сообщает `length` при достижении лимита, сохраняет output whitespace и возвращает
  HTTP 422 на ошибки контракта. Greedy generation больше не требует ручного `top_p=1`.
- `configs` включён в package discovery; wildcard упаковка старых tokenizer JSON удалена.

## Структура и удаление

Добавлены:

- `tokenizer/contract.py` — текст, control admission, encoding, BOS/overflow, stop policy;
- `tokenizer/bundle.py` — bundle manifest, verification, byte-identical copy;
- этот отчёт.

Переписаны tokenizer builder, special table/template, документация и устаревший README.
Удалён игнорировавшийся `tokenizer/tokenizer_friendship.json` с прежним словарём 131072;
он не был отслеживаемым Git-файлом, поэтому его удаление не входит в Git diff.
Удалены неиспользуемые экспортируемые группы special tokens и прежний permissive text reader.

Удалены `evaluation/harness.py` и зависимый `evaluation/code_eval.py`: прежний путь
обходил bundle/NFC/control contract и передавал `nullxes_shinra` в AutoModel без регистрации
при `trust_remote_code=False`. Рабочего SHINRA adapter в репозитории не было.
Интеграция lm-eval/HumanEval/MBPP явно недоступна до реализации и приёмки адаптера.
Остаются perplexity, document continuation и качественные chat probes.

## Выполнено только статически

- AST parse 47 Python-файлов; модули проекта не импортировались и не исполнялись.
- TOML parse `pyproject.toml`; safe YAML parse `configs/shinra.yaml`.
- Сравнение YAML с базовым Git-коммитом: все special IDs/strings и секция `model` неизменны.
- Проверены `loss_on_bos=False`, NFC/identity и выключенные training/Colab gates.
- По AST не найдены неразрешимые абсолютные импорты локальных модулей в production-каталогах.
- Просмотрены вызовы tokenizer в data/training/evaluation/inference и export/publish.
- Поиск прежних API/удалённых entrypoints и fake/mock/stub/TODO/FIXME в production-каталогах.
- `git diff --check`: ошибок whitespace нет.

Это не результаты тестов. `validate_config`, tokenizer APIs, SentencePiece/HF conversion,
packing и checkpoint/resume в runtime не выполнялись.

## Остаточные ограничения и следующий допуск

1. Нет утверждённого корпуса, обученного bundle, frozen SHA256 и измерений.
   `tokenizer.build.max_chars`, `input_sentence_size`, `num_threads` остаются null до аудита.
2. После отдельного разрешения нужны реальная сборка и измерения SP/HF parity,
   serialization/reload, EN/RU fertility, byte fallback, NFC/whitespace/numeric round-trip,
   BOS, literal-control rejection и границ контекста.
3. Нужны runtime-проверки tamper rejection для каждого bundle-файла, shard и checkpoint,
   loss masks, переходов между shards и восстановления полного training state.
4. Literal U+2581 (`▁`) зарезервирован Metaspace и отклоняется явно. Произвольный Unicode
   без исключений этой схемой не поддерживается. Управляющие строки в обычном тексте также
   отклоняются. Изменение этих политик требует нового контракта и пересборки bundle.
5. Exact/document identity disjointness не доказывает отсутствия семантических дубликатов,
   переводов и неверно размеченных произведений. Нужен upstream source/near-duplicate audit.
6. Старые packed v1/checkpoints несовместимы. Существующие тестовые fixtures packed v1
   также требуют переноса на новый контракт перед будущим запуском suite; здесь тесты,
   fixtures и mocks не запускались и не добавлялись.
7. Generic lm-eval adapter, tool calls and reasoning channels; V1 data packing now defines explicit independent, non-overlapping continuation windows
   длинных документов недоступны. Это явно обозначено, а не имитируется.

Никакого следующего запуска без разрешения Маги.
