# V2 UPGRADE data spec — Stage D pilot EN/RU (подготовка, без запусков)

Старт: нулевые веса (random init). C489 — референс, не resume-база.
Пропорция по honest-токенам: EN 75% / RU 25%. Code/math — позже.

## 1. Карантин (не источники обучения, из git не удаляются)

Как train-вход запрещены до отдельного решения:

- `data/synth/` генераторы и их выходы v0 (`language_core`, `semantic_primitives`,
  `compositional`, `knowledge_shaped`, `structured`) — шаблонное распределение S0.
- `data/s1/pilot/`, `data/s1/pilot_v1/`, `data/s1/p0_dev/` — пилоты/фикстуры,
  включая `s1_curriculum_pilot_v1.jsonl`, `audit_sample.jsonl`, `quality_report.json`.
- `data/s1/source_inspection.json`, `source_registry.json`, `frozen_diagnostic_hashes.json` —
  диагностика, не корпус.
- Любые `*.bin / *.parquet` packed-артефакты прошлых стадий без `phase_c.pack.json`-уровня
  receipt — подлежат удалению из scratch при подготовке (не из git, они игнорируются).

Разрешённые train-источники Stage D: только bounded EN-хвост FineWeb-Edu после границы C
и bounded RU-срез FineWeb2 `rus_Cyrl` из раздела 2.

## 2. Закреплённые источники

| Часть | `source_id` | Upstream | Revision SHA | Статус |
|---|---|---|---|---|
| EN | `fineweb-edu-en` | `HuggingFaceFW/fineweb-edu`, файл `sample/10BT/013_00000.parquet` | upstream commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; raw SHA `sha256:b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de` | VERIFIED (frozen); receipt обязателен в пилоте |
| RU | `fineweb2-ru` | `HuggingFaceFW/fineweb-2`, subset `rus_Cyrl` | upstream revision `null` (не заморожена) | UNVERIFIED — до заморозки SHA и receipt источник не использовать для обучения |

Лимиты: per-source 4 GiB / 8M honest-токенов cap-уровня canary; лимит пилота 8M новых honest-токенов (legacy honest, не counter-v1).

## 3. Порядок подготовки

1. Группировка дублей (MinHash) до split.
2. Frozen held-out freeze до packing: всего 64–256 seq (~0.1–0.5M tok),
   из них EN 48–192 seq / RU 16–64 seq (пропорция пилота 75/25), EN+RU раздельно.
3. Исключить из held-out совпадения с уже потреблёнными C-документами (граница —
   `phase_c.pack.json` + ledger/cursor S0/C);
   если восстановление C-множества невозможно — явно зафиксировать ограничение
   независимости в receipt пилота.
4. Очистка/нормализация без chat-ролей; токенизация `add_special_tokens=False`.
5. Wrap `[BOS=1] + body + [END_OF_TEXT=18]`; запрет body `2,4–17` (`data/pack.py`,
   `data/data_v1/phase_c.py` поведение скипа сохранить + считать статистику потерь).
6. Pack `seq 2048`, packer `shinra-v2-pack.v1` без смены версии; счётчик целей —
   отдельно `counter-v1` (пост-shift).

## 4. UNK-проверка (по артефактам, без ретейна)

До packing: `unk_rate`, `bytes/token`, fertility на EN/RU сэмплах +
доля скипа по forbidden IDs. Инструменты: `tokenizer/analyze_tokenizer.py`,
`tokenizer/special_tokens.py`. Результат — отчёт, не новый vocab.

## 5. Receipt пилота

`input JSONL SHA` + `shard SHA` + `pack report` + `resolved stage config` +
`tokenizer dir SHA` + `code commit` — рядом с checkpoint (см. `docs/C489_LINK.md` формат).
