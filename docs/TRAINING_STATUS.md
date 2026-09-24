# NULLXES SHINRA-4B-INSTRUCT | TRAINING LEDGER

**Дата фиксации:** 23-09-2026  
**Режим записи:** локальный документ. Машину не запускать.  
**S0 в целом:** не закрыт. Слепой gate провален. Identity lock не ставится.

DEV-20 = 20/20 на уже виденном наборе. Это не генерализация. S0.5 это подтвердил.

## Stage ledger

| Stage | Задача | Статус | Результат |
| --- | --- | ---: | --- |
| BASE | Raw 3.926B model validation | ✅ | exact params, forward/generation PASS |
| OPT-0 | Full-weight backward | ✅ | 100% gradient coverage |
| OPT-1 | Native AdamW viability | ✅ | FP16, `eps=1e-6`, finite |
| OPT-2 | 20-step stability gate | ✅ | 20/20 finite, full-weight updates |
| S0.1 | Initial identity acquisition | ✅ | identity learned, routing collisions found |
| S0.2 | Canonical route lock | ✅ | 14/18, stable canonical targets |
| S0.3 | Contrast generalization | ✅ | best `15/20`, epoch 3 |
| S0.4 | Micro route repair | ✅ | best `19/20`, step 4 |
| S0.4.1 | EN language-route repair | ✅ | DEV **20/20**, one optimizer step |
| S0.5 | Blind Identity Gate 80 | ❌ | **59/80** |
| S0.5A | Route autopsy | ✅ | failures разобраны отдельно от теста |
| S0.6 | Dataset build | ✅ | 96 unique, overlap с S0.5 = 0 |
| **S0.6** | **Native route training** | 🟡 **NEXT** | full-weight, disjoint set |
| S0.6 | Internal DEV | ⬜ | после обучения, не вместо слепого gate |
| S0.5 | Frozen recheck | ⬜ | старый Blind-80 не доучивать |
| S0.7 | Fresh blind gate | ⬜ | новый набор, не S0.5 |
| S0.IDENTITY-LOCK | Identity lock | ⬜ | только после S0.7 |
| S1 | Language + Semantics | ⬜ | dataset/curriculum TBD |
| S2 | Mathematics | ⬜ | TBD |
| S3 | Code | ⬜ | TBD |
| S4 | General Reasoning | ⬜ | TBD |
| S5 | Multilingual | ⬜ | TBD |
| S6 | Tools / Structured Output | ⬜ | TBD |
| S7 | NULLXES Native Agent Behavior | ⬜ | TBD |

## S0.5 — факт прогона

```text
BLIND-80        59/80   ❌
EOT             80/80   ✅
EN              26/40   ❌
RU              33/40   ❌
```

Optimizer внутри теста не включался. Картина снята целиком, потом аутопсия.

## S0.6 dataset

```text
examples        96
unique          96
S0.5 overlap    0
SHA256          65b7807e...7fef
```

## Current canonical candidate

Кандидат до S0.6 training — чекпоинт, который провалил слепой gate:

```text
/workspace/shinra-stage0/checkpoints/s041-language-repair/stage041-dev20-perfect
```

## Hard facts, которые не отменены провалом S0.5

```text
PARAMETERS      3,926,076,416
TRAINING        FULL-WEIGHT NATIVE
ADAPTERS        NONE
DEV-20          20/20
DEV FAILURES    0
EOT             80/80
S0.4.1 STEPS    1
PEAK VRAM       30.773 GiB
```

## Развилка

```text
                 SHINRA RAW BASE
                       │
                       ▼
              FULL-WEIGHT VIABILITY ✅
                       │
                       ▼
                 IDENTITY ACQUIRE ✅
                       │
                       ▼
                  ROUTE LOCK ✅
                       │
                       ▼
               CONTRAST REPAIR ✅
                       │
                       ▼
                  DEV 20/20 ✅
                       │
                       ▼
             ┌─────────────────────┐
             │ S0.5 BLIND-80 ❌   │
             │ 59/80              │
             │ EN 26/40  RU 33/40 │
             │ EOT 80/80          │
             └─────────────────────┘
                       │
                       ▼
                 S0.5A AUTOPSY ✅
                       │
                       ▼
              S0.6 DATASET 96 ✅
              overlap with S0.5 = 0
                       │
                       ▼
             ┌─────────────────────┐
             │ S0.6 TRAIN 🟡 NEXT │
             └─────────────────────┘
                       │
                       ▼
               S0.6 INTERNAL DEV ⬜
                       │
                       ▼
             S0.5 FROZEN RECHECK ⬜
                       │
                       ▼
             S0.7 FRESH BLIND ⬜
                       │
                 PASS  │
                       ▼
            🔒 S0.IDENTITY-LOCK
                       │
                       ▼
               S1 SEMANTICS
```

## Правило lock

S0.IDENTITY-LOCK ставится только после свежего S0.7. Internal DEV и повтор старого Blind-80 lock не дают: первый набор уже использовался для repair, второй уже виден.

После lock Stage 0 не трогается как curriculum. Identity остаётся маленьким preservation/replay slice, чтобы S1/S2/S3 не стёрли маршрут.

## Следующий ход

**S0.6 Native Route Training** на 96 уникальных примерах без пересечения с Blind-80. Машину из этого документа не запускать.
