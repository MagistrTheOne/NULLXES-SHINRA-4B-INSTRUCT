# NULLXES SHINRA-4B-INSTRUCT | TRAINING LEDGER

**Дата фиксации:** 23-09-2026  
**Режим записи:** локальный документ. Машину не запускать.  
**S0 в целом:** не закрыт. DEV-20 не является финальным доказательством generalization.

DEV-20 видели с ошибками, и repair строился вокруг этих ошибок. Поэтому Stage 0 **не получает общую зелёную галку**, пока не пройден слепой gate.

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
| S0.4.1 | EN language-route repair | ✅ | **20/20**, one optimizer step |
| **S0.5** | **Blind Identity Gate 80** | 🟡 **NEXT** | inference only |
| S0.LOCK | Identity Lock | ⬜ | requires blind gate |
| S1 | Language + Semantics | ⬜ | dataset/curriculum TBD |
| S2 | Mathematics | ⬜ | TBD |
| S3 | Code | ⬜ | TBD |
| S4 | General Reasoning | ⬜ | TBD |
| S5 | Multilingual | ⬜ | TBD |
| S6 | Tools + Structured Output | ⬜ | TBD |
| S7 | NULLXES Native Agent Behavior | ⬜ | TBD |

## Current canonical candidate

```text
/workspace/shinra-stage0/checkpoints/s041-language-repair/stage041-dev20-perfect
```

## Current hard facts

```text
PARAMETERS      3,926,076,416
TRAINING        FULL-WEIGHT NATIVE
ADAPTERS        NONE
DEV-20          20/20
DEV FAILURES    0
EOT             PASS
S0.4.1 STEPS    1
PEAK VRAM       30.773 GiB
```

DEV-20 = 20/20 на уже виденном наборе. Это не слепая генерализация.

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
             │ S0.5 BLIND-80 🟡   │
             │ TRAINING = FORBIDDEN│
             └─────────────────────┘
                       │
                 PASS  │
                       ▼
               🔒 IDENTITY LOCK
                       │
                       ▼
               S1 SEMANTICS
```

Следующая развилка отсутствует: ход один — **S0.5 BLIND-80**.

## Критерий S0.5

Blind-80 строится из **80 совершенно новых prompts**: 40 EN + 40 RU.

Проверяется:

- identity
- creator
- false identity
- false creator
- SHINRA ≠ NULLXES
- category boundary
- paraphrase generalization
- language isolation

Правила прогона:

- никакого optimizer
- никакого repair внутри теста
- сначала полная картина: 80 probes, failures, распределение по категориям

Если что-то падает, S0.5 остаётся красным. Route margins диагностируются отдельно, не внутри того же прогона.

Если проходит, результаты сохраняются и ставится:

> **S0 IDENTITY LOCK: ✅ CLOSED**

После этого Stage 0 больше не трогается как curriculum. Identity examples остаются маленьким preservation/replay slice в последующих стадиях, чтобы S1/S2/S3 не стёрли SHINRA.

## Следующий ход

**S0.5 BLIND-80.** Inference only. Training forbidden.
