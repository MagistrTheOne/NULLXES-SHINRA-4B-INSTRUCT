# Хранилища SHINRA

Три дома не смешиваются.

| Дом | Идентификатор | Роль |
| --- | --- | --- |
| Dataset Hub | `MagistrTheOne/NULLXES-SHINRA-S1-P0` | канонический корпус P0 |
| Storage Bucket | `hf://buckets/MagistrTheOne/nullxes-shinra-workspace` | изменяемое рабочее состояние |
| Model Hub | `MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT` | опубликованные веса и релизы |

`/tmp` — одноразовый scratch реплики. Локальный диск пользователя в конвейер не входит.

Будущий Job монтирует bucket так:

```text
hf://buckets/MagistrTheOne/nullxes-shinra-workspace:/workspace
```

Код читает `SHINRA_WORKSPACE`. Если переменная не задана и `/workspace` не смонтирован, путь `/tmp/shinra-workspace`, режим `EPHEMERAL`. `SHINRA_WORKSPACE_PERSISTENT=1` без записываемого `/workspace` завершается ошибкой и не подменяет checkpoint временным диском.

В bucket можно держать состояние сборки, компактные индексы дедупа, курсоры, манифесты и checkpoint обучения в `/workspace/training/p0/`. Туда не кладётся зеркало сырых датасетов и вторая копия корпуса. Корпус остаётся на Dataset Hub и читается по закреплённому commit.

`latest.json` пишется только после проверки checkpoint. Текущий Space инференса от `/workspace` не зависит.
