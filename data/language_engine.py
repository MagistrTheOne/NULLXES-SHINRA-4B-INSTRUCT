"""Deterministic, NULLXES-authored prose for tokenizer coverage only.

This corpus is deliberately separate from the verified task records consumed by
the model packer. The generator composes authored propositions into short
articles; it does not sample external text or generate vocabulary filler.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import multiprocessing
import re
import sqlite3
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Iterator

ENGINE_VERSION = "nullxes-native-language-engine.v1"
SCHEMA = "nullxes.shinra.tokenizer-language-corpus.v1"
LANGUAGES = ("en", "ru")

# Each entry is an authored, bilingual concept with a definition, a causal
# relation, a practical consequence, and a limitation. Keeping these clauses
# attached to one concept prevents arbitrary cross-domain sentence mixing.
CONCEPTS: tuple[tuple[str, ...], ...] = (
    ("feedback_control", "control", "A feedback controller compares a measured output with a target.", "The difference between them is an error signal.", "The controller changes its input to reduce that error.", "A delayed measurement can make a correct adjustment arrive too late.", "Регулятор с обратной связью сравнивает измеренный выход с заданным значением.", "Разность между ними образует сигнал ошибки.", "Регулятор меняет управляющее воздействие, чтобы уменьшить эту ошибку.", "Запаздывающий замер может привести к тому, что верная поправка поступит слишком поздно."),
    ("sensors", "physical_ai", "A sensor converts a physical quantity into a measurable signal.", "Calibration relates that signal to a stated unit and reference.", "A robot can use the calibrated value to estimate its surroundings.", "Noise and occlusion still limit what the sensor can establish.", "Датчик преобразует физическую величину в измеримый сигнал.", "Калибровка связывает сигнал с указанной единицей и эталоном.", "Робот использует откалиброванное значение для оценки окружения.", "Шум и перекрытие обзора всё равно ограничивают выводы по данным датчика."),
    ("state_estimation", "physical_ai", "A state estimate combines observations with a model of how a system changes.", "The estimate is updated when a new observation arrives.", "A controller can then choose an action using the current estimate.", "Uncertainty must remain explicit when observations are incomplete.", "Оценка состояния объединяет наблюдения с моделью изменения системы.", "При поступлении нового наблюдения оценка обновляется.", "Затем регулятор выбирает действие с учётом текущей оценки.", "Если наблюдения неполны, неопределённость следует сохранять явно."),
    ("collision_avoidance", "physical_ai", "Collision avoidance compares a planned path with observed obstacles.", "A safety margin accounts for sensing and braking uncertainty.", "The system slows down or stops when the remaining distance is insufficient.", "A clear sensor view is not proof that every part of the route is safe.", "Обход столкновений сопоставляет планируемый путь с обнаруженными препятствиями.", "Запас безопасности учитывает погрешность измерения и торможения.", "Система замедляется или останавливается, если расстояния недостаточно.", "Свободный обзор датчика не доказывает безопасность всего маршрута."),
    ("planning", "reasoning", "A plan is an ordered set of actions that is intended to reach a goal.", "Each action has preconditions and an expected effect.", "The planner checks whether an action is valid in the current state.", "A changed state can invalidate the remaining plan.", "План — это упорядоченный набор действий для достижения цели.", "У каждого действия есть предварительные условия и ожидаемый эффект.", "Планировщик проверяет допустимость действия в текущем состоянии.", "Изменившееся состояние может сделать оставшуюся часть плана недействительной."),
    ("cancellation", "instruction_following", "A cancellation request changes the status of a pending action.", "The system checks whether execution has already begun.", "If the action is still pending, it is removed from the queue.", "A completed action cannot be undone by changing its queue entry.", "Запрос отмены меняет статус ожидающего действия.", "Система проверяет, началось ли уже выполнение.", "Если действие ещё ожидает запуска, его удаляют из очереди.", "Завершённое действие нельзя отменить простым изменением записи в очереди."),
    ("algorithms", "computing", "An algorithm defines a finite sequence of steps for a specified problem.", "A precondition describes the inputs for which those steps are valid.", "An invariant helps verify that intermediate states preserve correctness.", "Termination and correctness are separate properties.", "Алгоритм задаёт конечную последовательность шагов для определённой задачи.", "Предусловие описывает входные данные, для которых шаги допустимы.", "Инвариант помогает проверить корректность промежуточных состояний.", "Завершение и правильность — разные свойства алгоритма."),
    ("complexity", "computing", "Complexity describes how resource use changes with input size.", "An asymptotic bound compares growth rates rather than one measured run.", "The bound helps identify workloads that may become expensive.", "It does not replace measurements on the target hardware.", "Сложность описывает, как расход ресурсов меняется с размером входа.", "Асимптотическая оценка сравнивает скорости роста, а не один замер.", "Она помогает выявить задачи, которые могут стать дорогими.", "Такая оценка не заменяет измерения на целевом оборудовании."),
    ("memory", "computing", "A program's memory use includes persistent state and temporary working buffers.", "Peak use can exceed the size of the final output.", "Chunking reduces some temporary buffers when computation is structured accordingly.", "The real peak still depends on autograd and kernel lifetimes.", "Расход памяти программы включает постоянное состояние и временные буферы.", "Пиковый расход может превышать размер итогового результата.", "Разбиение на блоки уменьшает часть временных буферов при подходящей организации вычислений.", "Фактический пик также зависит от autograd и времени жизни буферов ядра."),
    ("numerical_precision", "computing", "Finite-precision arithmetic rounds values to a representable set.", "Repeated operations can accumulate rounding error.", "A stable algorithm limits error growth for its intended range.", "Agreement on one input does not prove stability on every input.", "В арифметике конечной точности значения округляются до представимого набора.", "При повторных операциях ошибка округления может накапливаться.", "Устойчивый алгоритм ограничивает рост ошибки в заданном диапазоне.", "Совпадение на одном входе не доказывает устойчивость на всех входах."),
    ("causality", "reasoning", "A causal explanation identifies a change that contributes to an outcome.", "A correlation alone does not establish that relationship.", "An intervention can test whether changing one factor changes the result.", "The conclusion depends on which other conditions were held fixed.", "Причинное объяснение указывает изменение, которое влияет на результат.", "Одна корреляция сама по себе не устанавливает такую связь.", "Вмешательство проверяет, изменится ли результат при изменении одного фактора.", "Вывод зависит от того, какие остальные условия оставались постоянными."),
    ("state_machines", "computing", "A state machine represents a system with explicit states and transitions.", "A transition is allowed only when its condition is satisfied.", "Rejecting an invalid transition preserves the previous state.", "The transition table must cover every supported command.", "Конечный автомат представляет систему через явные состояния и переходы.", "Переход разрешён только при выполнении его условия.", "Отклонение недопустимого перехода сохраняет предыдущее состояние.", "Таблица переходов должна охватывать все поддерживаемые команды."),
    ("robot_motion", "physical_ai", "A motion command specifies a direction, speed, and reference frame.", "The reference frame determines how coordinates are interpreted.", "A controller translates the command into actuator signals.", "Actuator limits can prevent the requested motion.", "Команда движения задаёт направление, скорость и систему отсчёта.", "Система отсчёта определяет трактовку координат.", "Регулятор преобразует команду в сигналы исполнительным механизмам.", "Ограничения приводов могут не позволить выполнить запрошенное движение."),
    ("language_ambiguity", "language", "A sentence can admit more than one interpretation when a phrase has unclear scope.", "Context narrows the plausible readings but may not remove all ambiguity.", "A precise instruction states the object and the intended operation.", "When the missing detail changes the result, asking a question is safer than guessing.", "Предложение допускает несколько трактовок, если область действия фразы неясна.", "Контекст сужает круг вариантов, но не всегда устраняет неоднозначность.", "Точная инструкция называет объект и требуемое действие.", "Если недостающая деталь меняет результат, безопаснее уточнить её, чем угадывать."),
    ("dialogue_state", "language", "A dialogue state contains the relevant facts established in earlier turns.", "A new instruction can update or cancel part of that state.", "The response should use the latest compatible instruction.", "Old context should not override an explicit correction.", "Состояние диалога содержит важные факты из предыдущих реплик.", "Новая инструкция может обновить или отменить часть этого состояния.", "В ответе следует учитывать последнюю совместимую инструкцию.", "Старый контекст не должен отменять явное исправление."),
    ("materials", "engineering", "A material property describes how a substance responds under stated conditions.", "Temperature and loading can change the measured response.", "Engineers use those measurements to select a suitable component.", "A value without units or test conditions is incomplete evidence.", "Свойство материала описывает реакцию вещества при заданных условиях.", "Температура и нагрузка могут изменить измеренный отклик.", "Инженеры используют такие измерения при выборе компонента.", "Значение без единиц и условий испытания является неполным свидетельством."),
    ("energy", "physics", "Energy accounting tracks transfers between a system and its surroundings.", "A conversion changes the form of energy, not the conservation rule.", "Losses often appear as heat or other uncollected output.", "A practical balance must state its boundary and measurement interval.", "Энергетический баланс учитывает обмен системы с окружением.", "Преобразование меняет форму энергии, но не закон её сохранения.", "Потери часто проявляются как тепло или иной неучтённый выход.", "Практический баланс должен задавать границу системы и интервал измерения."),
    ("statistics", "reasoning", "A sample is a subset used to estimate properties of a larger population.", "The sampling method affects which observations are represented.", "An interval communicates uncertainty around an estimate.", "A narrow interval is not reliable when the sample is biased.", "Выборка — подмножество, по которому оценивают свойства всей совокупности.", "Метод отбора влияет на то, какие наблюдения представлены.", "Интервал показывает неопределённость оценки.", "Узкий интервал ненадёжен, если выборка смещена."),
    ("safety_interlock", "physical_ai", "A safety interlock blocks an operation while a required condition is false.", "The condition is checked before the operation is enabled.", "A failed check keeps the actuator in its safe state.", "The check itself must be monitored for faults.", "Блокировка безопасности запрещает операцию, пока обязательное условие ложно.", "Условие проверяется до разрешения операции.", "При неудачной проверке привод остаётся в безопасном состоянии.", "Саму проверку также необходимо контролировать на отказ."),
    ("version_control", "engineering", "A version identifier binds an artifact to a specific set of source bytes.", "A content hash detects changes to those bytes.", "Reproducibility requires recording inputs as well as the output hash.", "A matching name alone does not prove that two artifacts are identical.", "Идентификатор версии связывает артефакт с конкретными исходными байтами.", "Хэш содержимого обнаруживает изменения этих байтов.", "Для воспроизводимости фиксируют входы вместе с хэшем результата.", "Совпадение имени само по себе не доказывает идентичность артефактов."),
)

_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authored_text(concept: tuple[str, ...], language: str, ordinal: int) -> str:
    # Four discourse structures express the same authored, internally linked
    # propositions with distinct syntax rather than duplicating one prompt.
    en = concept[2:6]
    ru = concept[6:10]
    claims = en if language == "en" else ru
    if ordinal >= 96:
        raise ValueError("each authored concept supports at most 96 distinct discourse orderings")
    orders = tuple(itertools.permutations(range(4)))
    order = orders[ordinal % len(orders)]
    text = " ".join(claims[index] for index in order)
    if language == "en":
        lead = ("In {domain}, consider the following principle. ",
                "A useful engineering account begins with this distinction. ",
                "The following explanation connects a mechanism to its limits. ",
                "For a careful system design, keep these facts together. ")
    else:
        lead = ("Тема этого описания — «{domain}». Сначала зададим основное правило. ",
                "Рассмотрим «{domain}» через условие, следствие и ограничение. ",
                "Для темы «{domain}» важно связать механизм с его пределами. ",
                "В контексте «{domain}» полезно отделять наблюдение от вывода. ")
    domain = concept[1]
    if language == "ru":
        domain = {"control": "управление", "physical_ai": "физический ИИ", "reasoning": "рассуждение",
                  "instruction_following": "выполнение инструкций", "computing": "вычисления",
                  "language": "язык", "engineering": "инженерия", "physics": "физика"}[domain]
    return lead[ordinal // len(orders)].format(domain=domain) + text


def iter_records(root_seed: int, records_per_concept: int) -> Iterator[dict]:
    if type(root_seed) is not int or root_seed < 0 or type(records_per_concept) is not int or records_per_concept < 1:
        raise ValueError("root_seed and records_per_concept must be non-negative/positive integers")
    if records_per_concept > 96:
        raise ValueError("records_per_concept exceeds authored distinct discourse capacity (96)")
    for concept in CONCEPTS:
        family_id = f"language.{concept[0]}.v1"
        for language in LANGUAGES:
            for ordinal in range(records_per_concept):
                seed_payload = f"{ENGINE_VERSION}\0{root_seed}\0{family_id}\0{language}\0{ordinal}".encode()
                seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
                text = _authored_text(concept, language, ordinal)
                body = {"schema": "nullxes.shinra.tokenizer-language-record.v1", "record_id": "",
                        "text": text, "language": language, "domain": concept[1],
                        "family_id": family_id, "generator_version": ENGINE_VERSION, "seed": seed,
                        "ordinal": ordinal, "purpose": "tokenizer_training_only",
                        "provenance": "NULLXES authored bilingual concept propositions"}
                body["record_id"] = _sha_bytes(json.dumps(
                    [ENGINE_VERSION, root_seed, family_id, language, ordinal], separators=(",", ":")
                ).encode())
                body["text_sha256"] = _sha_bytes(text.encode("utf-8"))
                yield body


def _concept_records(args: tuple[tuple[str, ...], int, int]) -> list[dict]:
    concept, root_seed, records_per_concept = args
    return list(iter_records_for_concept(concept, root_seed, records_per_concept))


def iter_records_for_concept(concept: tuple[str, ...], root_seed: int,
                             records_per_concept: int) -> Iterator[dict]:
    family_id = f"language.{concept[0]}.v1"
    for language in LANGUAGES:
        for ordinal in range(records_per_concept):
            seed_payload = f"{ENGINE_VERSION}\0{root_seed}\0{family_id}\0{language}\0{ordinal}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
            text = _authored_text(concept, language, ordinal)
            body = {"schema": "nullxes.shinra.tokenizer-language-record.v1", "record_id": "",
                    "text": text, "language": language, "domain": concept[1],
                    "family_id": family_id, "generator_version": ENGINE_VERSION, "seed": seed,
                    "ordinal": ordinal, "purpose": "tokenizer_training_only",
                    "provenance": "NULLXES authored bilingual concept propositions"}
            body["record_id"] = _sha_bytes(json.dumps(
                [ENGINE_VERSION, root_seed, family_id, language, ordinal], separators=(",", ":")
            ).encode())
            body["text_sha256"] = _sha_bytes(text.encode("utf-8"))
            yield body


def iter_records_parallel(root_seed: int, records_per_concept: int,
                          workers: int) -> Iterator[dict]:
    if type(root_seed) is not int or root_seed < 0 \
            or type(records_per_concept) is not int or not 1 <= records_per_concept <= 96:
        raise ValueError("root_seed must be non-negative and records_per_concept must be in [1, 96]")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        for concept in CONCEPTS:
            yield from iter_records_for_concept(concept, root_seed, records_per_concept)
        return
    jobs = ((concept, root_seed, records_per_concept) for concept in CONCEPTS)
    with ProcessPoolExecutor(max_workers=min(workers, len(CONCEPTS)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for rows in pool.map(_concept_records, jobs, chunksize=1):
            yield from rows


def write_corpus(path: Path, root_seed: int, records_per_concept: int,
                 max_sentence_bytes: int, workers: int = 1) -> dict:
    audit_path = path.with_suffix(".audit.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    domains = {language: Counter() for language in LANGUAGES}
    words = {language: set() for language in LANGUAGES}
    chars = {language: 0 for language in LANGUAGES}
    byte_lengths = {language: 0 for language in LANGUAGES}
    if path.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite tokenizer-only corpus artifact: {path}")
    created_artifact = False
    try:
        with closing(sqlite3.connect(audit_path)) as db:
            db.execute("CREATE TABLE texts (sha TEXT PRIMARY KEY)")
            output_stream = path.open("xb")
            created_artifact = True
            with output_stream as out:
                for row in iter_records_parallel(root_seed, records_per_concept, workers):
                    encoded_text = row["text"].encode("utf-8")
                    if len(encoded_text) > max_sentence_bytes:
                        raise ValueError(f"{row['record_id']}: text exceeds tokenizer max_sentence_bytes")
                    try:
                        db.execute("INSERT INTO texts VALUES (?)", (row["text_sha256"],))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"duplicate tokenizer-only text: {row['record_id']}") from exc
                    out.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
                    language = row["language"]
                    counts[language] += 1
                    domains[language][row["domain"]] += 1
                    words[language].update(token.casefold() for token in _WORD.findall(row["text"]))
                    chars[language] += len(row["text"])
                    byte_lengths[language] += len(encoded_text)
                out.flush()
                os.fsync(out.fileno())
            db.commit()
    except Exception:
        if created_artifact:
            path.unlink(missing_ok=True)
        raise
    finally:
        audit_path.unlink(missing_ok=True)
    stats = {"records_by_language": dict(sorted(counts.items())),
             "records_by_domain": {key: dict(sorted(value.items())) for key, value in domains.items()},
             "unique_casefolded_word_forms_by_language": {key: len(value) for key, value in words.items()},
             "characters_by_language": chars, "utf8_bytes_by_language": byte_lengths,
             "exact_duplicate_text_records": 0, "record_count": sum(counts.values()),
             "records_per_concept": records_per_concept}
    return stats
