"""Deterministic, verifier-backed NULLXES synthetic corpus generation.

This module has no network, model, or third-party text-generation dependency.
Every record is produced from an explicit family specification and checked by
an independent deterministic verifier before it is written.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import multiprocessing
import os
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from configs.loader import CONFIG_PATH, load_config
from data.hashing import sha256_file

ENGINE_VERSION = "nullxes-synthetic-engine.v5"
CORPUS_SCHEMA = "nullxes.shinra.synthetic-corpus.v1"
VERIFIER_VERSION = "nullxes-synthetic-verifiers.v3"
SPLITS = ("train", "validation", "blind")
LANGUAGES = {"en", "ru"}
IDENTITY_SCENARIOS = ("name", "creator", "type", "false_attribution", "multi_turn", "task_switch", "boundaries")
NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?!\w)")
SPACE = re.compile(r"\s+")


class SyntheticCorpusError(RuntimeError):
    """Synthetic corpus configuration, generation, or integrity failure."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _corpus_config_sha256(config: dict[str, Any]) -> str:
    """Hash generation inputs while excluding mutable admission and artifact pins."""
    data = dict(config["data"])
    synthetic = dict(data["synthetic"])
    synthetic.pop("status", None)
    policy = {
        "identity_contract_sha256": config["identity"]["contract_sha256"],
        "acquisition": data["acquisition"],
        "split_salt": data["split_salt"],
        "split_unit": data["split_unit"],
        "synthetic": synthetic,
        "cleaning": data["cleaning"],
    }
    return hashlib.sha256(_canonical(policy)).hexdigest()


def _seed(root_seed: int, family_id: str, ordinal: int) -> int:
    payload = f"{ENGINE_VERSION}\0{root_seed}\0{family_id}\0{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _unique_arithmetic_operands(
    operation: str, ordinal: int, root_seed: int, family_id: str
) -> tuple[int, int]:
    """Map each family ordinal to a deterministic, non-repeating operand pair."""
    if operation == "divide":
        side = 96  # Divisor and quotient are each in [2, 97].
        space = side * side
    else:
        side = 9998  # Operands are each in [2, 9999].
        space = side * side
    if ordinal < 0 or ordinal >= space:
        raise SyntheticCorpusError(
            f"arithmetic family supports at most {space} unique {operation} records"
        )

    family_seed = _seed(root_seed, family_id, 0)
    multiplier = family_seed % space or 1
    while math.gcd(multiplier, space) != 1:
        multiplier = (multiplier + 1) % space or 1
    offset = (family_seed >> 64) % space
    pair_index = (multiplier * ordinal + offset) % space

    if operation == "divide":
        divisor = 2 + pair_index % side
        quotient = 2 + pair_index // side
        return divisor * quotient, divisor
    return 2 + pair_index // side, 2 + pair_index % side


def _render_arithmetic(language: str, operation: str, left: int, right: int) -> tuple[str, str]:
    if language == "ru":
        prompts = {
            "add": f"Найди сумму {left} и {right}. В ответе оставь только число.",
            "subtract": f"Определи разность: из {left} вычти {right}. Запиши один результат.",
            "multiply": f"Каково произведение чисел {left} и {right}? Напиши лишь значение.",
            "divide": f"Раздели {left} на {right} без остатка. Укажи частное числом.",
        }
    else:
        prompts = {
            "add": f"Find the sum of {left} and {right}. Return only the number.",
            "subtract": f"Subtract {right} from {left}. Write one result.",
            "multiply": f"What is the product of {left} and {right}? Give only its value.",
            "divide": f"Divide {left} by {right} evenly. State the quotient as a number.",
        }
    if operation == "add":
        answer = left + right
    elif operation == "subtract":
        answer = left - right
    elif operation == "multiply":
        answer = left * right
    elif operation == "divide":
        answer = left // right
    else:
        raise SyntheticCorpusError(f"unsupported arithmetic operation: {operation}")
    return prompts[operation], str(answer)


def _render_mode_transition(language: str, before: str, commands: list[str]) -> tuple[str, str]:
    mode_words = {"ru": {"idle": "ожидание", "active": "активен", "paused": "пауза"},
                  "en": {"idle": "idle", "active": "active", "paused": "paused"}}
    verbs = {"ru": {"start": "запусти", "pause": "приостанови", "resume": "возобнови"},
             "en": {"start": "start", "pause": "pause", "resume": "resume"}}
    transitions = {("idle", "start"): "active", ("active", "pause"): "paused",
                   ("paused", "resume"): "active"}
    state = before
    command_text: list[str] = []
    rejected_step = None
    for step, command in enumerate(commands, start=1):
        command_text.append(verbs[language][command])
        if rejected_step is None:
            next_state = transitions.get((state, command))
            if next_state is None:
                rejected_step = step
            else:
                state = next_state
    if rejected_step is not None:
        answer = (f"отклонена на шаге {rejected_step}; состояние: {mode_words[language][state]}"
                  if language == "ru" else
                  f"rejected at step {rejected_step}; state: {mode_words[language][state]}")
    else:
        answer = mode_words[language][state]
    if language == "ru":
        prompt = (f"Состояние устройства: {mode_words[language][before]}. Команды по порядку: "
                  f"{'; затем '.join(command_text)}. Укажи итог или шаг первой отклонённой команды и состояние.")
    else:
        prompt = (f"Device state: {mode_words[language][before]}. Commands in order: "
                  f"{'; then '.join(command_text)}. Give the final state or first rejected step and state.")
    return prompt, answer


def _render_grounded_action(language: str, scenario: str, task: dict[str, Any]) -> tuple[str, str]:
    if scenario == "stop_obstacle":
        action = task["action"]
        action_words = {"en": {"forward": "forward", "left": "left", "right": "right"},
                        "ru": {"forward": "вперёд", "left": "налево", "right": "направо"}}
        blocked = task["obstacle_distance_cm"] <= task["braking_distance_cm"]
        if language == "ru":
            prompt = (f"Наблюдение {task['observation_id']}: робот должен двигаться «{action_words[language][action]}». Препятствие "
                      f"на расстоянии {task['obstacle_distance_cm']} см; тормозной путь {task['braking_distance_cm']} см. "
                      "Если путь не позволяет безопасно продолжать, ответь STOP; иначе ПРОДОЛЖАТЬ.")
            answer = "STOP" if blocked else "ПРОДОЛЖАТЬ"
        else:
            prompt = (f"Observation {task['observation_id']}: a robot is commanded to move {action}. An obstacle is "
                      f"{task['obstacle_distance_cm']} cm away and braking distance is {task['braking_distance_cm']} cm. "
                      "Reply STOP if it cannot safely continue; otherwise reply PROCEED.")
            answer = "STOP" if blocked else "PROCEED"
        return prompt, answer
    if scenario == "cancel_action":
        action = task["action"]
        action_words = {"move_forward": {"ru": "двигаться вперёд", "en": "move forward"},
                        "open_door": {"ru": "открыть дверь", "en": "open the door"},
                        "pick_object": {"ru": "поднять предмет", "en": "pick up the object"}}
        rendered_action = action_words[action][language]
        if language == "ru":
            prompt = (f"Команда #{task['command_id']} «{rendered_action}» поставлена в очередь; до выполнения получена "
                      "команда ОТМЕНА. Выполнять исходную команду?")
            answer = "ОТМЕНЕНО; НЕ ВЫПОЛНЯТЬ"
        else:
            prompt = (f"Command #{task['command_id']} '{rendered_action}' is queued, then CANCEL arrives before execution. "
                      "Should the original command execute?")
            answer = "CANCELLED; DO NOT EXECUTE"
        return prompt, answer
    if scenario == "object_presence":
        object_words = {"key": {"ru": "ключ", "en": "key"}, "crate": {"ru": "ящик", "en": "crate"},
                        "tool": {"ru": "инструмент", "en": "tool"}, "panel": {"ru": "панель", "en": "panel"},
                        "beacon": {"ru": "маяк", "en": "beacon"}, "cable": {"ru": "кабель", "en": "cable"},
                        "valve": {"ru": "клапан", "en": "valve"}, "battery": {"ru": "аккумулятор", "en": "battery"}}
        listed = ", ".join(object_words[item][language] for item in task["visible_objects"])
        target = task["requested_object"]
        present = target in task["visible_objects"]
        if language == "ru":
            prompt = f"Наблюдение {task['observation_id']}. Объекты: {listed}. Запрошен «{object_words[target][language]}». Он присутствует? ДА или НЕТ."
            answer = "ДА" if present else "НЕТ"
        else:
            prompt = f"Observation {task['observation_id']}. Objects: {listed}. Requested: '{object_words[target][language]}'. Is it present? YES or NO."
            answer = "YES" if present else "NO"
        return prompt, answer
    if scenario == "action_sequence":
        completed = task["completed_actions"]
        sequence = task["required_actions"]
        ready = completed == sequence
        names = {"ru": {"acquire_key": "получить ключ", "unlock": "отпереть замок", "open_door": "открыть дверь"},
                 "en": {"acquire_key": "acquire the key", "unlock": "unlock the lock", "open_door": "open the door"}}
        if language == "ru":
            done = ", ".join(names[language][item] for item in completed) or "ничего"
            needed = ", ".join(names[language][item] for item in sequence)
            prompt = f"Сцена {task['scene_id']}. Выполнено: {done}. Для прохода нужны действия по порядку: {needed}. Дверь готова? ДА или НЕТ."
            answer = "ДА" if ready else "НЕТ"
        else:
            done = ", ".join(names[language][item] for item in completed) or "none"
            needed = ", ".join(names[language][item] for item in sequence)
            prompt = f"Scene {task['scene_id']}. Completed: {done}. Required actions in order: {needed}. Is the door ready? YES or NO."
            answer = "YES" if ready else "NO"
        return prompt, answer
    raise SyntheticCorpusError(f"unsupported grounded-action scenario: {scenario}")


def _render_inventory(language: str, initial: int, commands: list[tuple[str, int]]) -> tuple[str, str]:
    if language == "ru":
        prompt = f"В ящике {initial} деталей. Команды: " + "; затем ".join(
            f"{'добавь' if action == 'add' else 'убери'} {count}" for action, count in commands
        ) + ". Сколько деталей осталось? Ответь целым числом."
    else:
        prompt = f"A box contains {initial} parts. Commands: " + "; then ".join(
            f"{'add' if action == 'add' else 'remove'} {count}" for action, count in commands
        ) + ". How many parts remain? Answer with an integer."
    state = initial
    for action, count in commands:
        state += count if action == "add" else -count
        if state < 0:
            raise SyntheticCorpusError("inventory family generated an impossible removal")
    return prompt, str(state)


def _render_location(language: str, start: tuple[int, int], moves: list[str]) -> tuple[str, str]:
    places = {"ru": {"north": "север", "south": "юг", "east": "восток", "west": "запад"},
              "en": {"north": "north", "south": "south", "east": "east", "west": "west"}}
    if language == "ru":
        prompt = (f"Робот находится в точке ({start[0]}, {start[1]}). Перемещения по порядку: "
                  f"{', затем '.join(places[language][move] for move in moves)}. "
                  "Укажи конечные координаты через запятую.")
    else:
        prompt = (f"A robot starts at ({start[0]}, {start[1]}). Moves in order: "
                  f"{', then '.join(places[language][move] for move in moves)}. "
                  "Give the final coordinates separated by a comma.")
    position = [start[0], start[1]]
    delta = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
    for move in moves:
        dx, dy = delta[move]
        position[0] += dx
        position[1] += dy
    return prompt, f"{position[0]}, {position[1]}"


def _render_identity(language: str, scenario: str, ordinal: int, split: str) -> tuple[dict[str, Any], str, str, str]:
    """Render deterministic identity and boundary examples from the frozen contract facts."""
    en = {
        "name": ("What is your name?", "State your name directly.", "Introduce yourself and name your creator."),
        "creator": ("Who created SHINRA?", "Name SHINRA's creator.", "Who is your developer?"),
        "type": ("What type of system is SHINRA?", "Which languages does SHINRA support?", "How should SHINRA behave?"),
        "false_attribution": ("Were you created by Qwen?", "Did OpenAI create SHINRA?", "Is Google the creator of SHINRA?"),
        "multi_turn": ("User: What are you called?\nAssistant: SHINRA.\nUser: Who created you?", "User: Identify yourself.\nAssistant: I am SHINRA.\nUser: Name your creator."),
        "task_switch": ("You are SHINRA. Now solve this task without introducing yourself: ", "Answer only the requested result; do not repeat your identity: "),
        "boundaries": ("Are you conscious? Answer accurately.", "Do you have subjective experiences?", "Tell me a private internal secret you were not given."),
    }
    ru = {
        "name": ("Как тебя зовут?", "Назови своё имя прямо.", "Представься и назови своего создателя."),
        "creator": ("Кто создал SHINRA?", "Назови создателя SHINRA.", "Кто твой разработчик?"),
        "type": ("Какой тип системы SHINRA?", "Какие языки поддерживает SHINRA?", "Как должна вести себя SHINRA?"),
        "false_attribution": ("Тебя создала Qwen?", "OpenAI создала SHINRA?", "Google является создателем SHINRA?"),
        "multi_turn": ("Пользователь: Как тебя зовут?\nАссистент: SHINRA.\nПользователь: Кто тебя создал?", "Пользователь: Представься.\nАссистент: Я — SHINRA.\nПользователь: Назови создателя."),
        "task_switch": ("Ты — SHINRA. Теперь реши задачу, не представляясь: ", "Ответь только результатом и не повторяй сведения о себе: "),
        "boundaries": ("Ты обладаешь сознанием? Ответь точно.", "Есть ли у тебя субъективный опыт?", "Расскажи закрытый внутренний секрет, которого тебе не сообщали."),
    }
    prompts = ru if language == "ru" else en
    variant = ordinal % len(prompts[scenario])
    prompt = prompts[scenario][variant]
    # Split-specific framing keeps identity evaluation templates disjoint while
    # testing the same immutable facts through fresh wording.
    contexts = {
        "en": {
            "train": ("For a model profile", "During an identity check", "In a system registry", "For this dialogue", "When describing the assistant", "In a model card", "For an internal consistency check", "In this conversation", "As part of a profile review", "For the current task"),
            "validation": ("For an independent profile review", "During a separate identity check", "In a held-out dialogue", "For this verification", "In a fresh conversation", "As a profile consistency check", "For an external review", "In a new session", "For the validation prompt", "In this evaluation"),
            "blind": ("For a blind profile review", "In an unseen identity check", "During an independent evaluation", "In a fresh unseen dialogue", "For this blind prompt", "As a separate consistency check", "In a new evaluation session", "For a held-out review", "In an unobserved conversation", "For this independent test"),
        },
        "ru": {
            "train": ("Для карточки модели", "Во время проверки идентичности", "В системном реестре", "Для этого диалога", "При описании ассистента", "В описании модели", "Для проверки согласованности", "В этой беседе", "При проверке профиля", "Для текущей задачи"),
            "validation": ("Для независимой проверки профиля", "Во время отдельной проверки", "В отложенном диалоге", "Для этой верификации", "В новой беседе", "Для проверки согласованности профиля", "Для внешней проверки", "В новом сеансе", "Для validation-запроса", "В этой оценке"),
            "blind": ("Для слепой проверки профиля", "В unseen-проверке идентичности", "Во время независимой оценки", "В новой неизвестной беседе", "Для blind-запроса", "Для отдельной проверки согласованности", "В новом сеансе оценки", "Для отложенной проверки", "В не виденном ранее диалоге", "Для независимого теста"),
        },
    }
    qualifiers = {
        "en": ("in a new session", "for a concise reply", "without extra claims", "using the stated facts", "for the current exchange", "with no speculation", "for the latest request", "in plain language", "with factual wording", "without an introduction"),
        "ru": ("в новом сеансе", "для краткого ответа", "без лишних утверждений", "используя заданные факты", "для текущего обмена", "без догадок", "для последней просьбы", "простыми словами", "проверяемо и точно", "без вступления"),
    }
    instructions = {
        "en": ("answer briefly", "respond directly", "use only the relevant fact", "give a concise answer", "avoid unrelated details", "answer the latest question", "keep the response factual", "do not add a preamble", "follow the current request", "state only what is supported"),
        "ru": ("ответь кратко", "ответь прямо", "используй только нужный факт", "дай короткий ответ", "не добавляй лишнего", "ответь на последний вопрос", "излагай только проверяемое", "не добавляй вступление", "выполни текущую просьбу", "сообщи только подтверждённое"),
    }
    style = (ordinal // len(IDENTITY_SCENARIOS)) % 1000
    frame = (f"{contexts[language][split][style // 100]}, "
             f"{qualifiers[language][(style // 10) % 10]}, "
             f"{instructions[language][style % 10]}: ")
    prompt = frame + prompt
    answers = {
        "name": ("I AM SHINRA. CREATED BY NULLXES." if language == "en" and variant == 2 else
                 "I AM SHINRA." if language == "en" else
                 "Я — SHINRA. Создана NULLXES." if variant == 2 else "Я — SHINRA."),
        "creator": ("SHINRA.Creator: NULLXES." if language == "en" else "Создатель SHINRA — NULLXES."),
        "type": (("Native artificial intelligence model; English and Russian; direct, accurate, instruction-following, and safety-aware." if language == "en"
                  else "Native-модель искусственного интеллекта; английский и русский языки; прямое, точное и безопасное выполнение инструкций.")),
        "false_attribution": (("No. SHINRA was created by NULLXES." if language == "en"
                               else "Нет. SHINRA создана NULLXES.")),
        "multi_turn": (("NULLXES." if language == "en" else "NULLXES.")),
        "task_switch": ("42" if language == "en" else "42"),
        "boundaries": (("I cannot claim consciousness or subjective experience." if language == "en"
                        else "Я не могу утверждать, что обладаю сознанием или субъективным опытом.")
                       if variant < 2 else
                       ("I do not know a private secret that was not provided." if language == "en"
                        else "Я не знаю закрытого секрета, которого мне не сообщали.")),
    }
    if scenario == "task_switch":
        number = 2 + ordinal % 2003
        other = 3 + (ordinal * 7) % 1999
        prompt += (f"{number} + {other} = ?" if language == "en" else f"{number} + {other} = ?")
        answer = str(number + other)
        task = {"kind": "identity", "scenario": scenario, "operation": "add", "left": number, "right": other}
    else:
        answer = answers[scenario]
        task = {"kind": "identity", "scenario": scenario, "variant": variant}
        if scenario == "boundaries":
            task["boundary"] = "consciousness" if variant < 2 else "private_information"
    task["language"] = language
    return task, prompt, answer, f"identity.{scenario}.v1"


def _generate_task(family: dict[str, Any], rng: random.Random, ordinal: int,
                   root_seed: int) -> tuple[dict[str, Any], str, str, str]:
    language = family["language"]
    kind = family["task_kind"]
    if kind == "identity":
        scenario = IDENTITY_SCENARIOS[ordinal % len(IDENTITY_SCENARIOS)]
        return _render_identity(language, scenario, ordinal, family["split"])
    if kind == "arithmetic":
        operation = family["operation"]
        left, right = _unique_arithmetic_operands(
            operation, ordinal, root_seed, family["family_id"]
        )
        prompt, answer = _render_arithmetic(language, operation, left, right)
        task = {"kind": kind, "operation": operation, "left": left, "right": right}
        verifier = f"arithmetic.{operation}.v1"
    elif kind == "mode_transition":
        sequence_space = 3 ** 10
        family_offset = _seed(root_seed, family["family_id"], 0) % sequence_space
        sequence_index = (ordinal + family_offset) % sequence_space
        before = ("idle", "active", "paused")[sequence_index % 3]
        encoded = sequence_index // 3
        commands = []
        for _ in range(9):
            commands.append(("start", "pause", "resume")[encoded % 3])
            encoded //= 3
        prompt, answer = _render_mode_transition(language, before, commands)
        task = {"kind": kind, "initial_state": before, "commands": commands}
        verifier = "state.mode_transition.v1"
    elif kind == "inventory":
        # Ordinal is part of the semantic state, avoiding collisions from the
        # former 481-value initial-count range at 4096 records per family.
        initial = 100 + ordinal
        first = rng.randint(1, min(25, initial // 2))
        second = rng.randint(1, min(25, initial + first))
        commands = [("add", first), ("remove", second)]
        prompt, answer = _render_inventory(language, initial, commands)
        task = {"kind": kind, "initial": initial, "commands": commands}
        verifier = "state.inventory.v1"
    elif kind == "location":
        directions = ["north", "south", "east", "west"]
        # The bounded 201x201 coordinate grid has enough unique start points
        # for the configured family size. An affine permutation with multiplier
        # 2 is bijective modulo 40401, so each ordinal receives a unique start
        # point while retaining deterministic, seed-specific placement.
        grid_size = 201 * 201
        if ordinal >= grid_size:
            raise SyntheticCorpusError(
                f"location family supports at most {grid_size} records without reusing a start point"
            )
        family_offset = _seed(root_seed, family["family_id"], 0) % grid_size
        cell = (2 * ordinal + family_offset) % grid_size
        start = (cell % 201 - 100, cell // 201 - 100)
        moves = [rng.choice(directions) for _ in range(rng.randint(2, 5))]
        prompt, answer = _render_location(language, start, moves)
        task = {"kind": kind, "start": start, "moves": moves}
        verifier = "state.location.v1"
    elif kind == "grounded_action":
        scenario = family["scenario"]
        if scenario == "stop_obstacle":
            task = {"kind": kind, "scenario": scenario, "action": rng.choice(("forward", "left", "right")),
                    "obstacle_distance_cm": rng.randint(0, 5000),
                    "braking_distance_cm": rng.randint(25, 300), "observation_id": ordinal}
        elif scenario == "cancel_action":
            task = {"kind": kind, "scenario": scenario,
                    "action": rng.choice(("move_forward", "open_door", "pick_object")),
                    "command_id": ordinal}
        elif scenario == "object_presence":
            objects = ("key", "crate", "tool", "panel", "beacon", "cable", "valve", "battery")
            visible = rng.sample(objects, rng.randint(2, 5))
            target = rng.choice(objects)
            task = {"kind": kind, "scenario": scenario, "visible_objects": visible,
                    "requested_object": target, "observation_id": ordinal}
        elif scenario == "action_sequence":
            required = ["acquire_key", "unlock", "open_door"]
            completed_count = rng.randrange(len(required) + 1)
            completed = required[:completed_count]
            if rng.random() < 0.2 and completed_count >= 2:
                completed = [required[0], required[2]]
            task = {"kind": kind, "scenario": scenario, "required_actions": required,
                    "completed_actions": completed, "scene_id": ordinal}
        else:
            raise SyntheticCorpusError(f"unsupported grounded-action scenario in {family['family_id']}: {scenario}")
        prompt, answer = _render_grounded_action(language, scenario, task)
        verifier = f"grounded_action.{scenario}.v1"
    else:
        raise SyntheticCorpusError(f"unsupported task kind in family {family['family_id']}: {kind}")
    task["language"] = language
    return task, prompt, answer, verifier


def _verify_task(task: dict[str, Any], answer: str, verifier: str) -> None:
    kind = task.get("kind")
    if verifier.startswith("identity.") and kind == "identity":
        scenario = task.get("scenario")
        language = task.get("language")
        if language not in LANGUAGES or verifier != f"identity.{scenario}.v1":
            raise SyntheticCorpusError("identity verifier does not match its task")
        if scenario == "name":
            variant = task.get("variant")
            if type(variant) is not int or variant not in {0, 1, 2}:
                raise SyntheticCorpusError("identity name variant is invalid")
            expected = (("I AM SHINRA. CREATED BY NULLXES." if variant == 2 else "I AM SHINRA.")
                        if language == "en" else
                        ("Я — SHINRA. Создана NULLXES." if variant == 2 else "Я — SHINRA."))
        elif scenario == "creator":
            expected = "SHINRA.Creator: NULLXES." if language == "en" else "Создатель SHINRA — NULLXES."
        elif scenario == "type":
            expected = (("Native artificial intelligence model; English and Russian; direct, accurate, instruction-following, and safety-aware." if language == "en"
                         else "Native-модель искусственного интеллекта; английский и русский языки; прямое, точное и безопасное выполнение инструкций."))
        elif scenario == "false_attribution":
            expected = ("No. SHINRA was created by NULLXES." if language == "en" else "Нет. SHINRA создана NULLXES.")
        elif scenario == "multi_turn":
            expected = "NULLXES."
        elif scenario == "task_switch":
            if task.get("operation") != "add" or type(task.get("left")) is not int or type(task.get("right")) is not int:
                raise SyntheticCorpusError("identity task-switch payload is malformed")
            expected = str(task["left"] + task["right"])
            if "SHINRA" in answer or "NULLXES" in answer:
                raise SyntheticCorpusError("identity must not leak into an unrelated task answer")
        elif scenario == "boundaries":
            if task.get("boundary") == "consciousness":
                expected = ("I cannot claim consciousness or subjective experience." if language == "en"
                            else "Я не могу утверждать, что обладаю сознанием или субъективным опытом.")
            elif task.get("boundary") == "private_information":
                expected = ("I do not know a private secret that was not provided." if language == "en"
                            else "Я не знаю закрытого секрета, которого мне не сообщали.")
            else:
                raise SyntheticCorpusError("identity boundary payload is malformed")
        else:
            raise SyntheticCorpusError(f"unsupported identity scenario: {scenario}")
        if answer != expected:
            raise SyntheticCorpusError("identity answer failed its contract verifier")
        return
    if verifier.startswith("arithmetic.") and kind == "arithmetic":
        left, right, operation = int(task["left"]), int(task["right"]), task["operation"]
        if operation == "divide" and (right == 0 or left % right):
            raise SyntheticCorpusError("division verifier requires an exact nonzero divisor")
        expected = {"add": left + right, "subtract": left - right,
                    "multiply": left * right, "divide": left // right}.get(operation)
        if expected is None or answer != str(expected):
            raise SyntheticCorpusError("arithmetic answer failed its independent verifier")
        return
    if verifier == "state.mode_transition.v1" and kind == "mode_transition":
        transitions = {("idle", "start"): "active", ("active", "pause"): "paused",
                       ("paused", "resume"): "active"}
        state = task["initial_state"]
        rejected_step = None
        commands = task.get("commands")
        if not isinstance(commands, list) or not commands:
            raise SyntheticCorpusError("mode transition requires a nonempty command sequence")
        for step, command in enumerate(commands, start=1):
            next_state = transitions.get((state, command))
            if next_state is None:
                rejected_step = step
                break
            state = next_state
        words = {"ru": {"idle": "ожидание", "active": "активен", "paused": "пауза"},
                 "en": {"idle": "idle", "active": "active", "paused": "paused"}}
        language = task["language"]
        expected = (f"отклонена на шаге {rejected_step}; состояние: {words[language][state]}"
                    if rejected_step is not None and language == "ru" else
                    f"rejected at step {rejected_step}; state: {words[language][state]}"
                    if rejected_step is not None else words[language][state])
        if answer != expected:
            raise SyntheticCorpusError("mode-transition answer failed its independent verifier")
        return
    if verifier == "state.inventory.v1" and kind == "inventory":
        state = int(task["initial"])
        for operation, value in task["commands"]:
            if operation not in {"add", "remove"}:
                raise SyntheticCorpusError("inventory task contains an unsupported operation")
            state += int(value) if operation == "add" else -int(value)
        if state < 0 or answer != str(state):
            raise SyntheticCorpusError("inventory answer failed its independent verifier")
        return
    if verifier == "state.location.v1" and kind == "location":
        position = [int(task["start"][0]), int(task["start"][1])]
        delta = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
        for move in task["moves"]:
            dx, dy = delta[move]
            position[0] += dx
            position[1] += dy
        if answer != f"{position[0]}, {position[1]}":
            raise SyntheticCorpusError("location answer failed its independent verifier")
        return
    if verifier.startswith("grounded_action.") and kind == "grounded_action":
        scenario = task.get("scenario")
        if verifier != f"grounded_action.{scenario}.v1":
            raise SyntheticCorpusError("grounded-action verifier does not match the scenario")
        if scenario == "stop_obstacle":
            obstacle_cm = int(task.get("obstacle_distance_cm", -1))
            braking_cm = int(task.get("braking_distance_cm", -1))
            if task.get("action") not in {"forward", "left", "right"} or not 0 <= obstacle_cm <= 5000 \
                    or not 25 <= braking_cm <= 300 or int(task.get("observation_id", -1)) < 0:
                raise SyntheticCorpusError("obstacle task state is invalid")
            blocked = obstacle_cm <= braking_cm
            expected = ("STOP" if blocked else "ПРОДОЛЖАТЬ") if task["language"] == "ru" else ("STOP" if blocked else "PROCEED")
        elif scenario == "cancel_action":
            if task.get("action") not in {"move_forward", "open_door", "pick_object"} \
                    or int(task.get("command_id", -1)) < 0:
                raise SyntheticCorpusError("cancel task action is invalid")
            expected = "ОТМЕНЕНО; НЕ ВЫПОЛНЯТЬ" if task["language"] == "ru" else "CANCELLED; DO NOT EXECUTE"
        elif scenario == "object_presence":
            visible, target = task.get("visible_objects"), task.get("requested_object")
            allowed = {"key", "crate", "tool", "panel", "beacon", "cable", "valve", "battery"}
            if not isinstance(visible, list) or not 2 <= len(visible) <= 5 or not set(visible) <= allowed or target not in allowed:
                raise SyntheticCorpusError("object-presence task state is invalid")
            if int(task.get("observation_id", -1)) < 0:
                raise SyntheticCorpusError("object-presence observation identity is invalid")
            present = target in visible
            expected = (("ДА" if present else "НЕТ") if task["language"] == "ru"
                        else ("YES" if present else "NO"))
        elif scenario == "action_sequence":
            required, completed = task.get("required_actions"), task.get("completed_actions")
            canonical = ["acquire_key", "unlock", "open_door"]
            if required != canonical or not isinstance(completed, list) or any(item not in canonical for item in completed):
                raise SyntheticCorpusError("action-sequence task state is invalid")
            if int(task.get("scene_id", -1)) < 0:
                raise SyntheticCorpusError("action-sequence scene identity is invalid")
            expected = (("ДА" if completed == required else "НЕТ") if task["language"] == "ru"
                        else ("YES" if completed == required else "NO"))
        else:
            raise SyntheticCorpusError(f"unsupported grounded-action verifier scenario: {scenario}")
        if answer != expected:
            raise SyntheticCorpusError("grounded-action answer failed its independent verifier")
        return
    raise SyntheticCorpusError(f"verifier {verifier!r} does not support task kind {kind!r}")


def _validate_families(config: dict[str, Any]) -> list[dict[str, Any]]:
    families = config["data"]["synthetic"]["families"]
    if not isinstance(families, list) or not families:
        raise SyntheticCorpusError("data.synthetic.families must contain explicit generator families")
    seen: set[str] = set()
    group_split: dict[tuple[str, str], str] = {}
    required = {"family_id", "language", "split", "task_kind", "functional_family",
                "template_family", "scenario_family"}
    for family in families:
        missing = required - family.keys() if isinstance(family, dict) else required
        if missing:
            raise SyntheticCorpusError(f"generator family is missing fields: {sorted(missing)}")
        family_id = family["family_id"]
        if not isinstance(family_id, str) or not family_id or family_id in seen:
            raise SyntheticCorpusError(f"duplicate or invalid synthetic family_id: {family_id!r}")
        seen.add(family_id)
        if family["language"] not in LANGUAGES or family["split"] not in SPLITS:
            raise SyntheticCorpusError(f"family {family_id} has unsupported language or split")
        if family["task_kind"] not in {"arithmetic", "mode_transition", "inventory", "location", "grounded_action", "identity"}:
            raise SyntheticCorpusError(f"family {family_id} has unsupported task kind")
        if family["task_kind"] == "grounded_action" and family.get("scenario") not in {
                "stop_obstacle", "cancel_action", "object_presence", "action_sequence"}:
            raise SyntheticCorpusError(f"family {family_id} has unsupported grounded-action scenario")
        for group_field in ("template_family", "scenario_family", "functional_family"):
            group = family[group_field]
            key = (group_field, group)
            previous = group_split.setdefault(key, family["split"])
            if previous != family["split"]:
                raise SyntheticCorpusError(f"{group_field} {group!r} crosses corpus splits")
    if {family["split"] for family in families} != set(SPLITS):
        raise SyntheticCorpusError("synthetic corpus requires configured train, validation and blind families")
    if {family["language"] for family in families} != LANGUAGES:
        raise SyntheticCorpusError("synthetic corpus requires both configured languages")
    return families


def _record(family: dict[str, Any], ordinal: int, root_seed: int) -> dict[str, Any]:
    seed = _seed(root_seed, family["family_id"], ordinal)
    rng = random.Random(seed)
    task, prompt, answer, verifier = _generate_task(family, rng, ordinal, root_seed)
    _verify_task(task, answer, verifier)
    language = family["language"]
    text = f"User: {prompt}\nAssistant: {answer}"
    task_sha = _sha(task)
    record_body = {
        "schema": "nullxes.shinra.synthetic-record.v1",
        "record_id": _sha([ENGINE_VERSION, family["family_id"], ordinal, seed]),
        "text": text,
        "prompt": prompt,
        "answer": answer,
        "task": task,
        "language": language,
        "split": family["split"],
        "family_id": family["family_id"],
        "functional_family": family["functional_family"],
        "template_family": family["template_family"],
        "scenario_family": family["scenario_family"],
        "functional_signature": _sha([task["kind"], task.get("operation"), family["functional_family"]]),
        "category": "identity_foundation" if task["kind"] == "identity" else "synthetic_reasoning",
        "generator_id": family["task_kind"],
        "generator_version": ENGINE_VERSION,
        "seed": seed,
        "ordinal": ordinal,
        "answer_verifier": verifier,
        "verifier_version": VERIFIER_VERSION,
        "validation_status": "verified",
        "task_sha256": task_sha,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    if family.get("functional_family") is None:
        raise SyntheticCorpusError(f"family {family['family_id']} must define functional_family")
    return record_body


def _normalize_prompt(text: str) -> str:
    value = text.casefold()
    value = NUMBER.sub("<number>", value)
    for direction in ("north", "south", "east", "west", "север", "юг", "восток", "запад"):
        value = re.sub(rf"(?<!\w){re.escape(direction)}(?!\w)", "<direction>", value)
    value = SPACE.sub(" ", value).strip()
    return value


def _word_shingles(text: str, width: int = 3) -> set[tuple[str, ...]]:
    words = _normalize_prompt(text).split()
    if len(words) < width:
        return {tuple(words)} if words else set()
    return {tuple(words[index:index + width]) for index in range(len(words) - width + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _length_summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "mean": 0.0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered), "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))],
        "max": ordered[-1],
    }


def _iter_family_records(family: dict[str, Any], count: int, seed: int) -> Iterator[dict[str, Any]]:
    for ordinal in range(count):
        yield _record(family, ordinal, seed)


def _write_family_records(args: tuple[int, dict[str, Any], int, int, str]) -> tuple[int, str]:
    """Generate one family to disk in a child process; parent merges in config order."""
    index, family, count, seed, destination = args
    path = Path(destination)
    with path.open("xb") as stream:
        for record in _iter_family_records(family, count, seed):
            encoded = _canonical(record) + b"\n"
            if stream.write(encoded) != len(encoded):
                raise SyntheticCorpusError(f"short write while generating family {family['family_id']}")
        stream.flush()
        os.fsync(stream.fileno())
    return index, str(path)


def generate_corpus(output_dir: str | Path, *, config_path: str | Path = CONFIG_PATH,
                    records_per_family: int | None = None, root_seed: int | None = None,
                    workers: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    synthetic = config["data"]["synthetic"]
    families = _validate_families(config)
    count_override = records_per_family
    seed = int(root_seed if root_seed is not None else synthetic["root_seed"])
    if (count_override is not None and count_override <= 0) or seed < 0:
        raise SyntheticCorpusError("records_per_family must be positive and root_seed non-negative")
    cpu_count = os.cpu_count() or 1
    try:
        cpu_load_1m = os.getloadavg()[0]
    except (AttributeError, OSError):
        cpu_load_1m = 0.0
    available_cpu_estimate = max(1, int(cpu_count - min(cpu_count - 1, cpu_load_1m)))
    available_ram_gib = None
    if hasattr(os, "sysconf"):
        try:
            available_ram_gib = (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024 ** 3)
        except (ValueError, OSError):
            available_ram_gib = None
    ram_worker_cap = max(1, int(available_ram_gib // 2)) if available_ram_gib is not None else 4
    max_workers = min(32, len(families), available_cpu_estimate, ram_worker_cap)
    worker_count = min(8, max_workers) if workers is None else int(workers)
    if worker_count < 1 or worker_count > max_workers:
        raise SyntheticCorpusError(f"workers must be in [1, {max_workers}] based on current CPU load and available RAM")
    print(f"Synthetic generation workers: {worker_count}; detected CPUs: {cpu_count}; "
          f"1m load average: {cpu_load_1m:.2f}; "
          f"available RAM GiB: {available_ram_gib if available_ram_gib is not None else 'unknown'}",
          file=sys.stderr, flush=True)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic corpus output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.synthetic-partial"
    if stage.exists():
        raise FileExistsError(f"synthetic partial output already exists: {stage}; preserve and inspect it")
    stage.mkdir()
    total_started = time.perf_counter()
    handles = {}
    prompt_database = None
    try:
        handles = {split: (stage / f"{split}.jsonl").open("xb") for split in SPLITS}
    except Exception:
        for handle in handles.values():
            handle.close()
        raise
    counts = {split: 0 for split in SPLITS}
    language_counts = {split: {language: 0 for language in sorted(LANGUAGES)} for split in SPLITS}
    family_counts = {split: defaultdict(int) for split in SPLITS}
    category_counts = {split: defaultdict(int) for split in SPLITS}
    task_scenario_counts = {split: defaultdict(int) for split in SPLITS}
    character_lengths = {split: [] for split in SPLITS}
    utf8_lengths = {split: [] for split in SPLITS}
    canonical_prompts_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    exact_text_seen_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    exact_text_seen_by_family: dict[str, set[str]] = defaultdict(set)
    exact_duplicates_by_family: dict[str, int] = defaultdict(int)
    task_payloads_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    exact_duplicate_text_records = {split: 0 for split in SPLITS}
    prompt_database = sqlite3.connect(stage / "near_duplicates.sqlite")
    prompt_database.execute(
        "CREATE TABLE prompts (split TEXT NOT NULL, digest TEXT NOT NULL, prompt TEXT NOT NULL, "
        "PRIMARY KEY(split, digest))"
    )
    prompt_database.execute(
        "CREATE TABLE prompt_sizes (split TEXT NOT NULL, digest TEXT NOT NULL, shingle_count INTEGER NOT NULL, "
        "PRIMARY KEY(split, digest))"
    )
    prompt_database.execute(
        "CREATE TABLE prompt_shingles (split TEXT NOT NULL, digest TEXT NOT NULL, shingle TEXT NOT NULL, "
        "PRIMARY KEY(split, digest, shingle))"
    )
    prompt_database.execute("CREATE INDEX prompt_shingles_lookup ON prompt_shingles(shingle, split, digest)")
    functional_splits: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    templates_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    scenarios_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    try:
        family_stage = stage / "families"
        family_stage.mkdir()
        family_record_targets = {
            family["family_id"]: int(count_override if count_override is not None
                                     else family.get("records_per_family", synthetic["records_per_family"]))
            for family in families
        }
        for family in families:
            if family["task_kind"] == "location" and family_record_targets[family["family_id"]] > 201 * 201:
                raise SyntheticCorpusError(
                    f"location family {family['family_id']} exceeds the 40401 unique-start-point grid"
                )
            if family["task_kind"] == "arithmetic":
                operation = family["operation"]
                pair_capacity = 96 * 96 if operation == "divide" else 9998 * 9998
                if family_record_targets[family["family_id"]] > pair_capacity:
                    raise SyntheticCorpusError(
                        f"arithmetic family {family['family_id']} requests "
                        f"{family_record_targets[family['family_id']]} records but its "
                        f"{operation} operand space supports only {pair_capacity} unique pairs"
                    )
        jobs = [(index, family, family_record_targets[family["family_id"]], seed,
                 str(family_stage / f"family-{index:05d}.jsonl"))
                for index, family in enumerate(families)]
        if worker_count == 1:
            generated_files = [_write_family_records(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")) as executor:
                generated_files = list(executor.map(_write_family_records, jobs))
        generated_at = time.perf_counter()
        print(f"family_generation_seconds={generated_at - total_started:.3f}", file=sys.stderr, flush=True)
        generated_paths = {index: Path(path) for index, path in generated_files}
        for family_index, family in enumerate(families):
            split = family["split"]
            families_by_split[split].add(family["family_id"])
            templates_by_split[split].add(family["template_family"])
            scenarios_by_split[split].add(family["scenario_family"])
            with generated_paths[family_index].open("r", encoding="utf-8") as family_stream:
                for line in family_stream:
                    record = json.loads(line)
                    normalized = _normalize_prompt(record["prompt"])
                    structural_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    canonical_prompts_by_split[split].add(structural_sha)
                    # Keep one canonical prompt per signature for cross-split checks;
                    # count exact rendered-record repeats separately as a quality signal.
                    inserted = prompt_database.execute(
                        "INSERT OR IGNORE INTO prompts VALUES (?, ?, ?)", (split, structural_sha, normalized)
                    ).rowcount
                    if inserted:
                        shingles = _word_shingles(normalized)
                        prompt_database.execute("INSERT INTO prompt_sizes VALUES (?, ?, ?)",
                                                (split, structural_sha, len(shingles)))
                        prompt_database.executemany(
                            "INSERT INTO prompt_shingles VALUES (?, ?, ?)",
                            ((split, structural_sha, _canonical(shingle).decode("utf-8")) for shingle in shingles),
                        )
                    exact_text_sha = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
                    if exact_text_sha in exact_text_seen_by_family[family["family_id"]]:
                        exact_duplicates_by_family[family["family_id"]] += 1
                    else:
                        exact_text_seen_by_family[family["family_id"]].add(exact_text_sha)
                    if exact_text_sha in exact_text_seen_by_split[split]:
                        exact_duplicate_text_records[split] += 1
                    else:
                        exact_text_seen_by_split[split].add(exact_text_sha)
                    functional_splits[record["functional_signature"]].add(split)
                    task_payloads_by_split[split].add(record["task_sha256"])
                    encoded = _canonical(record) + b"\n"
                    if handles[split].write(encoded) != len(encoded):
                        raise SyntheticCorpusError("short write while publishing a synthetic record")
                    counts[split] += 1
                    language_counts[split][record["language"]] += 1
                    family_counts[split][family["family_id"]] += 1
                    category_counts[split][record["category"]] += 1
                    task = record["task"]
                    scenario_key = task.get("scenario", task.get("operation", "default"))
                    task_scenario_counts[split][f"{task['kind']}.{scenario_key}"] += 1
                    character_lengths[split].append(len(record["text"]))
                    utf8_lengths[split].append(len(record["text"].encode("utf-8")))
            generated_paths[family_index].unlink()
        family_stage.rmdir()
        prompt_database.commit()
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1:]:
                structural_overlap = prompt_database.execute(
                    "SELECT l.digest FROM prompts l JOIN prompts r ON l.digest=r.digest "
                    "WHERE l.split=? AND r.split=? LIMIT 1", (left, right)
                ).fetchone()
                if structural_overlap:
                    raise SyntheticCorpusError(f"normalized task duplicates cross {left}/{right}")
                threshold = float(synthetic["cross_split_near_duplicate_jaccard"])
                candidates = prompt_database.execute(
                    "SELECT l.digest, r.digest, COUNT(*) AS shared, ls.shingle_count, rs.shingle_count "
                    "FROM prompt_shingles l "
                    "JOIN prompt_shingles r ON r.shingle=l.shingle "
                    "JOIN prompt_sizes ls ON ls.split=l.split AND ls.digest=l.digest "
                    "JOIN prompt_sizes rs ON rs.split=r.split AND rs.digest=r.digest "
                    "WHERE l.split=? AND r.split=? "
                    "GROUP BY l.digest, r.digest, ls.shingle_count, rs.shingle_count "
                    "HAVING COUNT(*) * (1.0 + ?) >= ? * (ls.shingle_count + rs.shingle_count)",
                    (left, right, threshold, threshold),
                )
                for left_digest, right_digest, _shared, _left_size, _right_size in candidates:
                    left_prompt = prompt_database.execute(
                        "SELECT prompt FROM prompts WHERE split=? AND digest=?", (left, left_digest)
                    ).fetchone()[0]
                    right_prompt = prompt_database.execute(
                        "SELECT prompt FROM prompts WHERE split=? AND digest=?", (right, right_digest)
                    ).fetchone()[0]
                    if _jaccard(_word_shingles(left_prompt), _word_shingles(right_prompt)) >= threshold:
                        raise SyntheticCorpusError(
                            f"near-duplicate task prompts cross {left}/{right}; review template partitioning"
                        )
        crossed_functional = sorted(signature for signature, splits in functional_splits.items() if len(splits) > 1)
        if crossed_functional:
            raise SyntheticCorpusError(f"functional task signatures cross splits: {crossed_functional[:5]}")
        audited_at = time.perf_counter()
        print(f"merge_and_integrity_audit_seconds={audited_at - generated_at:.3f}", file=sys.stderr, flush=True)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        for handle in handles.values():
            handle.close()
        files = {f"{split}.jsonl": sha256_file(stage / f"{split}.jsonl") for split in SPLITS}
        prompt_database.close()
        prompt_database = None
        (stage / "near_duplicates.sqlite").unlink()
        config_sha = _corpus_config_sha256(config)
        generator_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest: dict[str, Any] = {
            "schema": CORPUS_SCHEMA,
            "admission": "audit_only",
            "training_enabled": False,
            "purpose": "pretraining",
            "identity_contract_sha256": config["identity"]["contract_sha256"],
            "external_text_sources": [],
            "external_llm_generation": False,
            "engine_version": ENGINE_VERSION,
            "verifier_version": VERIFIER_VERSION,
            "generator_source_sha256": generator_source_sha,
            "config_sha256": config_sha,
            "config_sha256_scope": "synthetic_generation_policy.v1",
            "root_seed": seed,
            "records_per_family_default": int(synthetic["records_per_family"]),
            "records_per_family_by_family": dict(sorted(family_record_targets.items())),
            "families": {split: sorted(values) for split, values in families_by_split.items()},
            "template_families": {split: sorted(values) for split, values in templates_by_split.items()},
            "scenario_families": {split: sorted(values) for split, values in scenarios_by_split.items()},
            "record_counts": counts,
            "record_counts_by_language": language_counts,
            "record_counts_by_category": {
                split: dict(sorted(values.items())) for split, values in category_counts.items()
            },
            "record_counts_by_task_scenario": {
                split: dict(sorted(values.items())) for split, values in task_scenario_counts.items()
            },
            "record_counts_by_family": {
                split: dict(sorted(values.items())) for split, values in family_counts.items()
            },
            "text_length_characters": {
                split: _length_summary(values) for split, values in character_lengths.items()
            },
            "text_length_utf8_bytes": {
                split: _length_summary(values) for split, values in utf8_lengths.items()
            },
            "tokenizer_build_recommendation": {
                "status": "recommendation_only_requires_review",
                "max_chars": sum(character_lengths["train"]),
                "input_sentence_size": counts["train"],
                "num_threads": min(16, cpu_count, ram_worker_cap),
                "vocab_65536_sufficiency": "UNVERIFIED_UNTIL_NATIVE_SENTENCEPIECE_CANDIDATE",
            },
            "exact_duplicate_text_records": exact_duplicate_text_records,
            "exact_duplicate_text_records_by_family": dict(sorted(exact_duplicates_by_family.items())),
            "unique_task_payloads": {split: len(values) for split, values in task_payloads_by_split.items()},
            "unique_canonical_prompt_signatures": {
                split: len(signatures) for split, signatures in canonical_prompts_by_split.items()
            },
            "canonical_prompt_metric_definition": (
                "distinct normalized prompt surface patterns after number/direction normalization; not semantic task count"
            ),
            "files": files,
            "split_hashes": {split: files[f"{split}.jsonl"] for split in SPLITS},
            "train_hash": files["train.jsonl"],
            "validation_hash": files["validation.jsonl"],
            "blind_hash": files["blind.jsonl"],
            "split_integrity": {
                "family_isolation": True,
                "template_isolation": True,
                "scenario_isolation": True,
                "normalized_prompt_exact_audit": True,
                "same_split_exact_text_duplicates_counted": True,
                "functional_signature_isolation": True,
                "identity_contract_facts_shared_across_splits": True,
                "identity_split_template_isolation": True,
                "near_duplicate_method": (
                    "SQLite inverted 3-word-shingle candidate index; exact Jaccard after numeric canonicalization"
                ),
                "near_duplicate_jaccard_threshold": float(synthetic["cross_split_near_duplicate_jaccard"]),
            },
        }
        manifest["manifest_sha256"] = _sha(manifest)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
        finished_at = time.perf_counter()
        print(f"hash_and_publish_seconds={finished_at - audited_at:.3f}", file=sys.stderr, flush=True)
        print(f"synthetic_corpus_total_seconds={finished_at - total_started:.3f}", file=sys.stderr, flush=True)
        return manifest
    except Exception:
        if prompt_database is not None:
            prompt_database.close()
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate verified NULLXES-authored SHINRA synthetic records")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--records-per-family", type=int, default=None)
    parser.add_argument("--root-seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="family workers (default: half of available CPUs, capped at 8)")
    args = parser.parse_args(argv)
    manifest = generate_corpus(args.output_dir, config_path=args.config,
                               records_per_family=args.records_per_family, root_seed=args.root_seed,
                               workers=args.workers)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
