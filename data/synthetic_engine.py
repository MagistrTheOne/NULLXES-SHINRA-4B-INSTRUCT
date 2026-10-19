"""Deterministic, verifier-backed NULLXES synthetic corpus generation.

This module has no network, model, or third-party text-generation dependency.
Every record is produced from an explicit family specification and checked by
an independent deterministic verifier before it is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from configs.loader import CONFIG_PATH, load_config
from data.hashing import sha256_file

ENGINE_VERSION = "nullxes-synthetic-engine.v1"
CORPUS_SCHEMA = "nullxes.shinra.synthetic-corpus.v1"
VERIFIER_VERSION = "nullxes-synthetic-verifiers.v1"
SPLITS = ("train", "validation", "blind")
LANGUAGES = {"en", "ru"}
NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?!\w)")
SPACE = re.compile(r"\s+")


class SyntheticCorpusError(RuntimeError):
    """Synthetic corpus configuration, generation, or integrity failure."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed(root_seed: int, family_id: str, ordinal: int) -> int:
    payload = f"{ENGINE_VERSION}\0{root_seed}\0{family_id}\0{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


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
    transitions = {"idle": {"start": "active"}, "active": {"pause": "paused"},
                   "paused": {"resume": "active"}}
    state = before
    command_text: list[str] = []
    for command in commands:
        command_text.append(verbs[language][command])
        state = transitions[state][command]
    if language == "ru":
        prompt = (f"Состояние устройства: {mode_words[language][before]}. Команды по порядку: "
                  f"{'; затем '.join(command_text)}. Каково конечное состояние? Ответь одним словом.")
    else:
        prompt = (f"Device state: {mode_words[language][before]}. Commands in order: "
                  f"{'; then '.join(command_text)}. What is the final state? Answer with one word.")
    return prompt, mode_words[language][state]


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


def _generate_task(family: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], str, str, str]:
    language = family["language"]
    kind = family["task_kind"]
    if kind == "arithmetic":
        operation = family["operation"]
        if operation == "divide":
            right = rng.randint(2, 97)
            left = right * rng.randint(2, 97)
        else:
            left, right = rng.randint(2, 9999), rng.randint(2, 9999)
        prompt, answer = _render_arithmetic(language, operation, left, right)
        task = {"kind": kind, "operation": operation, "left": left, "right": right}
        verifier = f"arithmetic.{operation}.v1"
    elif kind == "mode_transition":
        paths = {"idle": (["start"],), "active": (["pause"],), "paused": (["resume"],)}
        before = family["initial_state"]
        commands = list(paths[before][0])
        if family.get("two_step", False):
            followup = {"idle": "pause", "active": "resume", "paused": "pause"}.get(before)
            if followup:
                commands.append(followup)
        prompt, answer = _render_mode_transition(language, before, commands)
        task = {"kind": kind, "initial_state": before, "commands": commands}
        verifier = "state.mode_transition.v1"
    elif kind == "inventory":
        initial = rng.randint(20, 500)
        first = rng.randint(1, min(25, initial // 2))
        second = rng.randint(1, min(25, initial + first))
        commands = [("add", first), ("remove", second)]
        prompt, answer = _render_inventory(language, initial, commands)
        task = {"kind": kind, "initial": initial, "commands": commands}
        verifier = "state.inventory.v1"
    elif kind == "location":
        directions = ["north", "south", "east", "west"]
        start = (rng.randint(-100, 100), rng.randint(-100, 100))
        moves = [rng.choice(directions) for _ in range(rng.randint(2, 5))]
        prompt, answer = _render_location(language, start, moves)
        task = {"kind": kind, "start": start, "moves": moves}
        verifier = "state.location.v1"
    else:
        raise SyntheticCorpusError(f"unsupported task kind in family {family['family_id']}: {kind}")
    task["language"] = language
    return task, prompt, answer, verifier


def _verify_task(task: dict[str, Any], answer: str, verifier: str) -> None:
    kind = task.get("kind")
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
        for command in task["commands"]:
            try:
                state = transitions[(state, command)]
            except KeyError as exc:
                raise SyntheticCorpusError("mode transition contains an invalid command for the current state") from exc
        words = {"ru": {"idle": "ожидание", "active": "активен", "paused": "пауза"},
                 "en": {"idle": "idle", "active": "active", "paused": "paused"}}
        language = task["language"]
        if answer != words[language][state]:
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
        if not isinstance(family, dict) or required - family.keys():
            raise SyntheticCorpusError(f"generator family is missing fields: {sorted(required - family.keys())}")
        family_id = family["family_id"]
        if not isinstance(family_id, str) or not family_id or family_id in seen:
            raise SyntheticCorpusError(f"duplicate or invalid synthetic family_id: {family_id!r}")
        seen.add(family_id)
        if family["language"] not in LANGUAGES or family["split"] not in SPLITS:
            raise SyntheticCorpusError(f"family {family_id} has unsupported language or split")
        if family["task_kind"] not in {"arithmetic", "mode_transition", "inventory", "location"}:
            raise SyntheticCorpusError(f"family {family_id} has unsupported task kind")
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
    task, prompt, answer, verifier = _generate_task(family, rng)
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


def _iter_family_records(family: dict[str, Any], count: int, seed: int) -> Iterator[dict[str, Any]]:
    for ordinal in range(count):
        yield _record(family, ordinal, seed)


def generate_corpus(output_dir: str | Path, *, config_path: str | Path = CONFIG_PATH,
                    records_per_family: int | None = None, root_seed: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    synthetic = config["data"]["synthetic"]
    families = _validate_families(config)
    count = int(records_per_family if records_per_family is not None else synthetic["records_per_family"])
    seed = int(root_seed if root_seed is not None else synthetic["root_seed"])
    if count <= 0 or seed < 0:
        raise SyntheticCorpusError("records_per_family must be positive and root_seed non-negative")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic corpus output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.synthetic-partial"
    if stage.exists():
        raise FileExistsError(f"synthetic partial output already exists: {stage}; preserve and inspect it")
    stage.mkdir()
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
    exact_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    prompt_database = sqlite3.connect(stage / "near_duplicates.sqlite")
    prompt_database.execute(
        "CREATE TABLE prompts (split TEXT NOT NULL, digest TEXT NOT NULL, prompt TEXT NOT NULL, "
        "PRIMARY KEY(split, digest))"
    )
    functional_splits: dict[str, set[str]] = defaultdict(set)
    families_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    templates_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    scenarios_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    try:
        for family in families:
            split = family["split"]
            families_by_split[split].add(family["family_id"])
            templates_by_split[split].add(family["template_family"])
            scenarios_by_split[split].add(family["scenario_family"])
            for record in _iter_family_records(family, count, seed):
                normalized = _normalize_prompt(record["prompt"])
                structural_sha = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if structural_sha in exact_by_split[split]:
                    raise SyntheticCorpusError(f"within-split duplicate generated in {split}: {record['record_id']}")
                exact_by_split[split].add(structural_sha)
                prompt_database.execute("INSERT INTO prompts VALUES (?, ?, ?)",
                                        (split, structural_sha, record["prompt"]))
                functional_splits[record["functional_signature"]].add(split)
                encoded = _canonical(record) + b"\n"
                if handles[split].write(encoded) != len(encoded):
                    raise SyntheticCorpusError("short write while publishing a synthetic record")
                counts[split] += 1
                language_counts[split][record["language"]] += 1
        prompt_database.commit()
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1:]:
                structural_overlap = prompt_database.execute(
                    "SELECT l.digest FROM prompts l JOIN prompts r ON l.digest=r.digest "
                    "WHERE l.split=? AND r.split=? LIMIT 1", (left, right)
                ).fetchone()
                if structural_overlap:
                    raise SyntheticCorpusError(f"normalized task duplicates cross {left}/{right}")
                left_cursor = prompt_database.execute("SELECT prompt FROM prompts WHERE split=? ORDER BY digest", (left,))
                for (left_prompt,) in left_cursor:
                    left_shingles = _word_shingles(left_prompt)
                    right_cursor = prompt_database.execute("SELECT prompt FROM prompts WHERE split=? ORDER BY digest", (right,))
                    for (right_prompt,) in right_cursor:
                        if _jaccard(left_shingles, _word_shingles(right_prompt)) >= float(
                                synthetic["cross_split_near_duplicate_jaccard"]):
                            raise SyntheticCorpusError(
                                f"near-duplicate task prompts cross {left}/{right}; review template partitioning"
                            )
        crossed_functional = sorted(signature for signature, splits in functional_splits.items() if len(splits) > 1)
        if crossed_functional:
            raise SyntheticCorpusError(f"functional task signatures cross splits: {crossed_functional[:5]}")
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        for handle in handles.values():
            handle.close()
        files = {f"{split}.jsonl": sha256_file(stage / f"{split}.jsonl") for split in SPLITS}
        prompt_database.close()
        prompt_database = None
        (stage / "near_duplicates.sqlite").unlink()
        config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        generator_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest: dict[str, Any] = {
            "schema": CORPUS_SCHEMA,
            "admission": "audit_only",
            "training_enabled": False,
            "purpose": "pretraining",
            "external_text_sources": [],
            "external_llm_generation": False,
            "engine_version": ENGINE_VERSION,
            "verifier_version": VERIFIER_VERSION,
            "generator_source_sha256": generator_source_sha,
            "config_sha256": config_sha,
            "root_seed": seed,
            "records_per_family": count,
            "families": {split: sorted(values) for split, values in families_by_split.items()},
            "template_families": {split: sorted(values) for split, values in templates_by_split.items()},
            "scenario_families": {split: sorted(values) for split, values in scenarios_by_split.items()},
            "record_counts": counts,
            "record_counts_by_language": language_counts,
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
                "functional_signature_isolation": True,
                "near_duplicate_method": "3-word-shingle Jaccard after numeric canonicalization",
                "near_duplicate_jaccard_threshold": float(synthetic["cross_split_near_duplicate_jaccard"]),
            },
        }
        manifest["manifest_sha256"] = _sha(manifest)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
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
    args = parser.parse_args(argv)
    manifest = generate_corpus(args.output_dir, config_path=args.config,
                               records_per_family=args.records_per_family, root_seed=args.root_seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
