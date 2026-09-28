"""Seeded document generators. No LLM. Hidden graphs stay in semantics."""

from __future__ import annotations

import random

from data.synth.schema import LAYERS

EN_NAMES = ("Mira", "Tomas", "Alina", "Gleb", "Nadir", "Irena", "Pavel", "Saskia")
RU_NAMES = ("Мира", "Томас", "Алина", "Глеб", "Надир", "Ирена", "Павел", "Саския")
EN_NOUNS = ("river", "window", "stone", "lamp", "bridge", "kettle", "notebook", "gate")
RU_NOUNS = ("река", "окно", "камень", "лампа", "мост", "чайник", "тетрадь", "ворота")
EN_ADJ = ("quiet", "narrow", "cold", "bright", "heavy", "empty", "open", "dry")
RU_ADJ = ("тихая", "узкая", "холодная", "яркая", "тяжёлая", "пустая", "открытая", "сухая")
PLACES = ("Kassel", "Ostrow", "Selva", "Marlow")
RU_PLACES = ("Кассель", "Остров", "Сельва", "Марлоу")


def _pick(rng: random.Random, items: tuple[str, ...]) -> str:
    return items[rng.randrange(len(items))]


def generate_language_core(rng: random.Random, language: str) -> tuple[str, dict]:
    if language == "en":
        name = _pick(rng, EN_NAMES)
        noun = _pick(rng, EN_NOUNS)
        adj = _pick(rng, EN_ADJ)
        tense = rng.choice(("present", "past", "plural"))
        if tense == "present":
            text = f"{name} keeps a {adj} {noun} near the door."
        elif tense == "past":
            text = f"Yesterday {name} moved the {adj} {noun}."
        else:
            text = f"The {noun}s were {adj}, and {name} counted them twice."
        punct = rng.choice((".", "!", "..."))
        text = text[:-1] + punct
        semantics = {"morphology": tense, "head": noun, "modifier": adj, "agent": name}
        return text, semantics
    name = _pick(rng, RU_NAMES)
    noun = _pick(rng, RU_NOUNS)
    adj = _pick(rng, RU_ADJ)
    case = rng.choice(("nom", "acc", "prep", "neg"))
    if case == "nom":
        text = f"{name} видит {adj} {noun}."
    elif case == "acc":
        text = f"{name} отодвинула {noun} к стене."
    elif case == "prep":
        text = f"На столе лежала {adj} {noun}."
    else:
        text = f"{name} не трогала {noun}."
    if rng.random() < 0.25:
        text = text[:-1] + "?"
    semantics = {"morphology": case, "head": noun, "modifier": adj, "agent": name}
    return text, semantics


def generate_semantic_primitives(rng: random.Random, language: str) -> tuple[str, dict]:
    kind = rng.choice(("relation", "negation", "coref", "temporal", "comparison"))
    if language == "en":
        a, b = _pick(rng, EN_NAMES), _pick(rng, EN_NAMES)
        place = _pick(rng, PLACES)
        if kind == "relation":
            text = f"{a} helped {b} cross the bridge in {place}."
            graph = {"rel": "helped", "src": a, "dst": b, "loc": place}
        elif kind == "negation":
            text = f"{a} did not stay in {place}. {b} waited there anyway."
            graph = {"rel": "not_located", "src": a, "loc": place, "other": b}
        elif kind == "coref":
            text = f"{a} found a notebook. She left it on the kettle."
            graph = {"entity": a, "pronoun": "she", "object": "notebook"}
        elif kind == "temporal":
            text = f"Before dawn {a} left {place}. After that {b} locked the gate."
            graph = {"before": a, "after": b, "loc": place}
        else:
            text = f"{a} is taller than {b}, but {b} is quicker in {place}."
            graph = {"cmp": "taller", "more": a, "less": b, "loc": place}
        return text, {"kind": kind, "graph": graph}
    a, b = _pick(rng, RU_NAMES), _pick(rng, RU_NAMES)
    place = _pick(rng, RU_PLACES)
    if kind == "relation":
        text = f"{a} помогла {b} пройти мост в городе {place}."
        graph = {"rel": "helped", "src": a, "dst": b, "loc": place}
    elif kind == "negation":
        text = f"{a} не осталась в городе {place}. {b} всё равно ждал у ворот."
        graph = {"rel": "not_located", "src": a, "loc": place, "other": b}
    elif kind == "coref":
        text = f"{a} нашла тетрадь. Она положила её рядом с чайником."
        graph = {"entity": a, "pronoun": "она", "object": "тетрадь"}
    elif kind == "temporal":
        text = f"До рассвета {a} ушла из {place}. Потом {b} закрыл ворота."
        graph = {"before": a, "after": b, "loc": place}
    else:
        text = f"{a} выше, чем {b}, но {b} быстрее в {place}."
        graph = {"cmp": "taller", "more": a, "less": b, "loc": place}
    return text, {"kind": kind, "graph": graph}


def generate_compositional(rng: random.Random, language: str) -> tuple[str, dict]:
    obj = rng.choice(("cup", "gate", "lamp", "kettle"))
    start, end = rng.choice((("empty", "full"), ("closed", "open"), ("dark", "bright"), ("cold", "warm")))
    if language == "en":
        name = _pick(rng, EN_NAMES)
        texts = {
            "cup": (
                f"The cup was {start}. Then {name} poured water. The cup was {end}.",
                "cup",
            ),
            "gate": (
                f"The gate was {start}. {name} pushed the latch. The gate was {end}.",
                "gate",
            ),
            "lamp": (
                f"The lamp was {start}. {name} turned the switch. The lamp was {end}.",
                "lamp",
            ),
            "kettle": (
                f"The kettle was {start}. {name} set it on the stove. The kettle was {end}.",
                "kettle",
            ),
        }
        text, obj_id = texts[obj]
        return text, {"object": obj_id, "start": start, "end": end, "agent": name}
    name = _pick(rng, RU_NAMES)
    ru = {
        "cup": (f"Чашка была пустой. Потом {name} налила воду. Чашка стала полной.", "чашка", "пустой", "полный"),
        "gate": (f"Ворота были закрыты. {name} сдвинула засов. Ворота стали открытыми.", "ворота", "закрытый", "открытый"),
        "lamp": (f"Лампа была тёмной. {name} щелкнула выключателем. Лампа стала яркой.", "лампа", "тёмный", "яркий"),
        "kettle": (f"Чайник был холодным. {name} поставила его на плиту. Чайник стал тёплым.", "чайник", "холодный", "тёплый"),
    }
    text, obj_id, start_ru, end_ru = ru[obj]
    return text, {"object": obj_id, "start": start_ru, "end": end_ru, "agent": name}


def generate_knowledge_shaped(rng: random.Random, language: str) -> tuple[str, dict]:
    topic = rng.choice(("lerna", "varek", "k12", "minted"))
    minted = f"Nex-{rng.randint(3, 9)}"
    if language == "en":
        facts = {
            "lerna": (
                "Lerna-7 is a sealed research station north of the Varek. Night crews log ice noise, not traffic.",
                "Lerna-7",
            ),
            "varek": (
                "The Varek is a slow river that freezes from the banks inward. Barges wait at Marlow until the channel opens.",
                "Varek",
            ),
            "k12": (
                "K-12 is a pale mineral that darkens in ammonia vapor. Surveyors mark it with a dry charcoal tick.",
                "K-12",
            ),
            "minted": (
                f"{minted} is a dry plateau east of Lerna-7. Wind strips the soil down to grey plate rock.",
                minted,
            ),
        }
        text, entity = facts[topic]
        return text, {"entity": entity, "invented": True, "family": topic}
    facts = {
        "lerna": (
            "Лерна-7 — закрытая станция к северу от Варека. Ночные смены пишут шум льда, а не движение.",
            "Лерна-7",
        ),
        "varek": (
            "Варек — медленная река, которая стынет от берегов к середине. Баржи ждут у Марлоу, пока не откроется фарватер.",
            "Варек",
        ),
        "k12": (
            "K-12 — бледный минерал, темнеющий в парах аммиака. Съёмщики отмечают его сухой угольной чертой.",
            "K-12",
        ),
        "minted": (
            f"{minted} — сухое плато к востоку от Лерны-7. Ветер счищает почву до серой плитняковой породы.",
            minted,
        ),
    }
    text, entity = facts[topic]
    return text, {"entity": entity, "invented": True, "family": topic}


def generate_structured(rng: random.Random, language: str) -> tuple[str, dict]:
    kind = rng.choice(("arith", "list", "json", "code"))
    a, b = rng.randint(0, 40), rng.randint(0, 40)
    if kind == "arith":
        op = rng.choice(("+", "-", "*"))
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        else:
            result = a * b
        if language == "en":
            text = f"Compute {a} {op} {b}. The value is {result}."
        else:
            text = f"Вычислить {a} {op} {b}. Значение равно {result}."
        return text, {"kind": kind, "a": a, "b": b, "op": op, "result": result}
    if kind == "list":
        items = [_pick(rng, EN_NOUNS) for _ in range(3)]
        if language == "en":
            text = "Items: " + "; ".join(f"{i + 1}. {item}" for i, item in enumerate(items)) + "."
        else:
            ru_items = [_pick(rng, RU_NOUNS) for _ in range(3)]
            text = "Список: " + "; ".join(f"{i + 1}. {item}" for i, item in enumerate(ru_items)) + "."
            items = ru_items
        return text, {"kind": kind, "items": items}
    if kind == "json":
        key = rng.choice(("mass", "count", "depth"))
        payload = {key: a, "ok": True}
        text = "{" + f'"{key}": {a}, "ok": true' + "}"
        return text, {"kind": kind, "payload": payload}
    fn = rng.choice(("add", "span"))
    if fn == "add":
        text = "def add(x, y):\n    return x + y\n"
    else:
        text = "def span(xs):\n    return xs[-1] - xs[0]\n"
    return text, {"kind": kind, "fn": fn}


GENERATORS = {
    "language_core": generate_language_core,
    "semantic_primitives": generate_semantic_primitives,
    "compositional": generate_compositional,
    "knowledge_shaped": generate_knowledge_shaped,
    "structured": generate_structured,
}

GENERATOR_IDS = {layer: f"{layer}.v0" for layer in LAYERS}
