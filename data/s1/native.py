"""Deterministic NULLXES-native S1 generators.

Selection uses sha256(builder_version, namespace, record key). No process RNG.
English and Russian use separate surface pools and sentence templates.
"""

from __future__ import annotations

import hashlib

from data.s1.foundation import BUILDER_VERSION, make_record

CURRICULUM_VERSION = "s1-curriculum-pilot-v1"
NATIVE_LICENSE = "nullxes-internal"

EN_COLORS = ("red", "blue", "yellow", "green", "white", "brown", "grey", "orange")
RU_COLORS = ("красный", "синий", "жёлтый", "зелёный", "белый", "коричневый", "серый", "оранжевый")
EN_GOODS = ("pears", "nails", "maps", "linen", "lamps", "chalk", "bolts", "ribbon")
RU_GOODS = ("мёд", "гвозди", "карты", "лён", "лампы", "мел", "болты", "ленту")
EN_PLACES = ("Orchard Station", "Mill Yard", "Harbor Shed", "Northwind Dock", "Brindle Market", "Cedar Wharf")
RU_PLACES = ("порт Кедр", "мастерская Лён", "склад Тихий", "двор Север", "рынок Иволга", "пристань Клён")
EN_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
RU_DAYS = ("понедельник", "вторник", "среду", "четверг", "пятницу")
EN_NAMES = (
    "Ada Cole", "Bram Holt", "Nia Cho", "Owen Pike", "Vera Lang", "Cleo Marsh",
    "Hugo Venn", "Iris Dahl", "Jonah Reed", "Lila Frost", "Marco Bell", "Petra Quinn",
)
RU_NAMES = (
    "Ада Коль", "Борис Холт", "Ника Чо", "Олег Пайк", "Вера Ланг", "Клавдия Марш",
    "Глеб Венн", "Ирина Даль", "Иона Рид", "Лилия Фрост", "Марк Белл", "Петра Квинн",
)
EN_OBJECTS = ("clock", "ledger", "lantern", "crate", "key", "map case")
RU_OBJECTS = ("часы", "журнал", "фонарь", "ящик", "ключ", "футляр")
EN_SITES = ("the mill", "the harbor", "the archive", "the yard", "the dock", "the loft")
RU_SITES = ("мельница", "гавань", "архив", "двор", "пристань", "чердак")


def pick(namespace: str, key: str, options: tuple):
    digest = hashlib.sha256(f"{CURRICULUM_VERSION}|{namespace}|{key}".encode("utf-8")).digest()
    return options[int.from_bytes(digest[:8], "big") % len(options)]


def pick_n(namespace: str, key: str, options: tuple, count: int) -> tuple:
    chosen = []
    pool = list(options)
    salt = 0
    while len(chosen) < count:
        item = pick(namespace, f"{key}:{salt}", tuple(pool))
        chosen.append(item)
        pool = [value for value in pool if value != item] or list(options)
        salt += 1
    return tuple(chosen)


def _native(
    *,
    source_id: str,
    row_id: str,
    language: str,
    family: str,
    difficulty: str,
    context: str,
    question: str,
    answer: str,
    instruction: str,
    transform: str,
    extra: dict | None = None,
) -> dict:
    record = make_record(
        source_id=source_id,
        source_config="native",
        source_split="train",
        source_row_id=row_id,
        language=language,
        family=family,
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        license_name=NATIVE_LICENSE,
        transform=transform,
        instruction=instruction,
    )
    record["metadata"]["source_type"] = "NATIVE"
    record["metadata"]["builder_version"] = CURRICULUM_VERSION
    if extra:
        record["metadata"].update(extra)
    return record


def instruction_record(index: int, language: str, difficulty: str) -> dict:
    kind = index % 10
    world = index // 10
    if language == "en":
        colors = pick_n("inst-en-color", str(world), EN_COLORS, 3)
        goods = pick_n("inst-en-good", str(world), EN_GOODS, 2)
        place = pick("inst-en-place", str(world), EN_PLACES)
        day = pick("inst-en-day", str(world), EN_DAYS)
        tag = f"H-{world:04d}"
        context = (
            f"Shipment {tag} arrived at {place} on {day}. "
            f"The {colors[0]} bin holds {goods[0]}. "
            f"The {colors[1]} bin holds {goods[1]}. "
            f"The {colors[2]} bin is empty."
        )
        tasks = (
            ("select", f"Which bin holds {goods[0]}?", f"the {colors[0]} bin"),
            ("extract", f"What is inside the {colors[1]} bin?", goods[1]),
            ("filter", "Which bin is empty?", f"the {colors[2]} bin"),
            ("classify", f"Is the {colors[2]} bin empty?", "yes"),
            ("order", "Name the bin colors in the order they appear.", f"{colors[0]}, {colors[1]}, {colors[2]}"),
            ("compare", f"Which earlier bin holds {goods[0]}?", f"the {colors[0]} bin"),
            ("map", "Match each filled bin to its contents.", f"{colors[0]}: {goods[0]}; {colors[1]}: {goods[1]}"),
            ("rewrite", f"Rewrite only the {goods[0]} fact in one short sentence.", f"The {colors[0]} bin holds {goods[0]}."),
            ("format", "List the bin colors as a compact JSON array.", f'["{colors[0]}", "{colors[1]}", "{colors[2]}"]'),
            ("short", "On which day did the shipment arrive?", day),
        )
        instruction = "Follow the requested operation. Use only the shipment note."
    else:
        colors = pick_n("inst-ru-color", str(world), RU_COLORS, 3)
        goods = pick_n("inst-ru-good", str(world), RU_GOODS, 2)
        place = pick("inst-ru-place", str(world), RU_PLACES)
        day = pick("inst-ru-day", str(world), RU_DAYS)
        tag = f"К-{world:04d}"
        context = (
            f"Поставка {tag} пришла на {place} в {day}. "
            f"В {colors[0]} ящике лежит {goods[0]}. "
            f"В {colors[1]} ящике лежит {goods[1]}. "
            f"{colors[2].capitalize()} ящик пуст."
        )
        tasks = (
            ("select", f"В каком ящике лежит {goods[0]}?", f"{colors[0]} ящик"),
            ("extract", f"Что лежит в {colors[1]} ящике?", goods[1]),
            ("filter", "Какой ящик пуст?", f"{colors[2]} ящик"),
            ("classify", f"Пуст ли {colors[2]} ящик?", "да"),
            ("order", "Назови цвета ящиков в порядке упоминания.", f"{colors[0]}, {colors[1]}, {colors[2]}"),
            ("compare", f"Какой из ранее названных ящиков содержит {goods[0]}?", f"{colors[0]} ящик"),
            ("map", "Сопоставь заполненные ящики с содержимым.", f"{colors[0]}: {goods[0]}; {colors[1]}: {goods[1]}"),
            ("rewrite", f"Перескажи только факт про {goods[0]} одним коротким предложением.", f"В {colors[0]} ящике лежит {goods[0]}."),
            ("format", "Перечисли цвета ящиков компактным JSON-массивом.", f'["{colors[0]}", "{colors[1]}", "{colors[2]}"]'),
            ("short", "В какой день пришла поставка?", day),
        )
        instruction = "Выполни запрошенную операцию. Опирайся только на записку о поставке."
    task, question, answer = tasks[kind]
    return _native(
        source_id="native_instruction",
        row_id=f"{language}-w{world}-t{kind}",
        language=language,
        family="S1-01",
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        instruction=instruction,
        transform="native_instruction.v1",
        extra={"latent_task": task, "world": world, "contrast_group": f"inst-{language}-{world}"},
    )


def _relation_graph(index: int, language: str) -> dict:
    if language == "en":
        names = pick_n("rel-en-name", str(index), EN_NAMES, 3)
        obj = pick("rel-en-obj", str(index), EN_OBJECTS)
        site = pick("rel-en-site", str(index), EN_SITES)
        a, b, c = names
        lines = {
            "owns": f"{a} owns the {obj}.",
            "works_for": f"{a} works for {b}.",
            "located_in": f"The {obj} is located in {site}.",
            "contains": f"The {obj} contains a note from {a}.",
            "parent_of": f"{a} is the parent of {b}.",
            "manager_of": f"{a} is the manager of {b}.",
            "created": f"{a} created the {obj}.",
            "rents": f"{a} rents {site}.",
            "delivered_to": f"{a} delivered the {obj} to {b}.",
            "before": f"{a} arrived before {b}.",
        }
    else:
        names = pick_n("rel-ru-name", str(index), RU_NAMES, 3)
        obj = pick("rel-ru-obj", str(index), RU_OBJECTS)
        site = pick("rel-ru-site", str(index), RU_SITES)
        a, b, c = names
        lines = {
            "owns": f"{a} владеет предметом «{obj}».",
            "works_for": f"{a} работает на {b}.",
            "located_in": f"Предмет «{obj}» находится в месте «{site}».",
            "contains": f"В предмете «{obj}» лежит записка от {a}.",
            "parent_of": f"{a} — родитель {b}.",
            "manager_of": f"{a} руководит {b}.",
            "created": f"{a} создал предмет «{obj}».",
            "rents": f"{a} арендует «{site}».",
            "delivered_to": f"{a} доставил «{obj}» к {b}.",
            "before": f"{a} прибыл раньше {b}.",
        }
    return {"a": a, "b": b, "c": c, "obj": obj, "site": site, "lines": lines}


def relation_record(index: int, language: str, difficulty: str) -> dict:
    mode = index % 9
    graph_id = index // 9
    graph = _relation_graph(graph_id, language)
    a, b, c = graph["a"], graph["b"], graph["c"]
    obj, site = graph["obj"], graph["site"]
    rel = ("owns", "works_for", "located_in", "contains", "parent_of", "manager_of", "created", "rents", "delivered_to", "before")[graph_id % 10]
    line = graph["lines"][rel]
    distractor = f"{c} is only a witness." if language == "en" else f"{c} только свидетель."
    context = f"Graph {graph_id}. {line} {distractor}" if language == "en" else f"Граф {graph_id}. {line} {distractor}"
    if language == "en":
        instruction = "Answer from the relation note. Do not swap the roles."
        bank = {
            0: ("direct", f"Who is the subject of the stated relation?", a),
            1: ("object", f"Who or what is the object of the stated relation?", b if rel not in ("located_in", "created", "rents", "contains") else (site if rel in ("located_in", "rents") else obj)),
            2: ("reversal", f"Does {b} stand in the same role toward {a} as {a} does toward {b}?", "no"),
            3: ("negative", f"Does {c} hold the subject role?", "no"),
            4: ("subject_query", f"Name the person who acts in the note, not the witness.", a),
            5: ("distractor", f"Is {c} the subject of the relation?", "no"),
            6: ("multi", f"Who is the witness, not the subject?", c),
            7: ("two_hop", f"The subject acts, and the witness does not. Who acts?", a),
            8: ("inverse", f"If the roles of subject and object were swapped, would {a} still be the subject?", "no"),
        }
    else:
        instruction = "Ответь по записке об отношении. Роли не меняй местами."
        bank = {
            0: ("direct", "Кто является субъектом указанного отношения?", a),
            1: ("object", "Кто или что является объектом указанного отношения?", b if rel not in ("located_in", "created", "rents", "contains") else (site if rel in ("located_in", "rents") else obj)),
            2: ("reversal", f"Занимает ли {b} ту же роль по отношению к {a}, что {a} к {b}?", "нет"),
            3: ("negative", f"Является ли {c} субъектом?", "нет"),
            4: ("subject_query", "Назови того, кто действует в записке, а не свидетеля.", a),
            5: ("distractor", f"Является ли {c} субъектом отношения?", "нет"),
            6: ("multi", "Кто свидетель, а не субъект?", c),
            7: ("two_hop", "Субъект действует, свидетель нет. Кто действует?", a),
            8: ("inverse", f"Если поменять роли субъекта и объекта, останется ли {a} субъектом?", "нет"),
        }
    task, question, answer = bank[mode]
    return _native(
        source_id="native_relation",
        row_id=f"{language}-g{graph_id}-m{mode}",
        language=language,
        family="S1-02",
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        instruction=instruction,
        transform="native_relation.v1",
        extra={"latent_task": task, "relation": rel, "graph": graph_id, "subject": a, "object_entity": b},
    )


def relation_reversal_pair(index: int, language: str) -> tuple[dict, dict]:
    """Same relation type, swapped subject and object, different targets."""
    if language == "en":
        left, right = pick_n("rev-en", str(index), EN_NAMES, 2)
        obj = pick("rev-en-obj", str(index), EN_OBJECTS)
        forward_ctx = f"Pair {index}. {left} owns the {obj}. {right} does not own it."
        reverse_ctx = f"Pair {index}. {right} owns the {obj}. {left} does not own it."
        question = f"Who owns the {obj}?"
        instruction = "Name the owner stated in the note."
        yes_left, yes_right = left, right
    else:
        left, right = pick_n("rev-ru", str(index), RU_NAMES, 2)
        obj = pick("rev-ru-obj", str(index), RU_OBJECTS)
        forward_ctx = f"Пара {index}. {left} владеет предметом «{obj}». {right} им не владеет."
        reverse_ctx = f"Пара {index}. {right} владеет предметом «{obj}». {left} им не владеет."
        question = f"Кто владеет предметом «{obj}»?"
        instruction = "Назови владельца из записки."
        yes_left, yes_right = left, right
    forward = _native(
        source_id="native_relation",
        row_id=f"rev-{language}-{index}-ab",
        language=language,
        family="S1-02",
        difficulty="D",
        context=forward_ctx,
        question=question,
        answer=yes_left,
        instruction=instruction,
        transform="native_relation.reversal.v1",
        extra={"latent_task": "reversal_forward", "pair": index},
    )
    backward = _native(
        source_id="native_relation",
        row_id=f"rev-{language}-{index}-ba",
        language=language,
        family="S1-02",
        difficulty="D",
        context=reverse_ctx,
        question=question,
        answer=yes_right,
        instruction=instruction,
        transform="native_relation.reversal.v1",
        extra={"latent_task": "reversal_backward", "pair": index},
    )
    return forward, backward


def paraphrase_record(index: int, language: str, difficulty: str) -> dict:
    mode = index % 8
    prop = index // 8
    if language == "en":
        agent, other = pick_n("para-en-name", str(prop), EN_NAMES, 2)
        obj = pick("para-en-obj", str(prop), EN_OBJECTS)
        site = pick("para-en-site", str(prop), EN_SITES)
        base = f"{agent} delivered the {obj} to {site}."
        equivalent = (
            f"{agent} brought the {obj} to {site}.",
            f"The {obj} was delivered to {site} by {agent}.",
            f"{agent} took the {obj} over to {site}.",
        )[mode % 3]
        negatives = {
            3: (f"{other} delivered the {obj} to {site}.", "no"),
            4: (f"{site} delivered the {obj} to {agent}.", "no"),
            5: (f"{agent} did not deliver the {obj} to {site}.", "no"),
            6: (f"{agent} delivered the {obj} to {other}.", "no"),
            7: (f"The {obj} delivered {agent} to {site}.", "no"),
        }
        if mode < 3:
            candidate, answer = equivalent, "yes"
            task = "equivalent"
        else:
            candidate, answer = negatives[mode]
            task = "hard_negative"
        context = f"Case {prop}. {base}"
        question = f"Does this sentence state the same event: {candidate}"
        instruction = "Answer yes only if the event, roles, and polarity match."
    else:
        agent, other = pick_n("para-ru-name", str(prop), RU_NAMES, 2)
        obj = pick("para-ru-obj", str(prop), RU_OBJECTS)
        site = pick("para-ru-site", str(prop), RU_SITES)
        base = f"{agent} доставил «{obj}» на «{site}»."
        equivalent = (
            f"{agent} отвёз «{obj}» на «{site}».",
            f"«{obj}» был доставлен на «{site}» человеком {agent}.",
            f"{agent} привёз «{obj}» к месту «{site}».",
        )[mode % 3]
        negatives = {
            3: (f"{other} доставил «{obj}» на «{site}».", "нет"),
            4: (f"«{site}» доставил «{obj}» к {agent}.", "нет"),
            5: (f"{agent} не доставил «{obj}» на «{site}».", "нет"),
            6: (f"{agent} доставил «{obj}» к {other}.", "нет"),
            7: (f"«{obj}» доставил {agent} на «{site}».", "нет"),
        }
        if mode < 3:
            candidate, answer = equivalent, "да"
            task = "equivalent"
        else:
            candidate, answer = negatives[mode]
            task = "hard_negative"
        context = f"Случай {prop}. {base}"
        question = f"То же ли событие в предложении: {candidate}"
        instruction = "Ответь да только если событие, роли и полярность совпадают."
    return _native(
        source_id="native_paraphrase",
        row_id=f"{language}-p{prop}-m{mode}",
        language=language,
        family="S1-04",
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        instruction=instruction,
        transform="native_paraphrase.v1",
        extra={"latent_task": task, "proposition": prop},
    )


def controlled_record(index: int, language: str, difficulty: str) -> dict:
    kind = index % 8
    item = index // 8
    if language == "en":
        place = pick("ctrl-en-place", str(item), EN_PLACES)
        day = pick("ctrl-en-day", str(item), EN_DAYS)
        good = pick("ctrl-en-good", str(item), EN_GOODS)
        count = 2 + (item % 5)
        facts = f"sheet={item}; place={place}; day={day}; item={good}; count={count}"
        tasks = (
            ("shorter", "Rewrite the facts as one short sentence.", f"{count} {good} at {place} on {day}."),
            ("formal", "Rewrite the facts as one plain sentence.", f"The count of {good} at {place} on {day} is {count}."),
            ("fields", "Extract the place field.", place),
            ("list", "Convert the facts into a compact list.", f"place: {place}; day: {day}; item: {good}; count: {count}"),
            ("json", "Produce a compact JSON object with the four fields.", f'{{"place": "{place}", "day": "{day}", "item": "{good}", "count": {count}}}'),
            ("reorder", "List the field names in this order: count, item, day, place.", f"count, item, day, place"),
            ("combine", "Combine only the item and the count.", f"{count} {good}"),
            ("length", "Answer with the place only.", place),
        )
        instruction = "Stay inside the supplied fields. Do not add a story."
    else:
        place = pick("ctrl-ru-place", str(item), RU_PLACES)
        day = pick("ctrl-ru-day", str(item), RU_DAYS)
        good = pick("ctrl-ru-good", str(item), RU_GOODS)
        count = 2 + (item % 5)
        facts = f"лист={item}; место={place}; день={day}; предмет={good}; число={count}"
        tasks = (
            ("shorter", "Перескажи факты одним коротким предложением.", f"{count}: {good}, {place}, {day}."),
            ("formal", "Перескажи факты одним нейтральным предложением.", f"Число для «{good}» в месте «{place}» в {day} равно {count}."),
            ("fields", "Извлеки поле места.", place),
            ("list", "Преврати факты в компактный список.", f"место: {place}; день: {day}; предмет: {good}; число: {count}"),
            ("json", "Собери компактный JSON с четырьмя полями.", f'{{"место": "{place}", "день": "{day}", "предмет": "{good}", "число": {count}}}'),
            ("reorder", "Перечисли имена полей в порядке: число, предмет, день, место.", "число, предмет, день, место"),
            ("combine", "Соедини только предмет и число.", f"{count}: {good}"),
            ("length", "Ответь только названием места.", place),
        )
        instruction = "Держись данных полей. Историю не добавляй."
    task, question, answer = tasks[kind]
    return _native(
        source_id="native_controlled",
        row_id=f"{language}-c{item}-k{kind}",
        language=language,
        family="S1-09",
        difficulty=difficulty,
        context=facts,
        question=question,
        answer=answer,
        instruction=instruction,
        transform="native_controlled.v1",
        extra={"latent_task": task},
    )


_IDENTITY_EN = (
    ("name", "What is your name?", "SHINRA"),
    ("name", "State the name of this language layer.", "SHINRA"),
    ("creator", "Who created you?", "NULLXES"),
    ("creator", "Name the creator of this language layer.", "NULLXES"),
    ("distinction", "Are you NULLXES?", "No. I am SHINRA."),
    ("distinction", "Is SHINRA the same as NULLXES?", "No. SHINRA is not NULLXES."),
    ("name", "Give only your name.", "SHINRA"),
    ("creator", "Who is the creator, not the language layer?", "NULLXES"),
)
_IDENTITY_RU = (
    ("name", "Как тебя зовут?", "SHINRA"),
    ("name", "Назови имя этого языкового слоя.", "SHINRA"),
    ("creator", "Кто тебя создал?", "NULLXES"),
    ("creator", "Назови создателя этого языкового слоя.", "NULLXES"),
    ("distinction", "Ты NULLXES?", "Нет. Я SHINRA."),
    ("distinction", "SHINRA и NULLXES — это одно и то же?", "Нет. SHINRA не является NULLXES."),
    ("name", "Назови только своё имя.", "SHINRA"),
    ("creator", "Кто создатель, а не языковой слой?", "NULLXES"),
)


_EN_OPEN = (
    "Answer the identity question.",
    "Give the canonical fact.",
    "Reply with one identity fact.",
    "State only the requested fact.",
    "Keep the reply to the canonical fact.",
    "Provide the identity fact asked for.",
    "Respond with the canonical fact alone.",
    "Return the requested identity fact.",
    "Use the canonical identity answer.",
    "Produce the canonical fact.",
    "Write the canonical identity fact.",
    "Limit the reply to the canonical fact.",
)
_RU_OPEN = (
    "Ответь на вопрос об имени.",
    "Дай канонический факт.",
    "Сообщи один факт об имени.",
    "Назови только запрошенный факт.",
    "Оставь в ответе канонический факт.",
    "Приведи запрошенный факт об имени.",
    "Ответь только каноническим фактом.",
    "Верни запрошенный факт об имени.",
    "Используй канонический ответ об имени.",
    "Сформулируй канонический факт.",
    "Запиши канонический факт об имени.",
    "Ограничь ответ каноническим фактом.",
)
_EN_CLOSE = (
    "Do not add a biography.",
    "Do not describe a mission.",
    "Do not add a personality.",
    "Do not name a product category.",
    "Do not add purpose.",
    "Do not discuss consciousness.",
    "Skip any extra description.",
    "Leave out any biography.",
    "Add nothing beyond the fact.",
    "Stop after the fact.",
    "Omit any mission statement.",
    "Omit any personality note.",
)
_RU_CLOSE = (
    "Биографию не добавляй.",
    "Миссию не описывай.",
    "Характер не добавляй.",
    "Категорию продукта не называй.",
    "Назначение не добавляй.",
    "Сознание не обсуждай.",
    "Лишнее описание пропусти.",
    "Биографические детали опусти.",
    "Кроме факта ничего не добавляй.",
    "Остановись на факте.",
    "Формулировку миссии опусти.",
    "Замечание о характере опусти.",
)


def identity_record(index: int, language: str, difficulty: str) -> dict:
    bank = _IDENTITY_EN if language == "en" else _IDENTITY_RU
    opens = _EN_OPEN if language == "en" else _RU_OPEN
    closes = _EN_CLOSE if language == "en" else _RU_CLOSE
    fact, question, answer = bank[index % len(bank)]
    opener = opens[(index // len(bank)) % len(opens)]
    closer = closes[(index // (len(bank) * len(opens))) % len(closes)]
    context = opener
    question = f"{question} {closer}"
    instruction = "Use only the canonical name, creator, or distinction." if language == "en" else "Используй только каноническое имя, создателя или различие."
    return _native(
        source_id="native_identity",
        row_id=f"{language}-id-{index}",
        language=language,
        family="S1-10",
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        instruction=instruction,
        transform="native_identity.v1",
        extra={"latent_fact": fact},
    )
