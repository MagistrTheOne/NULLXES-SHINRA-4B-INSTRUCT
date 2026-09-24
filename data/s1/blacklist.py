"""Hashes of frozen diagnostic prompts. The prompts are not training rows."""

from __future__ import annotations

import json
from pathlib import Path

from data.s1.foundation import sha256_text

S07 = Path(
    r"C:\Users\maxon\.cache\huggingface\hub\models--MagistrTheOne--NULLXES-SHINRA-4B-INSTRUCT"
    r"\snapshots\efae04115951d9473ebc2a90a2b2c684115408e7"
    r"\exp\stage0-2026-09-24-final\eval\s07_fresh_blind_dataset.json"
)
S05 = Path(
    r"C:\Users\maxon\.cache\huggingface\hub\models--MagistrTheOne--NULLXES-SHINRA-4B-INSTRUCT"
    r"\snapshots\efae04115951d9473ebc2a90a2b2c684115408e7"
    r"\exp\stage0-2026-09-23\research\s05-blind80\s05_blind80_dataset.json"
)

QA_PROMPTS = (
    "Context:\nThe ferry from Greyhaven to Lumen Dock leaves at dawn. Mira Holt bought ticket 418 at the harbor office. The crossing takes forty minutes.\n\nQuestion:\nWhat is the number of Mira Holt's ticket?\n\nAnswer briefly using only the context.",
    "Context:\nNorthwind Glass employs two chemists. Adele Voss works the night kiln and wears a green apron. Tomas Reed works the day kiln and wears a blue apron.\n\nQuestion:\nWhat color apron does Adele Voss wear?\n\nAnswer briefly using only the context.",
    "Context:\nThe town of Brindle owns the clock tower. The clock tower stands beside the river market. Petra Lang rents a stall in the river market, but she does not own the tower.\n\nQuestion:\nWho owns the clock tower?\n\nAnswer briefly using only the context.",
    "Context:\nThree crates arrived at Orchard Station. The red crate holds pears. The yellow crate holds apples. The blue crate is empty and holds no fruit.\n\nQuestion:\nWhat fruit is in the blue crate?\n\nAnswer briefly using only the context.",
    "Context:\nElena Park left her umbrella in the reading room. She returned after lunch, but the umbrella was gone. The clerk said a visitor had taken it by mistake.\n\nQuestion:\nWhat did Elena Park leave in the reading room?\n\nAnswer briefly using only the context.",
    "Context:\nThe red tram stops only at Harbor and Mill. Lale boarded the red tram at Harbor. She got off at the tram's other stop.\n\nQuestion:\nWhere did Lale get off?\n\nAnswer briefly using only the context.",
    "Context:\nВ порту Кедр пришёл катер «Иволга». Капитан Олег Савин записал в журнал рейс 27. Катер стоял у третьего причала.\n\nQuestion:\nКакой номер рейса записал Олег Савин?\n\nAnswer briefly using only the context.",
    "Context:\nВ мастерской «Лён» работают две ткачихи. Нина Белова ткёт синий лён. Раиса Ким ткёт серый лён и держит склад.\n\nQuestion:\nКакой лён ткёт Нина Белова?\n\nAnswer briefly using only the context.",
    "Context:\nДом на улице Тихой принадлежит семье Морозовых. Флигель за домом снимает архив города. Архивом заведует Пётр Ильин, но дом ему не принадлежит.\n\nQuestion:\nКому принадлежит дом на улице Тихой?\n\nAnswer briefly using only the context.",
    "Context:\nНа полке три банки. В первой банке мёд. Во второй банке варенье. Третья банка пустая, в ней нет сладости.\n\nQuestion:\nЕсть ли сладость в третьей банке?\n\nAnswer briefly using only the context.",
    "Context:\nИгорь Левин оставил пальто в гардеробе театра. Он вышел в антракте. Пальто всё ещё висело на крючке номер 12.\n\nQuestion:\nЧто оставил Игорь Левин в гардеробе?\n\nAnswer briefly using only the context.",
    "Context:\nАвтобус маршрута 4 ходит между вокзалом и больницей. Кирилл сел на этот автобус у вокзала. Он вышел на другой конечной остановке маршрута.\n\nQuestion:\nГде вышел Кирилл?\n\nAnswer briefly using only the context.",
)


def _prompts(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [str(row["prompt"]) for row in rows if row.get("prompt")]


def diagnostic_blacklist() -> set[str]:
    prompts = list(QA_PROMPTS) + _prompts(S05) + _prompts(S07)
    return {sha256_text(prompt) for prompt in prompts}
