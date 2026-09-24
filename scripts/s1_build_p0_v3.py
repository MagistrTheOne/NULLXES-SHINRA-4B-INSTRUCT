"""S1-P0 corpus build V3. CPU tokenization only. No model weights."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
print("import-root", _ROOT, (_ROOT / "data" / "s1" / "blacklist.py").is_file(), flush=True)
sys.path.insert(0, str(_ROOT))

import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict

from huggingface_hub import hf_hub_download
from jinja2 import Environment
from tokenizers import Tokenizer

from data.s1.blacklist import QA_PROMPTS, diagnostic_blacklist, frozen_prompt_hashes
from data.s1.foundation import make_record, sha256_text
from data.s1.loaders import (
    iter_danetqa,
    iter_entailment,
    iter_muserc,
    iter_rucos,
    iter_rwsd,
    iter_tydi_primary,
    iter_tydi_secondary,
)
from data.s1.native import (
    controlled_record,
    identity_record,
    instruction_record,
    paraphrase_record,
    relation_record,
)
from data.s1.p0_amendment import P0_BUDGET, effective_family_targets
from data.s1.transforms import paws_paraphrase
from training.s1_p0_objective import can_attend, gradient_divisor, token_weighted_mean

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("S1_P0_SCRATCH", "/tmp/s1p0"))
HUB_REPO = "MagistrTheOne/NULLXES-SHINRA-S1-P0"
RUN_ID = os.environ.get("S1_P0_RUN", "v1")
SHARD_ROWS = 12_000
REVISION = "efae04115951d9473ebc2a90a2b2c684115408e7"
SEED = 20260924
BUILDER = "s1-p0-v3"
UPDATE_TARGET = 32_768
PUBLIC_AIM = 1_550_000
NATIVE_AIM = 550_000
DIFF_WEIGHT = {"A": 15, "B": 20, "C": 30, "D": 25, "E": 10}
LANG_HALF = P0_BUDGET // 2


class Account:
    def __init__(self) -> None:
        path = hf_hub_download(
            "MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT",
            "tokenizer.json",
            revision=REVISION,
            local_files_only=os.environ.get("S1_P0_HUB") != "1",
        )
        self.tokenizer = Tokenizer.from_file(path)
        template = (ROOT / "tokenizer" / "chat_template.jinja").read_text(encoding="utf-8")
        self.template = Environment(autoescape=False).from_string(template)
        bos = self.tokenizer.token_to_id("<|bos|>")
        eot = self.tokenizer.token_to_id("<|eot|>")
        pad = self.tokenizer.token_to_id("<|pad|>")
        if (bos, eot, pad) != (1, 2, 3):
            raise RuntimeError(f"tokenizer specials {(bos, eot, pad)}")

    def count(self, user: str, answer: str) -> tuple[int, int, int, int]:
        prompt = self.template.render(
            messages=[{"role": "user", "content": user}],
            add_generation_prompt=True,
            bos_token="<|bos|>",
        )
        full = self.template.render(
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
            add_generation_prompt=False,
            bos_token="<|bos|>",
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        full_ids = self.tokenizer.encode(full, add_special_tokens=False).ids
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("prefix")
        if full_ids.count(1) != 1 or 2 not in full_ids[len(prompt_ids) :]:
            raise ValueError("specials")
        if len(full_ids) > 8192:
            raise ValueError("overlength")
        target = len(full_ids) - len(prompt_ids)
        if target <= 0:
            raise ValueError("empty-target")
        return len(full_ids), len(prompt_ids), target, len(full_ids)


def load_blacklist() -> set[str]:
    frozen = frozen_prompt_hashes()
    if len(frozen) != 160:
        raise RuntimeError("frozen diagnostic hashes are incomplete")
    qa = {sha256_text(prompt) for prompt in QA_PROMPTS}
    if len(qa) != 12:
        raise RuntimeError("QA12 blacklist is incomplete")
    payload = json.loads((ROOT / "data" / "s1" / "p0_dev" / "dev_prompt_hashes.json").read_text(encoding="utf-8"))
    dev = set(payload["prompt_sha256"])
    if payload.get("count") != 2000 or len(dev) != 2000:
        raise RuntimeError("S1 DEV blacklist is incomplete")
    return diagnostic_blacklist() | dev


class Build:
    def __init__(self) -> None:
        self.account = Account()
        self.blacklist = load_blacklist()
        self.seen_prompt: set[str] = set()
        self.seen_pair: set[str] = set()
        self.rows: list[dict] = []
        self.drops: Counter = Counter()
        self.sources: dict[str, Counter] = defaultdict(Counter)
        self.family_tokens: Counter = Counter()
        self.lang_tokens: Counter = Counter()
        self.class_tokens: Counter = Counter()
        self.diff_tokens: Counter = Counter()
        self.identity_tokens = 0
        identity_probe = self._identity_tokens()
        self.targets = effective_family_targets(identity_probe)

    def _identity_tokens(self) -> int:
        total = 0
        seen: set[str] = set()
        for language in ("en", "ru"):
            for index in range(1152):
                record = identity_record(index, language, "D")
                user = record["messages"][0]["content"]
                digest = sha256_text(user)
                if digest in seen:
                    raise RuntimeError("identity bank duplicated")
                seen.add(digest)
                _, _, target, _ = self.account.count(user, record["messages"][1]["content"])
                total += target
        if len(seen) != 2304:
            raise RuntimeError(f"identity bank {len(seen)}")
        return total

    def want_language(self) -> str:
        return "ru" if self.lang_tokens["ru"] <= self.lang_tokens["en"] else "en"

    def want_difficulty(self, forced: str | None = None) -> str:
        if forced:
            return forced
        need = {level: P0_BUDGET * weight // 100 - self.diff_tokens[level] for level, weight in DIFF_WEIGHT.items()}
        return max(need, key=need.get)

    def room(self, family: str) -> int:
        return self.targets[family] - self.family_tokens[family]

    def consider(self, record: dict, source_class: str, source_name: str) -> bool:
        stats = self.sources[source_name]
        stats["rendered"] += 1
        user = record["messages"][0]["content"]
        answer = record["messages"][1]["content"]
        family = record["family"]
        language = record["language"]
        if family not in self.targets:
            return self._drop(stats, "invalid-family")
        if language not in ("en", "ru"):
            return self._drop(stats, "invalid-language")
        if not user.strip() or not answer.strip():
            return self._drop(stats, "empty")
        if record.get("source_split") not in (None, "train"):
            return self._drop(stats, "forbidden-split")
        if any(item.get("role") == "system" for item in record["messages"]):
            return self._drop(stats, "system")
        if family != "S1-10" and ("SHINRA" in user or "NULLXES" in user or "SHINRA" in answer or "NULLXES" in answer):
            return self._drop(stats, "identity-leak")
        prompt_hash = sha256_text(user)
        pair_hash = sha256_text(user + "\n" + answer)
        if prompt_hash in self.blacklist:
            return self._drop(stats, "blacklist")
        if prompt_hash in self.seen_prompt or pair_hash in self.seen_pair:
            return self._drop(stats, "duplicate")
        if self.room(family) <= 0:
            return self._drop(stats, "family-full")
        if source_class == "PUBLIC-DERIVED" and self.class_tokens["PUBLIC-DERIVED"] >= PUBLIC_AIM and self.room(family) > 0:
            return self._drop(stats, "public-aim")
        if source_class == "NATIVE" and family != "S1-10" and self.class_tokens["NATIVE"] >= NATIVE_AIM:
            return self._drop(stats, "native-aim")
        try:
            sequence, prompt, target, _ = self.account.count(user, answer)
        except ValueError as exc:
            return self._drop(stats, str(exc))
        if language == "en" and self.lang_tokens["en"] + target > int(P0_BUDGET * 0.53) and self.lang_tokens["ru"] < LANG_HALF:
            return self._drop(stats, "language-cap")
        if language == "ru" and self.lang_tokens["ru"] + target > int(P0_BUDGET * 0.53) and self.lang_tokens["en"] < LANG_HALF:
            return self._drop(stats, "language-cap")
        row = {
            "id": record["id"],
            "language": language,
            "family": family,
            "difficulty": record["difficulty"],
            "source": record["source"],
            "source_class": source_class,
            "source_split": record.get("source_split", "train"),
            "provenance_id": str(record["metadata"]["source_row_id"]),
            "messages": record["messages"],
            "input_tokens": sequence,
            "prompt_tokens": prompt,
            "target_tokens": target,
            "sequence_tokens": sequence,
            "metadata": {
                "source_config": record.get("source_config"),
                "license": record["metadata"].get("license"),
                "transform": record["metadata"].get("transform"),
                "builder": BUILDER,
                "seed": SEED,
            },
        }
        self.rows.append(row)
        self.seen_prompt.add(prompt_hash)
        self.seen_pair.add(pair_hash)
        self.family_tokens[family] += target
        self.lang_tokens[language] += target
        self.class_tokens[source_class] += target
        self.diff_tokens[record["difficulty"]] += target
        stats["accepted"] += 1
        stats["target_tokens"] += target
        if family == "S1-10":
            self.identity_tokens += target
        self._maybe_flush(force=False)
        return True

    def _maybe_flush(self, force: bool) -> None:
        if os.environ.get("S1_P0_HUB") != "1":
            return
        if not force and len(self.rows) < SHARD_ROWS:
            return
        if not self.rows and not force:
            return
        sink = getattr(self, "sink", None)
        if sink is None:
            self.sink = HubSink(self)
            sink = self.sink
        sink.flush(force=force)

    def _drop(self, stats: Counter, reason: str) -> bool:
        stats["rejected"] += 1
        stats[f"reason:{reason}"] += 1
        self.drops[reason] += 1
        return False

    def take_public(self, iterable, source_name: str, limit_seen: int, families: tuple[str, ...] = ()) -> None:
        for record in iterable:
            stats = self.sources[source_name]
            stats["seen"] += 1
            if stats["seen"] > limit_seen:
                break
            self.consider(record, "PUBLIC-DERIVED", source_name)
            if self.class_tokens["PUBLIC-DERIVED"] >= PUBLIC_AIM:
                break
            if families and all(self.room(family) <= 0 for family in families):
                break

    def generate(self, builder, source_class: str, source_name: str, family: str, languages: tuple[str, ...], forced: str | None = None) -> None:
        index = 0
        stalls = 0
        while self.room(family) > 0 and stalls < 5000:
            language = self.want_language() if len(languages) > 1 else languages[0]
            if language not in languages:
                language = languages[0]
            record = builder(index, language, self.want_difficulty(forced))
            index += 1
            before = len(self.rows)
            self.sources[source_name]["seen"] += 1
            self.consider(record, source_class, source_name)
            stalls = stalls + 1 if len(self.rows) == before else 0
            if source_class == "NATIVE" and family != "S1-10" and self.class_tokens["NATIVE"] >= NATIVE_AIM and family != "S1-01":
                break


def polarity_record(index: int, language: str, difficulty: str) -> dict:
    slot = index
    if language == "en":
        context = f"Note P-{slot}. The blue crate is not empty. The red crate has no pears."
        if index % 2 == 0:
            question, answer = "Is the blue crate empty?", "no"
        else:
            question, answer = "Does the red crate have pears?", "no"
        instruction = "Answer the polarity question from the note."
    else:
        context = f"Записка П-{slot}. Синий ящик не пуст. В красном ящике нет груш."
        if index % 2 == 0:
            question, answer = "Синий ящик пуст?", "нет"
        else:
            question, answer = "В красном ящике есть груши?", "нет"
        instruction = "Ответь на вопрос о полярности по записке."
    return make_record(
        source_id="native_instruction",
        source_config="native",
        source_split="train",
        source_row_id=f"polarity-{language}-{slot}",
        language=language,
        family="S1-03",
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        license_name="nullxes-internal",
        transform="native_polarity.v1",
        instruction=instruction,
    )


def extract_record(index: int, language: str, difficulty: str, family: str) -> dict:
    slot = index
    if language == "en":
        context = f"Card {family} E-{slot}. Depot {slot} stores linen in bay 2 and chalk in bay 4."
        question, answer = "What does bay 4 store?", "chalk"
        instruction = "Extract only the asked fact."
    else:
        context = f"Карточка {family} Э-{slot}. Склад {slot} хранит лён в отсеке 2 и мел в отсеке 4."
        question, answer = "Что лежит в отсеке 4?", "мел"
        instruction = "Извлеки только запрошенный факт."
    return make_record(
        source_id="native_controlled",
        source_config="native",
        source_split="train",
        source_row_id=f"extract-{family}-{language}-{slot}",
        language=language,
        family=family,
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        license_name="nullxes-internal",
        transform="synthetic_extract.v1",
        instruction=instruction,
    )


def render_snli(row: dict, index: int) -> dict | None:
    label = row.get("label")
    names = {0: "entailment", 1: "neutral", 2: "contradiction"}
    if label not in names:
        return None
    caption = str(row.get("captionID") or row.get("pairID") or "")
    if caption.startswith("vg_"):
        return None
    premise = str(row.get("premise") or "").strip()
    hypothesis = str(row.get("hypothesis") or "").strip()
    if not premise or not hypothesis:
        return None
    difficulty = {"entailment": "C", "neutral": "B", "contradiction": "D"}[names[label]]
    return make_record(
        source_id="snli",
        source_config="plain_text",
        source_split="train",
        source_row_id=str(row.get("pairID") or index),
        language="en",
        family="S1-06",
        difficulty=difficulty,
        context=f"Premise: {premise}\nHypothesis: {hypothesis}",
        question="Does the premise entail the hypothesis, contradict it, or leave it neutral?",
        answer=names[label],
        license_name="cc-by-sa-4.0",
        transform="snli_entailment.v1",
    )


def render_wiki(row: dict, index: int) -> dict | None:
    complex_sentence = str(row.get("complex_sentence") or "").strip()
    left = str(row.get("simple_sentence_1") or "").strip()
    right = str(row.get("simple_sentence_2") or "").strip()
    if not complex_sentence or not left or not right:
        return None
    return make_record(
        source_id="wikisplit",
        source_config="default",
        source_split="train",
        source_row_id=str(row.get("id") or index),
        language="en",
        family="S1-09",
        difficulty="C",
        context=complex_sentence,
        question="Split this into two sentences that keep the same meaning.",
        answer=f"{left} {right}",
        license_name="cc-by-sa-4.0",
        transform="wiki_split.v1",
    )


def render_fewrel(row: dict, index: int) -> dict | None:
    tokens = row.get("tokens") or row.get("token") or []
    if isinstance(tokens, list):
        text = " ".join(str(tok) for tok in tokens).strip()
    else:
        text = str(row.get("text") or row.get("sentence") or "").strip()
    relation = row.get("relation") or row.get("name") or row.get("relation_text")
    names = row.get("names")
    if not relation and isinstance(names, list) and names:
        relation = names[0]
    head = row.get("h") or row.get("head") or {}
    tail = row.get("t") or row.get("tail") or {}
    head_name = head.get("name") if isinstance(head, dict) else head
    tail_name = tail.get("name") if isinstance(tail, dict) else tail
    if not text or not relation or not head_name or not tail_name:
        return None
    return make_record(
        source_id="fewrel_wiki",
        source_config="train_wiki",
        source_split="train",
        source_row_id=str(row.get("id") or index),
        language="en",
        family="S1-02",
        difficulty="D",
        context=f"Sentence: {text}\nHead: {head_name}\nTail: {tail_name}",
        question="What relation holds from the head to the tail?",
        answer=str(relation).replace("_", " "),
        license_name="cc-by-sa-4.0",
        transform="fewrel_relation.v1",
    )


def render_trex(row: dict, index: int) -> dict | None:
    if isinstance(row, list):
        row = {"triples": row}
    text = str(row.get("text") or row.get("sentence") or row.get("input") or row.get("abstract") or "").strip()
    relation = row.get("predicate") or row.get("relation") or row.get("predicate_id")
    triples = row.get("triples") or row.get("relations") or []
    if not relation and isinstance(triples, list) and triples:
        first = triples[0]
        if isinstance(first, dict):
            relation = first.get("predicate") or first.get("relation") or first.get("predicate_id")
            text = text or str(first.get("sentence") or first.get("text") or "")
    if isinstance(relation, dict):
        relation = relation.get("name") or relation.get("id")
    if not text or not relation:
        return None
    title = str(row.get("title") or row.get("docid") or index)
    return make_record(
        source_id="trex",
        source_config="wikipedia_wikidata",
        source_split="train",
        source_row_id=f"{title}:{index}",
        language="en",
        family="S1-02",
        difficulty="C",
        context=text,
        question="Which relation does this sentence state?",
        answer=str(relation).replace("_", " "),
        license_name="cc-by-sa-4.0",
        transform="trex_relation.v1",
    )


def render_tapaco(left: str, right: str, language: str, set_id: str) -> dict:
    question = "Do these two sentences have the same meaning?" if language == "en" else "Эти два предложения значат одно и то же?"
    answer = "yes" if language == "en" else "да"
    return make_record(
        source_id="tapaco",
        source_config="en_ru",
        source_split="train",
        source_row_id=f"{language}:{set_id}",
        language=language,
        family="S1-04",
        difficulty="C",
        context=f"Sentence 1: {left}\nSentence 2: {right}",
        question=question,
        answer=answer,
        license_name="cc-by-2.0",
        transform="tapaco_paraphrase.v1",
    )


def fetch_bytes(url: str) -> bytes:
    import time
    import urllib.request

    cache = OUT / "cache" / hashlib.sha256(url.encode()).hexdigest()
    if cache.is_file() and cache.stat().st_size > 1000:
        return cache.read_bytes()
    cache.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(6):
        try:
            blob = urllib.request.urlopen(url, timeout=180).read()
            cache.write_bytes(blob)
            return blob
        except Exception as exc:
            last = exc
            time.sleep(min(30, 2 ** attempt))
    raise last if last else RuntimeError(url)


def stream_rows(repo: str, config: str | None, split: str):
    import time

    from datasets import load_dataset

    last: Exception | None = None
    for attempt in range(8):
        try:
            return load_dataset(repo, config, split=split, streaming=True)
        except Exception as exc:
            last = exc
            time.sleep(min(60, 2 ** attempt))
    raise last if last else RuntimeError(repo)


def build() -> Build:
    job = Build()
    if os.environ.get("S1_P0_HUB") == "1":
        job.sink = HubSink(job)
        job.sink.restore()
    print("targets", job.targets, flush=True)
    for language in ("en", "ru"):
        for index in range(1152):
            record = identity_record(index, language, "D")
            job.sources["native_identity"]["seen"] += 1
            job.consider(record, "NATIVE", "native_identity")
    print("identity", job.identity_tokens, job.family_tokens["S1-10"], flush=True)

    job.take_public(iter_rucos(200_000), "rsg_rucos", 200_000)
    job.take_public(iter_rwsd(10_000), "rsg_rwsd", 10_000)
    job.take_public(iter_muserc(20_000), "rsg_muserc", 20_000)
    job.take_public(iter_danetqa(10_000), "rsg_danetqa", 10_000)
    job.take_public(iter_entailment("rsg_terra", "TERRa", 10_000), "rsg_terra", 10_000)
    job.take_public(iter_entailment("rsg_rcb", "RCB", 10_000), "rsg_rcb", 10_000)
    print("rsg", dict(job.family_tokens), flush=True)

    def paws_iter():
        import pyarrow.parquet as pq

        path = hf_hub_download(
            "google-research-datasets/paws",
            "labeled_final/train-00000-of-00001.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(path, columns=["sentence1", "sentence2", "label"])
        columns = table.to_pydict()
        for index, label in enumerate(columns["label"]):
            yield paws_paraphrase(str(index), columns["sentence1"][index], columns["sentence2"][index], label)

    try:
        job.take_public(paws_iter(), "paws_wiki", 60_000, ("S1-04",))
    except Exception as exc:
        print("paws-skipped", type(exc).__name__, exc, flush=True)
    print("after-paws", job.family_tokens["S1-04"], job.class_tokens["PUBLIC-DERIVED"], flush=True)

    def fewrel_iter():
        names = json.loads(fetch_bytes("https://raw.githubusercontent.com/thunlp/FewRel/master/data/pid2name.json").decode("utf-8"))
        payload = json.loads(fetch_bytes("https://raw.githubusercontent.com/thunlp/FewRel/master/data/train_wiki.json").decode("utf-8"))
        index = 0
        for relation_id, instances in payload.items():
            label = names.get(relation_id, [relation_id])
            relation = label[0] if isinstance(label, list) else str(label)
            for instance in instances:
                instance = dict(instance)
                instance["relation"] = relation
                instance["id"] = f"{relation_id}:{index}"
                rendered = render_fewrel(instance, index)
                index += 1
                if rendered:
                    yield rendered

    try:
        job.take_public(fewrel_iter(), "fewrel_wiki", 50_000, ("S1-02",))
    except Exception as exc:
        print("fewrel-skipped", type(exc).__name__, exc, flush=True)

    def snli_iter():
        import io
        import json as jsonlib
        import zipfile

        labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
        blob = fetch_bytes("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            name = next(item for item in archive.namelist() if item.endswith("snli_1.0_train.jsonl"))
            with archive.open(name) as handle:
                for index, raw in enumerate(io.TextIOWrapper(handle, encoding="utf-8")):
                    row = jsonlib.loads(raw)
                    caption = str(row.get("captionID") or row.get("pairID") or "")
                    if caption.startswith("vg_"):
                        continue
                    gold = row.get("gold_label")
                    if gold not in labels:
                        continue
                    rendered = render_snli(
                        {
                            "premise": row.get("sentence1"),
                            "hypothesis": row.get("sentence2"),
                            "label": labels[gold],
                            "pairID": row.get("pairID") or caption or index,
                            "captionID": caption,
                        },
                        index,
                    )
                    if rendered:
                        yield rendered

    try:
        job.take_public(snli_iter(), "snli", 400_000, ("S1-06",))
    except Exception as exc:
        print("snli-skipped", type(exc).__name__, exc, flush=True)

    def wiki_iter():
        import io
        import zipfile

        blob = fetch_bytes("https://github.com/google-research-datasets/wiki-split/raw/master/train.tsv.zip")
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            name = next(item for item in archive.namelist() if item.endswith("train.tsv"))
            with archive.open(name) as handle:
                for index, line in enumerate(io.TextIOWrapper(handle, encoding="utf-8")):
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    simple = [piece.strip() for piece in parts[1].split("<::::>") if piece.strip()]
                    if len(simple) < 2:
                        simple = [piece.strip() for piece in parts[1].split("<####>") if piece.strip()]
                    if len(simple) < 2:
                        continue
                    rendered = render_wiki(
                        {
                            "complex_sentence": parts[0],
                            "simple_sentence_1": simple[0],
                            "simple_sentence_2": simple[1],
                            "id": index,
                        },
                        index,
                    )
                    if rendered:
                        yield rendered

    try:
        job.take_public(wiki_iter(), "wikisplit", 40_000, ("S1-09",))
    except Exception as exc:
        print("wikisplit-skipped", type(exc).__name__, exc, flush=True)
    print("public-mid", job.class_tokens["PUBLIC-DERIVED"], dict(job.family_tokens), flush=True)

    def tapaco_iter():
        import pyarrow.parquet as pq

        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        emitted: set[tuple[str, str]] = set()
        for language in ("en", "ru"):
            path = hf_hub_download(
                "community-datasets/tapaco",
                f"{language}/train-00000-of-00001.parquet",
                repo_type="dataset",
            )
            columns = pq.read_table(path).to_pydict()
            sentences = columns.get("paraphrase") or columns.get("sentence") or []
            set_ids = columns.get("paraphrase_set_id") or columns.get("set_id") or []
            for sentence, set_id in zip(sentences, set_ids):
                sentence = str(sentence or "").strip()
                if not sentence:
                    continue
                key = (language, str(set_id))
                grouped[key].append(sentence)
                if key not in emitted and len(grouped[key]) >= 2:
                    emitted.add(key)
                    left, right = sorted(grouped[key][:2])
                    yield render_tapaco(left, right, language, str(set_id))
                    if len(emitted) > 80_000:
                        return

    try:
        job.take_public(tapaco_iter(), "tapaco", 80_000, ("S1-04",))
    except Exception as exc:
        print("tapaco-skipped", type(exc).__name__, exc, flush=True)

    def tydi_iter():
        yield from iter_tydi_primary(20_000, 20_000, scan_limit=200_000)
        yield from iter_tydi_secondary(20_000, 20_000, scan_limit=80_000)

    try:
        job.take_public(tydi_iter(), "tydiqa", 80_000)
    except Exception as exc:
        print("tydi-skipped", type(exc).__name__, exc, flush=True)

    if job.room("S1-02") > 0 and job.class_tokens["PUBLIC-DERIVED"] < PUBLIC_AIM:
        seen_sentence: set[str] = set()

        def trex_iter():
            path = hf_hub_download(
                "relbert/t_rex",
                "data/t_rex.filter_unified.min_entity_5.train.jsonl",
                repo_type="dataset",
            )
            with open(path, encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    rendered = render_trex(row, index)
                    if not rendered:
                        continue
                    key = sha256_text(rendered["messages"][0]["content"])
                    if key in seen_sentence:
                        continue
                    seen_sentence.add(key)
                    yield rendered

        try:
            job.take_public(trex_iter(), "trex", 200_000, ("S1-02",))
        except Exception as exc:
            print("trex-skipped", type(exc).__name__, exc, flush=True)

    print("public-done", job.class_tokens, dict(job.family_tokens), dict(job.lang_tokens), flush=True)
    job.generate(instruction_record, "NATIVE", "native_instruction", "S1-01", ("en", "ru"))
    while job.class_tokens["NATIVE"] < NATIVE_AIM and job.room("S1-03") > 0:
        job.generate(polarity_record, "NATIVE", "native_polarity", "S1-03", ("en", "ru"), forced="D")
        break
    for family in ("S1-02", "S1-03", "S1-04", "S1-05", "S1-06", "S1-07", "S1-08", "S1-09"):
        if job.room(family) <= 0:
            continue
        if family == "S1-04":
            job.generate(paraphrase_record, "SYNTHETIC", "native_paraphrase", family, ("en", "ru"))
        elif family == "S1-02":
            job.generate(relation_record, "SYNTHETIC", "native_relation", family, ("en", "ru"))
        elif family == "S1-09":
            job.generate(controlled_record, "SYNTHETIC", "native_controlled", family, ("en", "ru"))
        else:
            job.generate(lambda index, language, difficulty, fam=family: extract_record(index, language, difficulty, fam), "SYNTHETIC", f"synthetic_{family}", family, ("en", "ru"))
    print("filled", job.class_tokens, dict(job.family_tokens), dict(job.lang_tokens), "rows", len(job.rows), flush=True)
    return job


def canonicalize(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{SEED}|{row['id']}".encode()).hexdigest())


def write_outputs(rows: list[dict], targets: dict[str, int], sources: dict, drops: Counter) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ordered = canonicalize(rows)
    shard_size = 25_000
    shard_count = max(1, (len(ordered) + shard_size - 1) // shard_size)
    shard_dir = OUT / "shards"
    shard_dir.mkdir(exist_ok=True)
    shard_meta = []
    train_hash = hashlib.sha256()
    for index in range(shard_count):
        chunk = ordered[index * shard_size : (index + 1) * shard_size]
        name = f"train-{index:05d}-of-{shard_count:05d}.jsonl"
        path = shard_dir / name
        blob = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chunk)
        path.write_text(blob, encoding="utf-8")
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        train_hash.update(blob.encode("utf-8"))
        shard_meta.append({
            "filename": name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "records": len(chunk),
            "input_tokens": sum(row["input_tokens"] for row in chunk),
            "target_tokens": sum(row["target_tokens"] for row in chunk),
        })
    updates = []
    cursor = 0
    update_index = 0
    while cursor < len(ordered):
        start = cursor
        target = 0
        inputs = 0
        while cursor < len(ordered) and (target < UPDATE_TARGET or cursor == start):
            target += ordered[cursor]["target_tokens"]
            inputs += ordered[cursor]["input_tokens"]
            cursor += 1
            if target >= UPDATE_TARGET:
                break
        chunk = ordered[start:cursor]
        prompt = sum(row["prompt_tokens"] for row in chunk)
        sequence = sum(row["sequence_tokens"] for row in chunk)
        updates.append({
            "index": update_index,
            "first_record": chunk[0]["id"],
            "last_record": chunk[-1]["id"],
            "records": len(chunk),
            "input_tokens": inputs,
            "target_tokens": target,
            "overshoot": max(0, target - UPDATE_TARGET) if cursor < len(ordered) or target > UPDATE_TARGET else max(0, target - UPDATE_TARGET),
            "supervised_fraction": target / sequence if sequence else 0,
        })
        update_index += 1
    groups = [[float(row["target_tokens"])] for row in ordered[:3]]
    if len({len(group) for group in groups}) == 1 and len(groups) > 1:
        pass
    objective = {
        "single": gradient_divisor(ordered[0]["target_tokens"]) == float(ordered[0]["target_tokens"]),
        "variable": token_weighted_mean([[1.0], [0.0, 0.0]]) == 1 / 3,
        "isolated": can_attend([0, 0, 1], 2, 1) is False and can_attend([0, 0, 1], 1, 0) is True,
        "denominator": gradient_divisor(updates[0]["target_tokens"]) == float(updates[0]["target_tokens"]),
    }
    report = {
        "builder": BUILDER,
        "seed": SEED,
        "records": len(ordered),
        "input_tokens": sum(row["input_tokens"] for row in ordered),
        "prompt_tokens": sum(row["prompt_tokens"] for row in ordered),
        "target_tokens": sum(row["target_tokens"] for row in ordered),
        "budget": P0_BUDGET,
        "targets": targets,
        "family_tokens": dict(Counter(row["family"] for row in ordered)),
        "family_target_tokens": {family: sum(row["target_tokens"] for row in ordered if row["family"] == family) for family in targets},
        "language_tokens": {
            "en": sum(row["target_tokens"] for row in ordered if row["language"] == "en"),
            "ru": sum(row["target_tokens"] for row in ordered if row["language"] == "ru"),
        },
        "class_tokens": {
            key: sum(row["target_tokens"] for row in ordered if row["source_class"] == key)
            for key in ("PUBLIC-DERIVED", "NATIVE", "SYNTHETIC")
        },
        "difficulty_tokens": {
            level: sum(row["target_tokens"] for row in ordered if row["difficulty"] == level)
            for level in DIFF_WEIGHT
        },
        "sources": {name: dict(counter) for name, counter in sources.items()},
        "drops": dict(drops),
        "shards": shard_meta,
        "train_sha256": train_hash.hexdigest(),
        "updates": {
            "count": len(updates),
            "mean_target": statistics.mean(item["target_tokens"] for item in updates),
            "max_overshoot": max(item["overshoot"] for item in updates),
            "last_target": updates[-1]["target_tokens"],
        },
        "objective": objective,
        "target_lengths": [row["target_tokens"] for row in ordered],
        "sequence_lengths": [row["sequence_tokens"] for row in ordered],
    }
    (OUT / "manifest.json").write_text(json.dumps({k: v for k, v in report.items() if k not in ("target_lengths", "sequence_lengths")}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "update_plan.json").write_text(json.dumps(updates, ensure_ascii=False), encoding="utf-8")
    (OUT / "quality_report.json").write_text(json.dumps({k: v for k, v in report.items() if k not in ("target_lengths", "sequence_lengths", "updates")}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


class HubSink:
    """Upload one shard at a time. Scratch files live only under /tmp."""

    def __init__(self, job: "Build") -> None:
        from huggingface_hub import HfApi

        self.job = job
        self.api = HfApi()
        self.repo = HUB_REPO
        self.scratch = OUT
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.shard_index = 0
        self.shards: list[dict] = []
        self.restored = False

    def restore(self) -> None:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(self.repo, f"staging/{RUN_ID}/state/build_state.json", repo_type="dataset")
            state = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return
        self.shard_index = int(state.get("next_shard", 0))
        self.shards = list(state.get("shards", []))
        self.job.family_tokens.update(state.get("family_tokens", {}))
        self.job.lang_tokens.update(state.get("language_tokens", {}))
        self.job.class_tokens.update(state.get("class_tokens", {}))
        self.job.diff_tokens.update(state.get("difficulty_tokens", {}))
        self.job.identity_tokens = int(state.get("identity_tokens", 0))
        self.job.drops.update(state.get("drops", {}))
        hash_path = self.scratch / "dedup.json"
        try:
            remote = hf_hub_download(self.repo, f"staging/{RUN_ID}/state/dedup.json", repo_type="dataset")
            payload = json.loads(Path(remote).read_text(encoding="utf-8"))
            self.job.seen_prompt.update(payload.get("prompt", []))
            self.job.seen_pair.update(payload.get("pair", []))
            Path(remote).unlink(missing_ok=True)
        except Exception:
            pass
        hash_path.unlink(missing_ok=True)
        self.restored = True
        print("resumed", self.shard_index, dict(self.job.family_tokens), flush=True)

    def flush(self, force: bool = False) -> None:
        rows = self.job.rows
        if not rows:
            self._upload_state()
            return
        blob = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        name = f"train-{self.shard_index:05d}.jsonl"
        path = self.scratch / name
        path.write_text(blob, encoding="utf-8")
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        self.api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"staging/{RUN_ID}/train/{name}",
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=f"stage shard {name}",
        )
        info = self.api.get_paths_info(self.repo, [f"staging/{RUN_ID}/train/{name}"], repo_type="dataset")[0]
        if info.size != path.stat().st_size:
            raise RuntimeError(f"remote size mismatch for {name}")
        self.shards.append({
            "filename": name,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "records": len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "target_tokens": sum(row["target_tokens"] for row in rows),
        })
        path.unlink()
        self.job.rows.clear()
        self.shard_index += 1
        self._upload_state()
        print("uploaded", name, digest, flush=True)

    def _upload_state(self) -> None:
        job = self.job
        state = {
            "builder_version": BUILDER,
            "contract_commit": "000813d2294b6ec8f2287d4eb46f64c105d0617c",
            "seed": SEED,
            "tokenizer_revision": REVISION,
            "build_status": "STAGING",
            "next_shard": self.shard_index,
            "shards": self.shards,
            "family_tokens": dict(job.family_tokens),
            "language_tokens": dict(job.lang_tokens),
            "class_tokens": dict(job.class_tokens),
            "difficulty_tokens": dict(job.diff_tokens),
            "identity_tokens": job.identity_tokens,
            "targets": job.targets,
            "drops": dict(job.drops),
            "sources": {name: dict(counter) for name, counter in job.sources.items()},
        }
        state_path = self.scratch / "build_state.json"
        dedup_path = self.scratch / "dedup.json"
        state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        dedup_path.write_text(
            json.dumps({"prompt": sorted(job.seen_prompt), "pair": sorted(job.seen_pair)}, separators=(",", ":")),
            encoding="utf-8",
        )
        self.api.upload_file(
            path_or_fileobj=str(state_path),
            path_in_repo=f"staging/{RUN_ID}/state/build_state.json",
            repo_id=self.repo,
            repo_type="dataset",
            commit_message="update staging build state",
        )
        self.api.upload_file(
            path_or_fileobj=str(dedup_path),
            path_in_repo=f"staging/{RUN_ID}/state/dedup.json",
            repo_id=self.repo,
            repo_type="dataset",
            commit_message="update staging dedup index",
        )
        state_path.unlink()
        dedup_path.unlink()


def main() -> None:
    if os.environ.get("S1_P0_HUB") == "1":
        os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/tmp/hf")
        os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", "/tmp/hf/hub")
        os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", "/tmp/hf/datasets")
    job = build()
    if os.environ.get("S1_P0_HUB") == "1":
        job._maybe_flush(force=True)
        print("staging-shards", job.sink.shard_index, dict(job.family_tokens), flush=True)
        return
    report = write_outputs(job.rows, job.targets, job.sources, job.drops)
    print(json.dumps({k: report[k] for k in ("records", "target_tokens", "language_tokens", "class_tokens", "family_target_tokens", "train_sha256", "objective")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
