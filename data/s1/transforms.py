"""Source-row transforms. They do not open datasets."""

from __future__ import annotations

from data.s1.foundation import make_record


def qa_extractive(source_id, config, row_id, language, context, question, answer, family, difficulty, license_name):
    return make_record(
        source_id=source_id,
        source_config=config,
        source_split="train",
        source_row_id=row_id,
        language=language,
        family=family,
        difficulty=difficulty,
        context=context,
        question=question,
        answer=answer,
        license_name=license_name,
        transform="qa_extractive.v1",
    )


def yes_no(source_id, config, row_id, language, context, question, answer, license_name):
    return make_record(
        source_id=source_id,
        source_config=config,
        source_split="train",
        source_row_id=row_id,
        language=language,
        family="S1-03",
        difficulty="D",
        context=context,
        question=question,
        answer=answer,
        license_name=license_name,
        transform="yes_no.v1",
    )


def entailment(source_id, config, row_id, language, premise, hypothesis, answer, license_name):
    question = "Does the premise entail the hypothesis, contradict it, or leave it unknown?"
    if language == "ru":
        question = "Из посылки следует гипотеза, она ей противоречит, или это неизвестно?"
    return make_record(
        source_id=source_id,
        source_config=config,
        source_split="train",
        source_row_id=row_id,
        language=language,
        family="S1-06",
        difficulty="D",
        context=f"Premise: {premise}\nHypothesis: {hypothesis}",
        question=question,
        answer=answer,
        license_name=license_name,
        transform="entailment.v1",
    )


def coreference(source_id, config, row_id, language, context, question, answer, license_name):
    return make_record(
        source_id=source_id,
        source_config=config,
        source_split="train",
        source_row_id=row_id,
        language=language,
        family="S1-05",
        difficulty="E",
        context=context,
        question=question,
        answer=answer,
        license_name=license_name,
        transform="coreference.v1",
    )


def multirc(source_id, config, row_id, language, context, question, answer, license_name):
    return make_record(
        source_id=source_id,
        source_config=config,
        source_split="train",
        source_row_id=row_id,
        language=language,
        family="S1-05",
        difficulty="E",
        context=context,
        question=question,
        answer=answer,
        license_name=license_name,
        transform="multirc.v1",
    )
