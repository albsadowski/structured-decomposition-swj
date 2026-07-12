#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, ".")

import structured_decomposition_swj.registry as registry
from structured_decomposition_swj.abox_builder import ABoxBuilder
from structured_decomposition_swj.evaluator import evaluate
from structured_decomposition_swj.prompts import TERM_EXTRACTION, PREDICATE_EXTRACTION
from structured_decomposition_swj.tasks.llm import chat_model

MODEL = "gpt-4o-mini"
OUT_DIR = Path("./expressivity/results")


def main():
    load_dotenv()
    task = registry.load_task("./tasks/hearsay.ttl")

    ds = pd.read_csv("./ruleevalxd.csv")
    test_df = ds[(ds["task"] == "hearsay") & (ds["split"] == "test")].copy()
    test_df["len"] = test_df["text"].str.len()
    row = test_df.sort_values("len").iloc[0]
    input_text = row["text"]
    print(f"instance (row {row.name}, answer={row['answer']}): {input_text}")

    term_defs = "\n".join([f"- {v.name}: {v.description}" for v in task.terms])
    predicate_defs = "\n".join(
        [f"- {p.name}({', '.join(p.terms)}): {p.description}" for p in task.predicates]
    )

    term_prompt = TERM_EXTRACTION.format(
        input=input_text,
        term_defs=term_defs,
        domain_knowledge=task.domain_knowledge,
    )

    builder = ABoxBuilder()
    res = evaluate(
        llm=chat_model(MODEL),
        task=task,
        input=input_text,
        abox_builder=builder,
        case_id="hearsay_appendix",
    )

    terms_rendered = "\n".join([f"- {t.term_name}: {t.text}" for t in res.terms])
    predicate_prompt = PREDICATE_EXTRACTION.format(
        input=input_text,
        predicate_defs=predicate_defs,
        terms=terms_rendered,
        domain_knowledge=task.domain_knowledge,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builder.save(OUT_DIR / "appendix_case.ttl")

    terms_json = "\n".join(t.model_dump_json(indent=2) for t in res.terms)
    predicates_json = "\n".join(p.model_dump_json(indent=2) for p in res.predicates)

    report = "\n".join(
        [
            f"MODEL: {MODEL}",
            f"INSTANCE (gold answer: {row['answer']}):",
            input_text,
            "",
            "=" * 70,
            "ENTITY IDENTIFICATION PROMPT (rendered)",
            "=" * 70,
            term_prompt,
            "",
            "=" * 70,
            "MODEL RESPONSE: terms (structured output)",
            "=" * 70,
            terms_json,
            "",
            "=" * 70,
            "ASSERTION EXTRACTION PROMPT (rendered)",
            "=" * 70,
            predicate_prompt,
            "",
            "=" * 70,
            "MODEL RESPONSE: predicates (structured output)",
            "=" * 70,
            predicates_json,
            "",
            "=" * 70,
            f"CLASSIFICATION: is_satisfied={res.is_satisfied} (gold: {row['answer']})",
            "=" * 70,
        ]
    )
    (OUT_DIR / "appendix_prompts.txt").write_text(report)
    print(f"wrote {OUT_DIR / 'appendix_prompts.txt'} and appendix_case.ttl")
    print(f"is_satisfied={res.is_satisfied}, gold={row['answer']}")


if __name__ == "__main__":
    main()
