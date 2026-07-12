import logging

import pandas as pd

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from ..evaluator import evaluate
from ..model import Task, EvaluationResult
from ..prompts import SD_DIRECT


logger = logging.getLogger(__name__)


def as_bool(res) -> bool:
    if not isinstance(res, str):
        return False

    text = res.strip().lower()
    if text.startswith("yes") or text.startswith("ai: yes"):
        return True

    for line in reversed(text.split("\n")):
        if "yes" in line and not ("no" in line):
            return True

    return False


def parse_label(content, labels: list[str]) -> str:
    if not isinstance(content, str):
        return "NONE"
    up = content.upper()
    best, best_pos = "NONE", -1
    for lab in labels:
        pos = up.rfind(lab.upper())
        if pos > best_pos:
            best, best_pos = lab, pos
    return best


def predict_sd(
    llm: BaseChatModel,
    task: Task,
    input: str,
    debug: bool,
    **kwargs,
) -> bool | str:
    res = evaluate(llm, task, input, **kwargs)
    if debug:
        logging.info(f"res: {res}")
    return res.prediction if res.prediction is not None else res.is_satisfied


def predict_few_shot(
    llm: BaseChatModel,
    task: Task,
    train_df: pd.DataFrame,
    input: str,
    debug: bool,
    cot: bool = False,
    **kwargs,
) -> bool | str:
    labels = sorted(str(a) for a in train_df["answer"].unique())
    is_multiclass = not set(labels) <= {"Yes", "No"}
    response_instruction = (
        f"Respond with exactly one of: {', '.join(labels)}."
        if is_multiclass
        else "Respond with a simple Yes or No."
    )

    fs_prompt = FewShotChatMessagePromptTemplate(
        examples=[
            {
                "input": row["text"],
                "output": row["answer"],
                "question": task.question,
            }
            for _, row in train_df.iterrows()
        ],
        example_prompt=ChatPromptTemplate.from_messages(
            [
                ("human", "{input} {question}"),
                ("ai", "{output}"),
            ]
        ),
    )
    chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You're a {persona}.

{domain_knowledge}

{response_instruction}
""",
                ),
                fs_prompt,
                (
                    "human",
                    "{input} {question}" + (" Think step by step." if cot else ""),
                ),
            ]
        )
        | llm
    )

    res = chain.invoke(
        {
            "input": input,
            "question": task.question,
            "domain_knowledge": task.domain_knowledge,
            "persona": task.persona,
            "response_instruction": response_instruction,
        }
    )
    if debug:
        logging.info(f"res: {res.content}")
    return parse_label(res.content, labels) if is_multiclass else as_bool(res.content)


def predict_sd_direct(
    llm: BaseChatModel,
    task: Task,
    input: str,
    debug: bool,
    **kwargs,
) -> bool:
    res = llm.with_structured_output(EvaluationResult).invoke(
        SD_DIRECT.format(
            input=input,
            task_domain=f"{task.domain} / {task.subdomain}"
            if task.subdomain
            else task.domain,
            task_name=task.task,
            domain_knowledge=task.domain_knowledge,
            predicate=task.predicate,
            term_defs="\n".join([f"- {v.name}: {v.description}" for v in task.terms]),
            predicate_defs="\n".join(
                [
                    f"- {p.name}({', '.join(p.terms)}): {p.description}"
                    for p in task.predicates
                ]
            ),
        )
    )

    if debug:
        logging.info(f"res: {res}")

    if isinstance(res, EvaluationResult):
        return res.is_satisfied
    else:
        logging.error("err: failed to parse evaluation result")
        return False
