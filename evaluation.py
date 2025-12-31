#!/usr/bin/env python3

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import structured_decomposition_swj.registry as registry
from structured_decomposition_swj.abox_builder import ABoxBuilder
from structured_decomposition_swj.tasks.llm import chat_model
from structured_decomposition_swj.tasks.predict import (
    predict_sd,
    predict_few_shot,
    predict_sd_direct,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s][%(asctime)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


TTL_FILES = {
    "hearsay": "./tasks/hearsay.ttl",
    "method_application": "./tasks/scierc.ttl",
    "urti": "./tasks/urti.ttl",
    "eligibility_nli": "./tasks/eligibility_nli.ttl",
}


TTL_NO_COMP_FILES = {
    "hearsay": "./tasks/hearsay_no_comp.ttl",
    "method_application": "./tasks/scierc_no_comp.ttl",
    "eligibility_nli": "./tasks/eligibility_nli_no_comp.ttl",
}


def predict(task_name: str, mode: str):
    task = registry.load_task(TTL_FILES[task_name])
    match mode:
        case "few-shot":
            return partial(predict_few_shot, task=task)
        case "cot":
            return partial(predict_few_shot, task=task, cot=True)
        case "sd":
            return partial(predict_sd, task=task)
        case "sd-no-comp":
            return partial(
                predict_sd, task=registry.load_task(TTL_NO_COMP_FILES[task_name])
            )
        case "sd-direct":
            return partial(predict_sd_direct, task=task)
        case _:
            raise ValueError(f"unsupported mode: {mode}")


def iter_modes(args):
    if args.mode == "all":
        yield from ("sd", "sd-no-comp", "sd-direct", "cot", "few-shot")
    else:
        yield args.mode


def iter_tasks(args):
    if args.task == "all":
        yield from ("hearsay", "method_application", "eligibility_nli")
    else:
        yield args.task


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--task", default="hearsay")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--mode", default="few-shot")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--persist-abox", action="store_true")

    args = parser.parse_args()
    if args.task not in TTL_FILES and args.task != "all":
        raise ValueError(f"unsupported task: {args.task}")
    if args.mode not in ["all", "few-shot", "cot", "sd", "sd-direct", "sd-no-comp"]:
        raise ValueError(f"unsupported mode: {args.mode}")
    if args.input is None:
        raise ValueError("input file is required")
    return args


def main():
    args = parse_args()

    logger.info(f"evaluating {args.task} with {args.mode} on {args.model}")
    load_dotenv()
    ds = pd.read_csv(args.input)

    abox_builder = None
    if args.persist_abox:
        abox_builder = ABoxBuilder()

    for task in iter_tasks(args):
        test_df = ds[(ds["task"] == task) & (ds["split"] == "test")]
        train_df = ds[(ds["task"] == task) & (ds["split"] == "train")]

        for mode in iter_modes(args):
            predictions = []

            logger.info(f"Task: {task}, Mode: {mode}")

            for ix, row in tqdm(test_df.iterrows(), total=len(test_df)):
                try:
                    res = predict(task, mode)(
                        llm=chat_model(args.model),
                        train_df=train_df,
                        input=row["text"],
                        debug=args.debug,
                        abox_builder=abox_builder,
                        case_id=f"{task}_{ix}",
                    )
                except Exception as e:
                    logger.info(f"failed to predict for {row['text']}: {e}")
                    res = False
                logger.info(f"res={res} vs expected={row['answer']}")
                predictions.append(res)

            truth = test_df["answer"] == "Yes"
            acc = accuracy_score(truth, predictions)
            prec = precision_score(truth, predictions)
            recall = recall_score(truth, predictions)
            f1 = f1_score(truth, predictions)

            logger.info(f"Accuracy: {acc:.3f}")
            logger.info(f"Precision: {prec:.3f}")
            logger.info(f"Recall: {recall:.3f}")
            logger.info(f"F1: {f1:.3f}")

    if abox_builder:
        print(f"Case count: {abox_builder.case_count}")
        abox_builder.save(Path("./abox.ttl"))


if __name__ == "__main__":
    main()
