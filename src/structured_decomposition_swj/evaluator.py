from functools import cache
from logging import getLogger
import tempfile
from typing import Optional

from nltk.inference import ResolutionProver
from langchain_core.language_models.chat_models import BaseChatModel
from owlready2 import sync_reasoner_pellet, World, Thing

from .model import (
    TermExtractionResult,
    TermMention,
    EvaluationResult,
    PredicateExtractionResult,
    PredicateMention,
    Task,
)
from .prompts import TERM_EXTRACTION, PREDICATE_EXTRACTION
from .registry import get_graph
from .abox_builder import ABoxBuilder

logger = getLogger(__name__)


@cache
def prover():
    return ResolutionProver()


def extract_terms(llm: BaseChatModel, task: Task, input: str) -> list[TermMention]:
    res = llm.with_structured_output(TermExtractionResult).invoke(
        TERM_EXTRACTION.format(
            input=input,
            term_defs="\n".join([f"- {v.name}: {v.description}" for v in task.terms]),
            domain_knowledge=task.domain_knowledge,
        ),
    )
    if isinstance(res, TermExtractionResult):
        return res.terms
    else:
        logger.error("err: failed to parse term extraction result")
        return []


def extract_predicates(
    llm: BaseChatModel, task: Task, input: str, terms: list[TermMention]
) -> list[PredicateMention]:
    res = llm.with_structured_output(PredicateExtractionResult).invoke(
        PREDICATE_EXTRACTION.format(
            input=input,
            predicate_defs="\n".join(
                [
                    f"- {p.name}({', '.join(p.terms)}): {p.description}"
                    for p in task.predicates
                ]
            ),
            terms="\n".join([f"- {t.term_name}: {t.text}" for t in terms]),
            domain_knowledge=task.domain_knowledge,
        ),
    )
    if isinstance(res, PredicateExtractionResult):
        return res.predicates
    else:
        logger.error("err: failed to parse predicate extraction result")
        return []


def evaluate(
    llm: BaseChatModel,
    task: Task,
    input: str,
    min_confidence: float = 0.5,
    abox_builder: Optional[ABoxBuilder] = None,
    case_id: Optional[str] = None,
    **kwargs,
) -> EvaluationResult:
    terms = extract_terms(llm=llm, task=task, input=input)
    predicates = extract_predicates(llm=llm, task=task, input=input, terms=terms)

    g, meta = get_graph(task.task)

    world = World()

    filtered_predicates = [
        p
        for p in predicates
        if not meta.predicate_map.get(p.name, {}).get("is_complementary", False)
        and p.confidence >= min_confidence
    ]

    with tempfile.NamedTemporaryFile(suffix=".rdf", delete=False) as f:
        g.serialize(f.name, format="xml")
        onto = world.get_ontology(f"file://{f.name}").load()

    with onto:
        individuals = {}
        for term in terms:
            if term.term_name:
                individuals[term.term_name] = Thing(f"{term.term_name}_instance")

        for pred in filtered_predicates:
            mapping = meta.predicate_map.get(pred.name)
            if not mapping:
                continue

            pred_terms = mapping["terms"]

            if mapping.get("class") and len(pred_terms) == 1:
                owl_class = world[mapping["class"]]
                ind = individuals.get(pred_terms[0])
                if ind and owl_class and owl_class not in ind.is_a:
                    ind.is_a.append(owl_class)

            elif mapping.get("property") and len(pred_terms) == 2:
                owl_prop = world[mapping["property"]]
                subj = individuals.get(pred_terms[0])
                obj = individuals.get(pred_terms[1])
                if subj and obj and owl_prop:
                    prop_name = mapping["property"].split("#")[-1]
                    getattr(subj, prop_name).append(obj)

        sync_reasoner_pellet(world, infer_property_values=True)

        target_class = world[meta.target_class]
        target_ind = individuals.get(meta.target_term)
        is_satisfied = bool(
            target_ind and target_class and target_class in target_ind.is_a
        )

        inferred_classes = {}
        for term_name, ind in individuals.items():
            if ind:
                inferred_classes[term_name] = [
                    str(c.iri) for c in ind.is_a if hasattr(c, "iri")
                ]

    if abox_builder and case_id:
        abox_builder.add_case(
            case_id=case_id,
            terms=terms,
            predicates=filtered_predicates,
            meta=meta,
            inferred_classes=inferred_classes,
            input_text=input,
            is_satisfied=is_satisfied,
        )

    return EvaluationResult(
        is_satisfied=is_satisfied,
        terms=terms,
        predicates=predicates,
    )
