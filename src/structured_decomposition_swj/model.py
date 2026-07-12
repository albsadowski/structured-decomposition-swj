from pydantic import BaseModel


class Term(BaseModel):
    name: str
    description: str


class Predicate(BaseModel):
    name: str
    terms: list[str]
    description: str


class Task(BaseModel):
    domain: str
    subdomain: str | None
    task: str
    persona: str
    question: str
    predicate: str
    domain_knowledge: str
    terms: list[Term]
    predicates: list[Predicate]


class OntologyMetadata(BaseModel):
    target_class: str | None = None
    target_term: str
    term_class_map: dict[str, str | None]
    predicate_map: dict[str, dict]
    output_classes: dict[str, str] = {}


class TermMention(BaseModel):
    text: str
    term_name: str | None
    confidence: float
    explanation: str


class PredicateMention(BaseModel):
    name: str
    arguments: list[str]
    confidence: float
    explanation: str


class TermExtractionResult(BaseModel):
    terms: list[TermMention]


class PredicateExtractionResult(BaseModel):
    predicates: list[PredicateMention]


class EvaluationResult(BaseModel):
    is_satisfied: bool
    terms: list[TermMention]
    predicates: list[PredicateMention]
    prediction: str | None = None
