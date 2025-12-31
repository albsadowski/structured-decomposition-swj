from pathlib import Path
from typing import Optional
from rdflib import Graph, Namespace, RDF, OWL, RDFS, URIRef, Literal

from .model import TermMention, PredicateMention, OntologyMetadata


class ABoxBuilder:
    def __init__(self, base_namespace: str = "http://example.org/hearsay#"):
        self.graph = Graph()
        self.base_ns = Namespace(base_namespace)
        self.inst_ns = Namespace(base_namespace + "instances/")

        self.graph.bind("", self.base_ns)
        self.graph.bind("inst", self.inst_ns)
        self.graph.bind("owl", OWL)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)

        self._case_count = 0

    def add_case(
        self,
        case_id: str,
        terms: list[TermMention],
        predicates: list[PredicateMention],
        meta: OntologyMetadata,
        inferred_classes: Optional[dict[str, list[str]]] = None,
        input_text: Optional[str] = None,
        is_satisfied: Optional[bool] = None,
    ) -> None:
        case_uri = self.inst_ns[case_id]
        self.graph.add((case_uri, RDF.type, self.base_ns["TestCase"]))

        if input_text:
            self.graph.add((case_uri, RDFS.comment, Literal(input_text[:500])))

        if is_satisfied is not None:
            self.graph.add(
                (case_uri, self.base_ns["hasSatisfied"], Literal(is_satisfied))
            )

        individuals = {}
        for term in terms:
            if not term.term_name:
                continue

            ind_uri = self.inst_ns[f"{case_id}_{term.term_name}"]
            individuals[term.term_name] = ind_uri

            self.graph.add((ind_uri, self.base_ns["belongsToCase"], case_uri))
            self.graph.add((ind_uri, RDFS.label, Literal(term.text)))
            self.graph.add(
                (ind_uri, self.base_ns["confidence"], Literal(term.confidence))
            )

            term_class = meta.term_class_map.get(term.term_name)
            if term_class:
                self.graph.add((ind_uri, RDF.type, URIRef(term_class)))

        for pred in predicates:
            mapping = meta.predicate_map.get(pred.name)
            if not mapping:
                continue

            pred_terms = mapping["terms"]

            if mapping["class"] and len(pred_terms) == 1:
                ind_uri = individuals.get(pred_terms[0])
                if ind_uri:
                    self.graph.add((ind_uri, RDF.type, URIRef(mapping["class"])))
            elif mapping["property"] and len(pred_terms) == 2:
                subj_uri = individuals.get(pred_terms[0])
                obj_uri = individuals.get(pred_terms[1])
                if subj_uri and obj_uri:
                    self.graph.add((subj_uri, URIRef(mapping["property"]), obj_uri))

        if inferred_classes:
            for term_name, class_uris in inferred_classes.items():
                ind_uri = individuals.get(term_name)
                if ind_uri:
                    for class_uri in class_uris:
                        self.graph.add((ind_uri, RDF.type, URIRef(class_uri)))

        self._case_count += 1

    def save(self, path: Path, format: str = "turtle") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(path, format=format)

    @property
    def case_count(self) -> int:
        return self._case_count
