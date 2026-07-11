from functools import cache
from rdflib import Graph, Namespace, RDF
from .model import Task, Term, Predicate, OntologyMetadata

SD = Namespace("http://example.org/structured-decomposition#")

_ontology_meta = {}
_source_ttl = {}


@cache
def _load_graph(ttl_path: str) -> Graph:
    graph = Graph()
    graph.parse(ttl_path, format="turtle")
    return graph


@cache
def get_graph(task_name: str) -> tuple[Graph, OntologyMetadata]:
    return _load_graph(_source_ttl[task_name]), _ontology_meta[task_name]


def _get_lit(g: Graph, subj, pred):
    obj = g.value(subj, pred)
    return str(obj) if obj else ""


def _get_lit_multi(g: Graph, subj, *preds):
    for pred in preds:
        val = _get_lit(g, subj, pred)
        if val:
            return val
    return ""


def _get_uri(g: Graph, subj, pred):
    obj = g.value(subj, pred)
    return str(obj) if obj else None


def _get_uri_multi(g: Graph, subj, *preds):
    for pred in preds:
        val = _get_uri(g, subj, pred)
        if val:
            return val
    return None


def _get_subjects_multi(g: Graph, *type_uris):
    seen = set()
    for type_uri in type_uris:
        for subj in g.subjects(RDF.type, type_uri):
            if subj not in seen:
                seen.add(subj)
                yield subj


def _parse_terms_string(terms_str: str) -> list[str]:
    if not terms_str:
        return []
    return [t.strip() for t in terms_str.split(",") if t.strip()]


def load_task(ttl_path: str) -> Task:
    g = _load_graph(ttl_path)
    task_node = next(g.subjects(RDF.type, SD.Task))
    terms = []
    term_class_map = {}
    for term_node in _get_subjects_multi(g, SD.Term, SD.EntitySpec):
        name = _get_lit_multi(g, term_node, SD.termName, SD.entityName)
        description = _get_lit_multi(
            g, term_node, SD.termDescription, SD.entityDescription
        )

        if not name:
            continue

        terms.append(
            Term(
                name=name,
                description=description,
            )
        )
        term_class_map[name] = _get_uri(g, term_node, SD.mapsToClass)

    predicates = []
    predicate_map = {}
    for pred_node in _get_subjects_multi(g, SD.Predicate, SD.AssertionSpec):
        name = _get_lit_multi(g, pred_node, SD.predicateName, SD.assertionName)
        description = _get_lit_multi(
            g, pred_node, SD.predicateDescription, SD.assertionDescription
        )

        if not name:
            continue

        terms_raw = _get_lit_multi(
            g, pred_node, SD.predicateTerms, SD.assertionEntities
        )
        pred_terms = _parse_terms_string(terms_raw)

        predicates.append(
            Predicate(
                name=name,
                terms=pred_terms,
                description=description,
            )
        )
        complement_of = _get_uri(g, pred_node, SD.complementOf)
        predicate_map[name] = {
            "class": _get_uri(g, pred_node, SD.mapsToClass),
            "property": _get_uri(g, pred_node, SD.mapsToProperty),
            "terms": pred_terms,
            "is_complementary": complement_of is not None,
        }

    task_name = _get_lit(g, task_node, SD.taskName)
    task = Task(
        domain=_get_lit(g, task_node, SD.domain),
        subdomain=_get_lit(g, task_node, SD.subdomain),
        task=task_name,
        persona=_get_lit(g, task_node, SD.persona),
        question=_get_lit(g, task_node, SD.question),
        domain_knowledge=_get_lit(g, task_node, SD.domainKnowledge),
        predicate="",
        terms=terms,
        predicates=predicates,
    )

    target_class = _get_uri(g, task_node, SD.targetClass)

    output_classes = {}
    for oc in g.objects(task_node, SD.outputClass):
        cls = _get_uri(g, oc, SD.mapsToClass)
        label = _get_lit(g, oc, SD.classLabel)
        if cls and label:
            output_classes[cls] = label

    assert target_class is not None or output_classes, (
        f"neither targetClass nor outputClass found in {ttl_path}"
    )

    _source_ttl[task_name] = ttl_path
    _ontology_meta[task_name] = OntologyMetadata(
        target_class=target_class,
        target_term=_get_lit(g, task_node, SD.targetTerm),
        term_class_map=term_class_map,
        predicate_map=predicate_map,
        output_classes=output_classes,
    )

    return task
