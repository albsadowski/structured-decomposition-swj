#!/usr/bin/env python3
from rdflib import Graph, Namespace, RDF
from rdflib.namespace import OWL


SD = Namespace("http://example.org/structured-decomposition#")
SWRL = Namespace("http://www.w3.org/2003/11/swrl#")

ONTOLOGIES = {
    "Hearsay": "./tasks/hearsay.ttl",
    "Method Application": "./tasks/scierc.ttl",
    "Clinical Trial Eligibility": "./tasks/eligibility_nli.ttl",
    "Multi-class Relation": "./tasks/scierc_relation.ttl",
}

DESCRIPTION_PROPS = [
    SD.entityDescription,
    SD.termDescription,
    SD.assertionDescription,
    SD.predicateDescription,
    SD.domainKnowledge,
    SD.question,
    SD.persona,
]


def rdf_list_len(g: Graph, head) -> int:
    n = 0
    while head and head != RDF.nil:
        n += 1
        head = g.value(head, RDF.rest)
    return n


def metrics(path: str) -> dict:
    g = Graph()
    g.parse(path, format="turtle")

    classes = len(set(g.subjects(RDF.type, OWL.Class)))
    obj_props = len(set(g.subjects(RDF.type, OWL.ObjectProperty)))

    rules = list(g.subjects(RDF.type, SWRL.Imp))
    body_atoms = sum(rdf_list_len(g, g.value(r, SWRL.body)) for r in rules)
    head_atoms = sum(rdf_list_len(g, g.value(r, SWRL.head)) for r in rules)

    words = 0
    for prop in DESCRIPTION_PROPS:
        for lit in g.objects(None, prop):
            words += len(str(lit).split())

    return {
        "classes": classes,
        "object_properties": obj_props,
        "swrl_rules": len(rules),
        "rule_atoms": f"{body_atoms}+{head_atoms}" if rules else "0",
        "description_words": words,
    }


def main():
    header = f"{'Ontology':<28} {'Classes':>7} {'ObjProps':>8} {'Rules':>5} {'Atoms':>6} {'DescWords':>9}"
    print(header)
    print("-" * len(header))
    for name, path in ONTOLOGIES.items():
        m = metrics(path)
        print(
            f"{name:<28} {m['classes']:>7} {m['object_properties']:>8} "
            f"{m['swrl_rules']:>5} {m['rule_atoms']:>6} {m['description_words']:>9}"
        )


if __name__ == "__main__":
    main()
