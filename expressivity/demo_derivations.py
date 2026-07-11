#!/usr/bin/env python3

import tempfile

from rdflib import URIRef
import structured_decomposition_swj.registry as registry
from owlready2 import (
    World,
    Thing,
    sync_reasoner_pellet,
    OwlReadyInconsistentOntologyError,
)

NS = "http://example.org/scierc-relation#"

registry.load_task("./tasks/scierc_relation.ttl")
_G, _META = registry.get_graph("scierc_relation")

REL_CLASSES = {
    "Rel", "DirectedRel", "SymmetricRel", "UsageRel", "CompositionalRel",
    "TaxonomicRel", "UsedFor", "EvaluateFor", "PartOf", "FeatureOf",
    "HyponymOf", "Compare", "Conjunction",
}


def fresh():
    world = World()
    with tempfile.NamedTemporaryFile(suffix=".rdf", delete=False) as f:
        _G.serialize(f.name, format="xml")
        onto = world.get_ontology(f"file://{f.name}").load()
    return world, onto


def reason(world):
    sync_reasoner_pellet(world, infer_property_values=True, debug=0)


_ORDER = [
    "UsedFor", "EvaluateFor", "PartOf", "FeatureOf", "HyponymOf", "Compare",
    "Conjunction", "UsageRel", "CompositionalRel", "TaxonomicRel", "DirectedRel",
    "SymmetricRel", "Rel",
]


def rel_chain(ind):
    names = [c.name for c in ind.INDIRECT_is_a if getattr(c, "name", "") in REL_CLASSES]
    return sorted(names, key=lambda n: _ORDER.index(n) if n in _ORDER else 99)


def edge(world, subj, prop_local, obj):
    """True if the (inferred) triple is present in the reasoned RDF graph."""
    rg = world.as_rdflib_graph()
    return (URIRef(subj.iri), URIRef(NS + prop_local), URIRef(obj.iri)) in rg


def hr(title):
    print(f"\n--- {title} " + "-" * (60 - len(title)))


def demo_subsumption():
    hr("1. subsumption: one assertion -> multi-level structured output")
    world, onto = fresh()
    with onto:
        m = Thing("Euclidean_distance")
        t = Thing("manifold_learning_methods")
        getattr(m, "usedFor").append(t)
    reason(world)
    print("  LLM asserted:  usedFor(Euclidean_distance, manifold_learning_methods)")
    print(f"  OWL derived :  {' < '.join(rel_chain(m))}")
    print("  -> a query for DirectedRel returns this instance though only the leaf was set")


def demo_symmetric():
    hr("2. symmetric property: reverse edge for free (Compare / Conjunction)")
    world, onto = fresh()
    with onto:
        a = Thing("our_model")
        b = Thing("baseline_CRF")
        getattr(a, "comparedWith").append(b)
    reason(world)
    rev = edge(world, b, "comparedWith", a)
    print("  LLM asserted:  comparedWith(our_model, baseline_CRF)")
    print(f"  OWL derived :  comparedWith(baseline_CRF, our_model) = {rev}")


def demo_transitive_and_chain():
    hr("3. transitivity + property chain: facts across mentions")
    world, onto = fresh()
    with onto:
        head = Thing("attention_head")
        layer = Thing("attention_layer")
        net = Thing("transformer")
        getattr(head, "partOf").append(layer)
        getattr(layer, "partOf").append(net)

        lstm = Thing("LSTM")
        dep = Thing("dependency_parsing")
        parsing = Thing("parsing")
        getattr(lstm, "usedFor").append(dep)
        getattr(dep, "hyponymOf").append(parsing)
    reason(world)
    print("  LLM asserted:  partOf(head,layer), partOf(layer,transformer)")
    print(f"  OWL derived :  partOf(attention_head, transformer) = {edge(world, head, 'partOf', net)}")
    print("  LLM asserted:  usedFor(LSTM,dependency_parsing), hyponymOf(dependency_parsing,parsing)")
    print(f"  OWL derived :  usedFor(LSTM, parsing) = {edge(world, lstm, 'usedFor', parsing)}  (property chain)")


def demo_consistency():
    hr("4. consistency: contradictory extraction is caught (disjoint classes)")
    world, onto = fresh()
    with onto:
        s = Thing("method_X")
        a = Thing("task_A")
        b = Thing("task_B")
        getattr(s, "usedFor").append(a)
        getattr(s, "partOf").append(b)
    detected = False
    try:
        reason(world)
    except OwlReadyInconsistentOntologyError:
        detected = True
    print("  LLM asserted:  usedFor(method_X, task_A) AND partOf(method_X, task_B)")
    print(f"  OWL derived :  ABox inconsistent (UsedFor and PartOf are disjoint) = {detected}")
    print("  -> a contradictory extraction is flagged rather than silently scored")


def main():
    print("=" * 66)
    print("OWL expressivity on the PRODUCTION ontology (tasks/scierc_relation.ttl)")
    print("loaded via the framework's own registry -> owlready2 -> Pellet path")
    print("=" * 66)
    demo_subsumption()
    demo_symmetric()
    demo_transitive_and_chain()
    demo_consistency()
    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()
