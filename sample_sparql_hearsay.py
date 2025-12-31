#!/usr/bin/env python3

import argparse
from pathlib import Path
from rdflib import Graph, Namespace


HEARSAY = Namespace("http://example.org/hearsay#")
INST = Namespace("http://example.org/hearsay#instances/")


QUERIES = {
    "hearsay_count": {
        "description": "Count all Hearsay instances",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT (COUNT(DISTINCT ?s) AS ?hearsay_count)
            WHERE {
                ?s a :Hearsay .
            }
        """,
    },
    "total_cases": {
        "description": "Count total test cases",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT (COUNT(DISTINCT ?case) AS ?total_cases)
            WHERE {
                ?case a :TestCase .
            }
        """,
    },
    "satisfaction_summary": {
        "description": "Count satisfied vs unsatisfied cases",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?satisfied (COUNT(?case) AS ?count)
            WHERE {
                ?case a :TestCase ;
                      :hasSatisfied ?satisfied .
            }
            GROUP BY ?satisfied
        """,
    },
    "introduced_for_legal_issue_count": {
        "description": "Count statements with IntroducedForLegalIssue",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT (COUNT(DISTINCT ?s) AS ?count)
            WHERE {
                ?s :introducedForLegalIssue ?l .
            }
        """,
    },
    "out_of_court_count": {
        "description": "Count OutOfCourtStatement instances",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT (COUNT(DISTINCT ?s) AS ?count)
            WHERE {
                ?s a :OutOfCourtStatement .
            }
        """,
    },
    "has_assertion_count": {
        "description": "Count statements with hasAssertion",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT (COUNT(DISTINCT ?s) AS ?count)
            WHERE {
                ?s :hasAssertion ?a .
            }
        """,
    },
    "proves_truth_count": {
        "description": "Count statements with provesTruthOfAssertion",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT (COUNT(DISTINCT ?s) AS ?count)
            WHERE {
                ?s :provesTruthOfAssertion ?l .
            }
        """,
    },
    "predicate_counts_detailed": {
        "description": "Individual counts for each predicate type",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?predicate_type ?count
            WHERE {
                {
                    SELECT ("Statement" AS ?predicate_type) (COUNT(DISTINCT ?x) AS ?count)
                    WHERE { ?x a :Statement }
                }
                UNION
                {
                    SELECT ("OutOfCourtStatement" AS ?predicate_type) (COUNT(DISTINCT ?x) AS ?count)
                    WHERE { ?x a :OutOfCourtStatement }
                }
                UNION
                {
                    SELECT ("hasAssertion" AS ?predicate_type) (COUNT(DISTINCT ?x) AS ?count)
                    WHERE { ?x :hasAssertion ?a }
                }
                UNION
                {
                    SELECT ("introducedForLegalIssue" AS ?predicate_type) (COUNT(DISTINCT ?x) AS ?count)
                    WHERE { ?x :introducedForLegalIssue ?l }
                }
                UNION
                {
                    SELECT ("provesTruthOfAssertion" AS ?predicate_type) (COUNT(DISTINCT ?x) AS ?count)
                    WHERE { ?x :provesTruthOfAssertion ?l }
                }
                UNION
                {
                    SELECT ("Hearsay" AS ?predicate_type) (COUNT(DISTINCT ?x) AS ?count)
                    WHERE { ?x a :Hearsay }
                }
            }
            ORDER BY DESC(?count)
        """,
    },
    "list_hearsay_cases": {
        "description": "List all cases that resulted in Hearsay",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?case ?statement ?label
            WHERE {
                ?statement a :Hearsay ;
                           :belongsToCase ?case .
                OPTIONAL { ?statement rdfs:label ?label }
            }
            ORDER BY ?case
        """,
    },
    "list_non_hearsay_cases": {
        "description": "List cases where statement is NOT hearsay",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?case ?statement ?label
            WHERE {
                ?case a :TestCase ;
                      :hasSatisfied false .
                ?statement :belongsToCase ?case ;
                           a :Statement .
                OPTIONAL { ?statement rdfs:label ?label }
            }
            ORDER BY ?case
        """,
    },
    "missing_predicates": {
        "description": "Find statements missing key predicates (potential extraction failures)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?case ?statement 
                   (BOUND(?isOOC) AS ?has_out_of_court)
                   (BOUND(?assertion) AS ?has_assertion)
                   (BOUND(?legal) AS ?has_introduced)
                   (BOUND(?proves) AS ?has_proves_truth)
            WHERE {
                ?statement :belongsToCase ?case ;
                           a :Statement .
                OPTIONAL { ?statement a :OutOfCourtStatement . BIND(true AS ?isOOC) }
                OPTIONAL { ?statement :hasAssertion ?assertion }
                OPTIONAL { ?statement :introducedForLegalIssue ?legal }
                OPTIONAL { ?statement :provesTruthOfAssertion ?proves }
            }
            ORDER BY ?case
        """,
    },
    "cases_missing_out_of_court": {
        "description": "Cases with Statement but no OutOfCourtStatement classification",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?case ?statement
            WHERE {
                ?statement :belongsToCase ?case ;
                           a :Statement .
                FILTER NOT EXISTS { ?statement a :OutOfCourtStatement }
                FILTER NOT EXISTS { ?statement a :InCourtStatement }
            }
        """,
    },
    "type_distribution": {
        "description": "Distribution of all RDF types in the ABox",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?type (COUNT(?individual) AS ?count)
            WHERE {
                ?individual rdf:type ?type .
                FILTER(STRSTARTS(STR(?type), "http://example.org/hearsay#"))
            }
            GROUP BY ?type
            ORDER BY DESC(?count)
        """,
    },
    "sample_triples": {
        "description": "Sample 20 triples from the ABox",
        "sparql": """
            SELECT ?s ?p ?o
            WHERE {
                ?s ?p ?o .
            }
            LIMIT 20
        """,
    },
    "false_negatives": {
        "description": "Cases where ground truth is hearsay but system did not classify as Hearsay (error analysis)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?case ?statement ?label ?comment
            WHERE {
                ?case a :TestCase ;
                    :hasSatisfied true ;
                    rdfs:comment ?comment .
                ?statement :belongsToCase ?case ;
                        a :Statement .
                FILTER NOT EXISTS { ?statement a :Hearsay }
                OPTIONAL { ?statement rdfs:label ?label }
            }
        """,
    },
    "false_positives": {
        "description": "Cases where ground truth is NOT hearsay but system classified as Hearsay (error analysis)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?case ?statement ?label ?comment
            WHERE {
                ?case a :TestCase ;
                    :hasSatisfied false ;
                    rdfs:comment ?comment .
                ?statement :belongsToCase ?case ;
                        a :Hearsay .
                OPTIONAL { ?statement rdfs:label ?label }
            }
        """,
    },
    "predicate_bottleneck": {
        "description": "For false negatives: which predicates were missing? (identifies extraction failures)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT 
                ?case
                (BOUND(?ooc) AS ?has_OutOfCourt)
                (BOUND(?assertion) AS ?has_Assertion)
                (BOUND(?introduced) AS ?has_IntroducedFor)
                (BOUND(?proves) AS ?has_ProvesTruth)
            WHERE {
                ?case a :TestCase ;
                    :hasSatisfied true .
                ?statement :belongsToCase ?case ;
                        a :Statement .
                FILTER NOT EXISTS { ?statement a :Hearsay }
                OPTIONAL { ?statement a :OutOfCourtStatement . BIND(true AS ?ooc) }
                OPTIONAL { ?statement :hasAssertion ?a . BIND(true AS ?assertion) }
                OPTIONAL { ?statement :introducedForLegalIssue ?l . BIND(true AS ?introduced) }
                OPTIONAL { ?statement :provesTruthOfAssertion ?p . BIND(true AS ?proves) }
            }
        """,
    },
    "reasoning_trace": {
        "description": "Complete reasoning trace for a specific case (demonstrates auditability)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?case ?entity ?type_or_property ?value ?confidence
            WHERE {
                ?case a :TestCase ;
                    rdfs:comment ?input .
                ?entity :belongsToCase ?case .
                {
                    ?entity rdf:type ?type_or_property .
                    FILTER(STRSTARTS(STR(?type_or_property), "http://example.org/hearsay#"))
                    BIND("(class)" AS ?value)
                }
                UNION
                {
                    ?entity ?type_or_property ?related .
                    ?related :belongsToCase ?case .
                    FILTER(?type_or_property != :belongsToCase)
                    BIND(STR(?related) AS ?value)
                }
                OPTIONAL { ?entity :confidence ?confidence }
            }
            ORDER BY ?case ?entity
        """,
    },
    "missing_predicate_summary": {
        "description": "Aggregate: which predicate is most commonly missing in false negatives?",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT 
                (COUNT(*) AS ?total_false_negatives)
                (SUM(IF(!BOUND(?ooc), 1, 0)) AS ?missing_OutOfCourt)
                (SUM(IF(!BOUND(?assertion), 1, 0)) AS ?missing_HasAssertion)
                (SUM(IF(!BOUND(?introduced), 1, 0)) AS ?missing_IntroducedFor)
                (SUM(IF(!BOUND(?proves), 1, 0)) AS ?missing_ProvesTruth)
            WHERE {
                ?case a :TestCase ;
                    :hasSatisfied true .
                ?statement :belongsToCase ?case ;
                        a :Statement .
                FILTER NOT EXISTS { ?statement a :Hearsay }
                OPTIONAL { ?statement a :OutOfCourtStatement . BIND(true AS ?ooc) }
                OPTIONAL { ?statement :hasAssertion ?a . BIND(true AS ?assertion) }
                OPTIONAL { ?statement :introducedForLegalIssue ?l . BIND(true AS ?introduced) }
                OPTIONAL { ?statement :provesTruthOfAssertion ?p . BIND(true AS ?proves) }
            }
        """,
    },
    "low_confidence_extractions": {
        "description": "Extractions with confidence below threshold (flags uncertain interpretations)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?case ?entity ?label ?confidence
            WHERE {
                ?entity :belongsToCase ?case ;
                        :confidence ?confidence .
                FILTER(?confidence < 0.75)
                OPTIONAL { ?entity rdfs:label ?label }
            }
            ORDER BY ?confidence
        """,
    },
    "in_court_statements": {
        "description": "Statements NOT classified as OutOfCourtStatement (potential in-court statements)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?case ?statement ?label ?comment
            WHERE {
                ?case a :TestCase ;
                    rdfs:comment ?comment .
                ?statement :belongsToCase ?case ;
                        a :Statement .
                FILTER NOT EXISTS { ?statement a :OutOfCourtStatement }
                OPTIONAL { ?statement rdfs:label ?label }
            }
        """,
    },
    "trace_summary": {
        "description": "Summary of which predicates were satisfied across all cases",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?predicate 
                   (SUM(IF(?val = true, 1, 0)) AS ?satisfied)
                   (SUM(IF(?val = false, 1, 0)) AS ?not_satisfied)
                   (COUNT(?val) AS ?total)
            WHERE {
                ?case ?predicate ?val .
                FILTER(STRSTARTS(STR(?predicate), STR(:trace_)))
            }
            GROUP BY ?predicate
            ORDER BY ?predicate
        """,
    },
    "trace_failure_analysis": {
        "description": "Find which predicate most often causes non-hearsay",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?predicate (COUNT(?case) AS ?failures)
            WHERE {
                ?case :hasSatisfied false ;
                      ?predicate false .
                FILTER(STRSTARTS(STR(?predicate), STR(:trace_)))
            }
            GROUP BY ?predicate
            ORDER BY DESC(?failures)
        """,
    },
    "trace_all_satisfied_but_not_hearsay": {
        "description": "Cases where all predicates true but not hearsay (debugging)",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?case
            WHERE {
                ?case :hasSatisfied false .
                FILTER NOT EXISTS {
                    ?case ?pred false .
                    FILTER(STRSTARTS(STR(?pred), STR(:trace_)))
                }
            }
        """,
    },
    "trace_case_detail": {
        "description": "Full trace for each case",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?case ?satisfied ?predicate ?value
            WHERE {
                ?case a :TestCase ;
                      :hasSatisfied ?satisfied ;
                      ?predicate ?value .
                FILTER(STRSTARTS(STR(?predicate), STR(:trace_)))
            }
            ORDER BY ?case ?predicate
        """,
    },
    "trace_pattern_frequency": {
        "description": "Most common predicate satisfaction patterns",
        "sparql": """
            PREFIX : <http://example.org/hearsay#>
            SELECT ?pattern (COUNT(?case) AS ?frequency)
            WHERE {
                SELECT ?case (GROUP_CONCAT(CONCAT(STRAFTER(STR(?pred), "trace_"), "=", STR(?val)); separator=",") AS ?pattern)
                WHERE {
                    ?case a :TestCase ;
                          ?pred ?val .
                    FILTER(STRSTARTS(STR(?pred), STR(:trace_)))
                }
                GROUP BY ?case
            }
            GROUP BY ?pattern
            ORDER BY DESC(?frequency)
            LIMIT 20
        """,
    },
}


def load_graph(path: Path) -> Graph:
    g = Graph()

    suffix = path.suffix.lower()
    g.parse(path, format="turtle")
    return g


def run_query(graph: Graph, sparql: str) -> list[dict]:
    qres = graph.query(sparql)
    results = []
    if qres.vars is None:
        return []
    variables = [str(v) for v in qres.vars]

    for row in qres:
        result = {}
        for i, var in enumerate(variables):
            value = row[i]
            result[var] = str(value) if value is not None else None
        results.append(result)
    return results


def print_results(results: list[dict], query_name: str, description: str = "") -> None:
    print(f"\n{'=' * 70}")
    print(f"Query: {query_name}")
    if description:
        print(f"Description: {description}")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    columns = list(results[0].keys())
    widths = {
        col: max(len(col), max(len(str(r.get(col, ""))[:50]) for r in results))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in results:
        values = [str(row.get(col, ""))[:50].ljust(widths[col]) for col in columns]
        print(" | ".join(values))

    print(f"\nTotal: {len(results)} result(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Query persisted ABox with SPARQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available queries:\n"
        + "\n".join(f"  {k}: {v['description']}" for k, v in QUERIES.items()),
    )
    parser.add_argument("abox_path", type=Path, help="Path to ABox file (.ttl)")
    parser.add_argument(
        "--query", "-q", choices=list(QUERIES.keys()), help="Named query to run"
    )
    parser.add_argument("--all", "-a", action="store_true", help="Run all queries")
    parser.add_argument("--custom", "-c", type=str, help="Custom SPARQL query")
    parser.add_argument(
        "--list-queries", "-l", action="store_true", help="List available queries"
    )

    args = parser.parse_args()

    if args.list_queries:
        print("Available queries:")
        for name, info in QUERIES.items():
            print(f"  {name}: {info['description']}")
        return

    graph = load_graph(args.abox_path)

    if args.all:
        for name, info in QUERIES.items():
            results = run_query(graph, info["sparql"])
            print_results(results, name, info["description"])
    elif args.custom:
        results = run_query(graph, args.custom)
        print_results(results, "custom")
    elif args.query:
        info = QUERIES[args.query]
        results = run_query(graph, info["sparql"])
        print_results(results, args.query, info["description"])
    else:
        for name in [
            "total_cases",
            "hearsay_count",
            "satisfaction_summary",
            "predicate_counts_detailed",
        ]:
            info = QUERIES[name]
            results = run_query(graph, info["sparql"])
            print_results(results, name, info["description"])


if __name__ == "__main__":
    main()
