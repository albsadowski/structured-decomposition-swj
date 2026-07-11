# Structured Decomposition Framework (SWJ)

Implementation of the Structured Decomposition Framework for rule-based reasoning over natural language, integrating LLMs with OWL 2 ontologies and SWRL rules.

This repository accompanies the paper: **"Structured Decomposition for LLM Reasoning: Cross-Domain Validation and Semantic Web Integration"** (submitted to Semantic Web Journal).

## Overview

The framework decomposes rule-based reasoning into three steps:

1. **Entity Identification** - Extract individuals from text according to OWL class definitions
2. **Assertion Extraction** - Determine which properties hold for identified entities
3. **Rule Application** - Apply SWRL rules via OWL reasoner (Pellet) to infer classifications

LLMs serve as *ontology population engines*, translating unstructured text into ABox assertions according to authored TBox specifications. The symbolic reasoner handles inference with deterministic guarantees.

## Installation

Requires Python 3.13+. Using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

Create a `.env` file with API keys for the LLM providers you intend to use:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
FIREWORKS_API_KEY=...
```

## Usage

### Evaluation

Run evaluation on a task dataset:
```bash
uv run evaluation.py --input ./ruleevalxd.csv --task hearsay --model gpt-4o-mini --mode sd
```

**Arguments:**
- `--input` - Path to CSV file with columns: `split`, `text`, `answer`, `domain`, `task`
- `--task` - Task name (`hearsay`, `method_application`, `eligibility_nli`, or `all`)
- `--model` - Model identifier (e.g., `gpt-4o-mini`, `claude-sonnet-4-5-20250929`)
- `--mode` - Prompting strategy:
  - `few-shot` - Standard few-shot prompting
  - `cot` - Chain-of-thought prompting
  - `sd` - Structured decomposition with SWRL verification
  - `sd-no-comp` - SD without complementary predicates
  - `sd-direct` - SD without symbolic verification (ablation)
- `--persist-abox` - Save populated ABox to `./abox.ttl`
- `--debug` - Enable debug output

### Querying the ABox

Runs with `--persist-abox` write the populated ABox to `./abox.ttl`. Inspect the reasoning traces with SPARQL using `sample_sparql_hearsay.py`, which ships about two dozen named queries over the hearsay namespace:
```bash
uv run sample_sparql_hearsay.py ./abox.ttl --all
```

A ready-to-query hearsay ABox (94 cases) is included at `diagnostics/haiku_hearsay_rerun_2026-07/abox.ttl`:
```bash
uv run sample_sparql_hearsay.py diagnostics/haiku_hearsay_rerun_2026-07/abox.ttl --list-queries
```

## Task Ontologies

Task definitions are formalized as OWL 2 ontologies in `./tasks/`:

| Task | Ontology | Domain |
|------|----------|--------|
| Hearsay Determination | `hearsay.ttl` | Legal |
| Method Application | `scierc.ttl` | Scientific |
| Clinical Trial Eligibility | `eligibility_nli.ttl` | Medical |
| URTI Diagnosis | `urti.ttl` | Medical (negative validation) |
| Relation Classification | `scierc_relation.ttl` | Scientific (multi-class OWL 2 expressivity demonstration) |

Variants with `_no_comp` suffix exclude complementary predicates (used in ablation studies).

Each ontology defines:
- **TBox** - Classes, properties, and SWRL rules encoding classification logic
- **sd: namespace** - Entity/assertion specifications for LLM prompting

## Expressivity Demonstration

The `expressivity/` directory contains the multi-class relation task from the paper's OWL expressivity section. It classifies SciERC relations into seven types using OWL 2 constructs instead of a single SWRL rule: a class hierarchy, existential restrictions, a property chain, symmetric and transitive properties, and a disjointness axiom. The ontology is `tasks/scierc_relation.ttl`.

Run the task through the same pipeline as the binary tasks:
```bash
uv run evaluation.py --task scierc_relation --mode sd --model gpt-5-mini --input expressivity/scierc_relation.csv
```
Per-model, per-condition results (macro-F1, per-class report, per-instance predictions, inconsistency count) are written to `expressivity/results/scierc_relation_<mode>_<model>.json`. The committed files back the relation results table in the paper.

Helper scripts:
- `demo_derivations.py` - reproduces the reasoner derivations on `scierc_relation.ttl` (subsumption, symmetric edges, transitivity, property chains, inconsistency) that the LLM never asserts. Run: `uv run python expressivity/demo_derivations.py`
- `ontology_metrics.py` - computes the per-ontology size metrics (class, property, and rule counts, rule atoms, annotation word volume) for the ontology-engineering table. Run: `uv run python expressivity/ontology_metrics.py`
- `capture_prompts.py` - writes the term and assertion prompts for one hearsay instance to `expressivity/results/appendix_prompts.txt` and `appendix_case.ttl` (the prompt appendix). Run: `uv run python expressivity/capture_prompts.py`

## Results and Analysis

- `results.csv` - aggregate metrics for the main experiments, per model, task, and mode.
- `stats.py` - paired t-tests, Wilcoxon signed-rank, Cohen's d, and confidence intervals over `results.csv`.
- `plot.py` - generates the paper figures (`fig_improvement.png`, `fig_precision_recall.png`) from `results.csv`.

## Related Resources

- **Evaluation datasets**: [ruleeval-xd](https://github.com/albsadowski/ruleeval-xd)
- **Prior work**: [structured-decomposition](https://github.com/albsadowski/structured-decomposition) (SMT-based approach)

## License

MIT
