# Structured Decomposition Framework (SWJ)

Implementation of the Structured Decomposition Framework for rule-based reasoning over natural language, integrating LLMs with OWL 2 ontologies and SWRL rules.

This repository accompanies the paper: **"Structured Decomposition for LLM Reasoning: Cross-Domain Validation and Semantic Web Integration"** (submitted to Semantic Web Journal).

## Overview

The framework decomposes rule-based reasoning into three steps:

1. **Entity Identification** - Extract individuals from text according to OWL class definitions
2. **Assertion Extraction** - Determine which properties hold for identified entities
3. **Rule Application** - Apply SWRL rules via OWL reasoner (Pellet) to infer classifications

LLMs serve as *ontology population engines*, translating unstructured text into ABox assertions according to expert-authored TBox specifications. The symbolic reasoner handles inference with deterministic guarantees.

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

Inspect persisted reasoning traces with SPARQL:
```bash
uv run query_abox.py ./abox.ttl --all
```

## Task Ontologies

Task definitions are formalized as OWL 2 ontologies in `./tasks/`:

| Task | Ontology | Domain |
|------|----------|--------|
| Hearsay Determination | `hearsay.ttl` | Legal |
| Method Application | `scierc.ttl` | Scientific |
| Clinical Trial Eligibility | `eligibility_nli.ttl` | Medical |
| URTI Diagnosis | `urti.ttl` | Medical (negative validation) |

Variants with `_no_comp` suffix exclude complementary predicates (used in ablation studies).

Each ontology defines:
- **TBox** - Classes, properties, and SWRL rules encoding classification logic
- **sd: namespace** - Entity/assertion specifications for LLM prompting

## Related Resources

- **Evaluation datasets**: [ruleeval-xd](https://github.com/albsadowski/ruleeval-xd)
- **Prior work**: [structured-decomposition](https://github.com/albsadowski/structured-decomposition) (SMT-based approach)

## License

MIT
