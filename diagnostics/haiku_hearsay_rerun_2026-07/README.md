# Diagnostic re-run: claude-haiku / hearsay / sd-no-comp (July 2026)

Per-example responses for the original December 2025 runs were not retained. This directory holds a diagnostic re-execution of the single combination `claude-haiku` (claude-haiku-4-5-20251001) x `hearsay` x `sd-no-comp` (the paper's SD condition), used for the failure analysis in the paper ("Diagnosing a Regression Through the ABox").

Command (from the repository root):

```
uv run evaluation.py --input ruleevalxd.csv --task hearsay --model claude-haiku --mode sd-no-comp --persist-abox
```

Executed 2026-07-08, temperature 0.

Results: accuracy 0.660, precision 0.562, recall 1.000, F1 0.719 (41 TP, 32 FP, 0 FN, 21 TN).

Contents:

- `predictions.csv`: per-example index (into `ruleevalxd.csv`), gold label, predicted label, input text.
- `abox.ttl`: the populated ABox for all 94 test cases, persisted after reasoning (case IDs `hearsay_<index>`).
- `responses.jsonl`: raw structured LLM responses (entity identification and assertion extraction) with the model's per-item explanations, exported from the LangChain response cache.
