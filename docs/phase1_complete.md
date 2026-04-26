# Phase 1 — Implementation Notes

## Status: ✅ Complete (25/25 tests passing)

---

## What was built

### `loader.py`

Loads BFCL `.json` files and correlates questions with ground truth by `id`.

Two GT formats exist in the dataset:
- **AST format** (in `possible_answer/`): `{"id": "simple_0", "ground_truth": [{"func_name": {"param": [val1, val2]}}]}`
- **Exec format** (inline in question file): `{"id": "exec_simple_0", "ground_truth": ["func_name(param=val)"], "execution_result_type": ["exact_match"]}`

The loader handles both transparently. `_build_gt_index()` returns both
the GT dict and the `execution_result_type` per id.

### `sampler.py`

Proportional sampling with configurable per-category weights defined in
`DEFAULT_WEIGHTS`. Uses `random.sample` with a fixed seed for reproducibility.
Stratified train/val/test split (70/15/15) preserves category distribution
within each split.

### `evaluator.py`

The heart of Phase 1. Fully deterministic, no external API calls.

**Parser pipeline** (tries in order, returns on first success):
1. Qwen native `<tool_call>{...}</tool_call>` tag
2. JSON fenced block (` ```json ``` `)
3. JSON inline (finds first `[` or `{` and extracts balanced structure)
4. Python call-string with balanced-parenthesis extractor

**Critical bug fixed**: `re.sub(r"^.*?(?=\w+\s*\()", ...)` exhibited a
Python regex engine quirk where zero-length matches at position 0 caused
`re.finditer` to return a second match at position 1, stripping the first
character of the function name. Fixed by removing the `re.sub` entirely and
using a `re.finditer` on `\b([A-Za-z_]\w*(?:\.\w+)*)\s*\(` with explicit
balanced-paren walking.

**Matching rules**:
- Function name: case-insensitive comparison
- Arg values: type coercion (`"20"==20`, `"true"==True`, hyphen/underscore
  normalized), list-of-acceptable-values semantics from BFCL GT format
- Optional params: GT encodes optional by including `""` or `None` in the
  acceptable values list
- Parallel/multiple: bipartite greedy matching — each predicted call must
  match exactly one GT entry; count mismatch detected first

**Hallucination taxonomy**:
```
label=0  →  hallucination_type=None
label=1  →  hallucination_type ∈ {
    NO_CALL_MADE       model produced no parseable tool call
    WRONG_FUNCTION     function name does not match any GT entry
    MISSING_ARGS       required argument absent from predicted call
    EXTRA_ARGS         predicted call has args not in function schema
    WRONG_ARG_VALUES   arg name correct but value outside acceptable set
    WRONG_ARG_NAMES    arg present but under wrong key name
    WRONG_CALL_COUNT   # predicted calls ≠ # GT calls (parallel/multiple)
}
```

### `runner.py`

Wraps HuggingFace `transformers` + `bitsandbytes` (NF4 4-bit).

Key method for Phase 2:
```python
def generate_with_hidden_state(self, messages) -> tuple[str, torch.Tensor]:
    # Registers a forward hook on model.model.layers[-1]
    # Returns (raw_output_string, hidden_state_tensor[batch, seq, hidden])
```

The hook fires during the *prefill* pass. The returned tensor has shape
`[1, seq_len, 4096]` where `seq_len` is the length of the full input
(system prompt + function schemas + user query).

### `pipeline.py`

CLI orchestrator. Runs steps 1-5 in sequence:
1. `load_all()` — load corpus
2. `proportional_sample()` — sample with weights
3. `run_inference_on_samples()` — Qwen inference
4. `evaluate()` per sample — assign labels
5. Write `labeled_dataset.jsonl` + `splits/`

`--skip_inference` flag re-uses existing `model_raw_output` fields for
re-evaluation without re-running inference. Useful when tuning the evaluator.

---

## Output schema

Each line of `labeled_dataset.jsonl`:

```jsonc
{
  "id": "simple_42",
  "category": "simple",                    // BFCL category name
  "question": [[                           // list of turns, each is list of messages
    {"role": "user", "content": "..."}
  ]],
  "functions": [{                          // OpenAI function schema format
    "name": "...",
    "description": "...",
    "parameters": {"type": "dict", "properties": {...}, "required": [...]}
  }],
  "ground_truth": [                        // list of acceptable calls
    {"function_name": {"param": [val1, val2]}}
  ],
  "execution_result_type": [],             // ["exact_match"] for exec samples
  "model_raw_output": "func(a=1, b=2)",   // raw string from Qwen
  "label": 0,                             // 0=correct, 1=hallucination
  "hallucination_type": null,             // null if label=0
  "eval_details": {                       // diagnostic dict from evaluator
    "category": "simple",
    "raw_output_len": 24,
    "predicted_calls": [{"name": "...", "arguments": {...}}],
    "n_predicted": 1,
    "n_expected": 1,
    "comparisons": [{"matched": true, "name_match": true, "args_match": true, ...}]
  }
}
```

---

## Known limitations / future improvements

- Multi-turn evaluation currently applies single-turn AST logic per turn.
  For `multi_turn_*` categories the state-based evaluation (comparing
  backend state after execution) is not implemented — only response-based
  matching. This may inflate false negatives on multi-turn samples.

- The `execution_result_type: "real_time"` category (samples that require
  live API execution) is currently evaluated with AST matching, which may
  not be accurate. These samples could be filtered out or handled separately.

- Qwen's tool-call format may vary between model versions. The parser
  handles 4 formats but new variants may appear. The fallback is
  `NO_CALL_MADE` which is conservative (marks as hallucination).
