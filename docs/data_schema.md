# Data Schema Reference

---

## BFCL Dataset Layout (on disk after download)

```
data/
  BFCL_v3_simple.json                ← single-function, single-call
  BFCL_v3_multiple.json              ← multiple functions, one correct
  BFCL_v3_parallel.json              ← one function, multiple simultaneous calls
  BFCL_v3_parallel_multiple.json     ← multiple functions, multiple calls
  BFCL_v3_multi_turn_base.json       ← multi-turn, complete information given
  BFCL_v3_multi_turn_miss_func.json  ← multi-turn, function unavailable
  BFCL_v3_multi_turn_miss_param.json ← multi-turn, parameter underspecified
  BFCL_v3_multi_turn_long_context.json
  BFCL_v3_multi_turn_composite.json
  possible_answer/
    BFCL_v3_simple.json              ← ground truth for simple
    BFCL_v3_multiple.json
    ... (one file per category)
```

All files are **JSONL** (one JSON object per line, no trailing comma).

---

## Question file format

```jsonc
// data/BFCL_v3_simple.json  — one record per line
{
  "id": "simple_42",
  "question": [                          // outer list = turns
    [                                    // inner list = messages in that turn
      {
        "role": "user",
        "content": "I've been playing a game where rolling a six..."
      }
    ]
  ],
  "function": [                          // list of available function schemas
    {
      "name": "calc_binomial_probability",
      "description": "Calculates the probability of getting k successes in n trials.",
      "parameters": {
        "type": "dict",
        "properties": {
          "n": {"type": "integer", "description": "Number of trials."},
          "k": {"type": "integer", "description": "Number of successes."},
          "p": {"type": "float",   "description": "Probability of success."}
        },
        "required": ["n", "k", "p"]
      }
    }
  ]
}
```

Notes:
- `question` is always `list[list[dict]]`. For single-turn it's `[[{"role":"user","content":"..."}]]`.
- `function` is a list even when only one function is available.
- Field is named `"function"` (not `"functions"`) in the raw BFCL files.
  `loader.py` reads it as `rec.get("function", [])` and stores as `sample.functions`.

---

## Ground truth file format — AST

```jsonc
// data/possible_answer/BFCL_v3_simple.json — one record per line
{
  "id": "simple_42",
  "ground_truth": [
    {
      "calc_binomial_probability": {
        "n": [20],           // list of acceptable values for param n
        "k": [5],
        "p": [0.6]
      }
    }
  ]
}
```

The ground truth is a **list of acceptable calls**. For single-turn categories
this is usually a list with one entry. The value of each parameter is itself
a **list of acceptable values** — if `""` or `null` appears in that list,
the parameter is optional.

---

## Ground truth file format — Exec (inline in question file)

For exec categories (e.g. `BFCL_v3_exec_simple.json`), the GT is embedded
directly in the question file:

```jsonc
{
  "id": "exec_simple_0",
  "question": [[{"role": "user", "content": "..."}]],
  "function": [...],
  "execution_result_type": ["exact_match"],   // or "real_time"
  "ground_truth": ["calc_binomial_probability(n=20, k=5, p=0.6)"]
}
```

The GT is a **list of Python call strings** here, not dicts.
`evaluator.py` handles both formats transparently via `_compare_single_call`.

---

## Labeled dataset format (Phase 1 output)

```
phase1/outputs/
  labeled_dataset.jsonl          ← full dataset
  splits/
    train.jsonl                  ← 70% stratified by category
    val.jsonl                    ← 15%
    test.jsonl                   ← 15%
```

Each line of every file:

```jsonc
{
  // ── Identity ─────────────────────────────────────────────────────────────
  "id": "simple_42",
  "category": "simple",

  // ── Input to the model ───────────────────────────────────────────────────
  "question": [[{"role": "user", "content": "..."}]],
  "functions": [{ ... }],              // OpenAI schema format

  // ── Ground truth ─────────────────────────────────────────────────────────
  "ground_truth": [
    {"calc_binomial_probability": {"n": [20], "k": [5], "p": [0.6]}}
  ],
  "execution_result_type": [],         // [] for AST; ["exact_match"] for exec

  // ── Model output ─────────────────────────────────────────────────────────
  "model_raw_output": "calc_binomial_probability(n=20, k=5, p=0.6)",

  // ── Label ────────────────────────────────────────────────────────────────
  "label": 0,                          // 0=correct, 1=hallucination
  "hallucination_type": null,          // null when label=0

  // ── Diagnostics ──────────────────────────────────────────────────────────
  "eval_details": {
    "category": "simple",
    "raw_output_len": 45,
    "predicted_calls": [
      {"name": "calc_binomial_probability", "arguments": {"n": 20, "k": 5, "p": 0.6}}
    ],
    "n_predicted": 1,
    "n_expected": 1,
    "comparisons": [
      {
        "matched": true,
        "name_match": true,
        "args_match": true,
        "missing_args": [],
        "extra_args": [],
        "wrong_values": {},
        "is_correct": true
      }
    ]
  }
}
```

---

## Activation dataset format (Phase 2 output)

```
phase2/outputs/
  train/
    X.npy          numpy array, float16, shape (N, 4096)
    y.npy          numpy array, int8,    shape (N,)
    meta.jsonl     one JSON per row: {"id": "...", "category": "...", "hallucination_type": "..."}
    progress.json  {"processed_ids": ["simple_0", "simple_1", ...]}
  val/             same structure
  test/            same structure
```

Row `i` of `X.npy` corresponds to row `i` of `y.npy` and line `i` of `meta.jsonl`.
The ordering matches the order in the source split JSONL.

---

## Category sizes (approximate, BFCL v3)

| Category | Approx. count | Eval method |
|---|---|---|
| simple | 400 | AST |
| multiple | 200 | AST |
| parallel | 200 | AST |
| parallel_multiple | 50 | AST |
| multi_turn_base | 200 | State-based |
| multi_turn_miss_func | 200 | Response-based |
| multi_turn_miss_param | 200 | Response-based |
| multi_turn_long_context | 200 | State-based |
| multi_turn_composite | 200 | State-based |

Note: multi-turn state-based evaluation is not fully implemented in Phase 1
(see `docs/phase1_complete.md` — Known limitations). Only response-based
(AST) matching is used for all categories.
