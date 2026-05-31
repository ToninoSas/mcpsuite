# Data Schema Reference

---

## BFCL Dataset Layout (on disk after download)

```
data/
  BFCL_v3_simple.json                ← single-function, single-call
  BFCL_v3_multiple.json              ← multiple functions, one correct
  BFCL_v3_parallel.json              ← one function, multiple simultaneous calls
  BFCL_v3_parallel_multiple.json     ← multiple functions, multiple calls
  BFCL_v3_live_multiple.json         ← real-world: one correct out of N
  BFCL_v3_live_parallel.json         ← real-world: parallel calls
  BFCL_v3_live_parallel_multiple.json
  BFCL_v3_multi_turn_base.json       ← multi-turn, complete information given
  BFCL_v3_multi_turn_miss_func.json  ← multi-turn, function unavailable
  BFCL_v3_multi_turn_miss_param.json ← multi-turn, parameter underspecified
  BFCL_v3_multi_turn_long_context.json  ← escluso (troppo lungo per T4)
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
- Le categorie `live_*` possono includere `role: "system"` messages incorporati
  nei turni. `build_prompt()` li scarta per evitare conflitto con il system
  prompt custom (Qwen) o con il template nativo (Llama).

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
outputs/<experiment>/
  labeled_dataset.jsonl          ← full dataset
  metrics.json                   ← per-run metrics (vedi sotto)
  splits/
    train.jsonl                  ← 70% stratified by category
    val.jsonl                    ← 15%
    test.jsonl                   ← 15%
  activations/                   ← solo se --capture_activations
    train/X.npy + y.npy + meta.jsonl + shape.json
    val/...
    test/...
```

Each line of every `*.jsonl` file:

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

## `metrics.json` (Phase 1 output)

```jsonc
{
  "model": "Qwen/Qwen3.5-9B",          // o "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "total": 1000,
  "n_correct": 917,
  "n_halluc": 83,
  "accuracy": 0.917,
  "hallucination_rate": 0.083,
  "hallucination_types": {
    "WRONG_ARG_VALUES": 30,
    "WRONG_FUNCTION":   25,
    "NO_CALL_MADE":     10,
    "MISSING_ARGS":      8,
    "EXTRA_ARGS":        5,
    "WRONG_CALL_COUNT":  5
  },
  "per_category": {
    "simple": {
      "n": 400, "n_correct": 372, "n_halluc": 28,
      "accuracy": 0.93, "hallucination_rate": 0.07,
      "hallucination_types": { /* ... */ }
    },
    /* ... per category ... */
  },
  "splits": {"train": 700, "val": 150, "test": 150},
  "timings_sec": {
    "load": 0.7, "sample": 0.0, "inference": 1200.3,
    "eval": 0.5, "save": 0.4, "activations": 0.6, "total": 1202.5
  },
  "inference_samples_per_sec": 0.83
}
```

Il campo `"model"` è stato aggiunto con il supporto multi-modello: serve a
tracciare la provenienza dei dati di una run quando si confrontano esperimenti
Qwen vs Llama.

---

## Activation dataset format (Phase 2 output)

```
outputs/<experiment>/activations/
  train/
    X.npy          numpy array, float16, shape (N, 32, 4096)
    y.npy          numpy array, int8,    shape (N,)
    meta.jsonl     one JSON per row: {"id": "...", "category": "...", "hallucination_type": "..."}
    shape.json     {"X_shape": [N, 32, 4096], "num_layers": 32, "hidden_size": 4096, "model": "..."}
  val/             same structure
  test/            same structure
```

Row `i` of `X.npy` corresponds to row `i` of `y.npy` and line `i` of `meta.jsonl`.
The ordering matches the order in the source split JSONL.

Il campo `"model"` in `shape.json` permette di distinguere attivazioni Qwen
da Llama quando si caricano dataset da sorgenti diverse.

---

## Phase 3 `metrics.json` (training output)

Schema attuale (con seed riproducibile):

```jsonc
{
  "seed": 42,
  "layers": [
    {
      "layer": 0,
      "cv_auroc_mean": 0.612, "cv_auroc_std": 0.024,
      "fold_aurocs": [0.601, 0.625, 0.589, 0.633, 0.610],
      "cv_accuracy_mean": 0.687, "cv_accuracy_std": 0.018,
      "fold_accuracies": [/* ... */],
      "mean_epochs": 23,
      "n_cv_pool": 1267,
      "pos_cv_pool": 148
    },
    /* ... un entry per ogni layer 0..31 ... */
  ]
}
```

`plot_comparison.py` accetta anche il vecchio formato (lista piatta senza il
wrapper `{"seed": ..., "layers": ...}`).

---

## Category sizes (approximate, BFCL v3)

| Category | Approx. count | Eval method |
|---|---|---|
| simple | 400 | AST |
| multiple | 200 | AST |
| parallel | 200 | AST |
| parallel_multiple | 50 | AST |
| live_multiple | 1000+ | AST |
| live_parallel | ~16 | AST |
| live_parallel_multiple | ~24 | AST |
| multi_turn_base | 200 | State-based |
| multi_turn_miss_func | 200 | Response-based |
| multi_turn_miss_param | 200 | Response-based |
| multi_turn_long_context | 200 | State-based (escluso) |
| multi_turn_composite | 200 | State-based |

Note: multi-turn state-based evaluation is not fully implemented in Phase 1
(see `docs/phase1_complete.md` — Known limitations). Only response-based
(AST) matching is used for all categories.
