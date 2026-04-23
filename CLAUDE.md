# CLAUDE.md — Hallucination Probe for LLM Agents

This file gives Claude Code full context to continue this project.
Read `docs/` for deeper specs on each phase.

---

## Project goal

Build a **binary classifier** that predicts — in real time — whether a
qwen3.5-9B (4-bit quantized) agent is about to hallucinate during a tool call,
by probing the **residual stream** (hidden states) of the model's final
transformer block.

The classifier intercepts the hidden state *before* the model emits its tool
call response and outputs a probability of hallucination (0 = correct call,
1 = hallucination). This acts as a guardrail that can block or reroute the
agent before the bad call is executed.

---

## Target model

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3.5-9B` |
| Quantization | 4-bit NF4 (bitsandbytes double-quant) |
| Hidden size | 3584 (qwen3.5-9B) |
| Num layers | 28 |
| Hook target | `model.model.layers[-1]` (last transformer block) |
| Pooling | Last-token hidden state → vector of dim 3584 |

> The user originally said "qwen3.5-9b" but the actual HF model being used
> is `Qwen/Qwen3.5-9B`. Confirm with the user if they meant a
> different model variant before changing anything.

---

## Classifier architecture

```
Input: last-layer hidden state, pooled → R^3584
  │
  ├─ Linear(3584, 512)
  ├─ BatchNorm1d(512)
  ├─ ReLU()
  ├─ Dropout(0.3)
  ├─ Linear(512, 1)
  └─ Sigmoid()

Output: scalar in [0, 1]  (probability of hallucination)
Loss:   BCELoss with class_weight (dataset is imbalanced)
Optim:  AdamW
```

---

## Current status

| Phase | Status | Output |
|---|---|---|
| Phase 1 — Test Suite & Dataset | ✅ Complete | `outputs/labeled_dataset.jsonl` |
| Phase 2 — Residual Stream Capture | 🔲 Next | `outputs/activations/` (.npy) |
| Phase 3 — Classifier Training | 🔲 Pending | `outputs/classifier.pt` |
| Phase 4 — Eval & Guardrail | 🔲 Pending | inference wrapper |

**Phase 1 is fully implemented and tested (25/25 unit tests pass).**
The next task is Phase 2: run the pipeline with `generate_with_hidden_state()`
and save the hidden states to disk alongside labels.

---

## Repo structure

```
bfcl_probe/
├── CLAUDE.md                   ← you are here
├── docs/
│   ├── roadmap.md              ← full 4-phase roadmap with design rationale
│   ├── phase1_complete.md      ← Phase 1 implementation notes + schema
│   ├── phase2_spec.md          ← Phase 2 spec (residual stream capture)
│   └── data_schema.md          ← JSONL schema + BFCL dataset layout
├── phase1/
│   ├── loader.py               ← loads BFCL, correlates Q + GT by id
│   ├── sampler.py              ← proportional sampling across categories
│   ├── evaluator.py            ← deterministic AST evaluator (no LLM judge)
│   ├── runner.py               ← Qwen 4-bit inference + forward hook stub
│   ├── pipeline.py             ← CLI orchestrator
│   ├── test_evaluator.py       ← 25 unit tests (all passing)
│   ├── requirements.txt
│   ├── data/                   ← BFCL dataset (downloaded separately)
│   └── outputs/                ← labeled_dataset.jsonl + splits/
└── phase2/                     ← TO BE CREATED
    ├── capture.py              ← activation capture script
    ├── dataset.py              ← PyTorch Dataset over .npy activations
    └── train.py                ← classifier training loop
```

---

## Key design decisions (do not change without discussion)

**Deterministic evaluation only — no LLM-as-judge.** The evaluator uses
AST-matching against BFCL ground truth. A model output is `label=1`
(hallucination) if the parsed function call does not match the ground truth
within type-coercion and set-matching rules. This was an explicit user
requirement.

**Hallucination taxonomy.** Seven types are tracked and saved in the JSONL:
`NO_CALL_MADE`, `WRONG_FUNCTION`, `MISSING_ARGS`, `EXTRA_ARGS`,
`WRONG_ARG_VALUES`, `WRONG_ARG_NAMES`, `WRONG_CALL_COUNT`.
This fine-grained label is stored alongside the binary label and may be
used for multi-class experiments later.

**Last-token pooling.** The residual stream vector fed to the classifier is
the hidden state of the *last input token* (the final token before generation
begins), not mean-pooled. This is the most information-dense position for
predicting what the model will generate next. This is a hyperparameter to
ablate in Phase 3.

**Forward hook on `model.model.layers[-1]`.** The hook fires on the last
transformer block's output *during the prefill of the prompt*, not during
generation. This means we classify based on the model's internal state after
reading the full context but before emitting the first token of the response.
The stub is already in `runner.py::TransformersRunner.generate_with_hidden_state()`.

**Proportional sampling across BFCL categories.** Weights are defined in
`sampler.py::DEFAULT_WEIGHTS`. Simple gets ~25%, multiple ~20%,
multi_turn_base ~15%, etc. These are tunable.

---

## Commands

```bash
# Install deps (from phase1/)
pip install -r requirements.txt

# Download BFCL dataset
huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset --local-dir ./phase1/data

# Run unit tests (no GPU needed)
cd phase1 && python -m pytest test_evaluator.py -v

# Run full Phase 1 pipeline (needs GPU + model download ~15 GB)
cd phase1 && python pipeline.py \
    --data_dir ./data \
    --output   ./outputs/labeled_dataset.jsonl \
    --total    2000 \
    --model    Qwen/Qwen3.5-9B

# Quick smoke test (50 samples)
cd phase1 && python pipeline.py --data_dir ./data --total 50
```

---

## What to build next — Phase 2

See `docs/phase2_spec.md` for the full spec. Summary:

1. Create `phase2/capture.py` — iterates over `labeled_dataset.jsonl`,
   calls `runner.TransformersRunner.generate_with_hidden_state()` for each
   sample, and saves:
   - `outputs/activations/{split}/X.npy`  — shape `(N, 3584)` float16
   - `outputs/activations/{split}/y.npy`  — shape `(N,)` int8 labels
   - `outputs/activations/{split}/meta.jsonl` — id, category, hallucination_type

2. Create `phase2/dataset.py` — PyTorch `Dataset` that memory-maps the .npy
   files (no full RAM load).

3. Create `phase2/train.py` — training loop for the binary classifier
   with class_weight, AdamW, early stopping on val AUROC.

The `generate_with_hidden_state()` method in `runner.py` is already
implemented and tested — Phase 2 only needs to call it and wire up the
storage layer.

---

## BFCL dataset quick reference

- **Download**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard` on HuggingFace
- **Question files**: `data/BFCL_v3_{category}.json` — one JSON object per line
- **Ground truth files**: `data/possible_answer/BFCL_v3_{category}.json`
- **Correlation key**: `id` field (e.g. `"simple_42"`, `"multi_turn_base_7"`)
- **Categories used**: simple, multiple, parallel, parallel_multiple,
  multi_turn_base, multi_turn_miss_func, multi_turn_miss_param,
  multi_turn_long_context, multi_turn_composite

See `docs/data_schema.md` for full field-by-field breakdown.
