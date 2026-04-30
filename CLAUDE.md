# CLAUDE.md — Hallucination Probe for LLM Agents

This file gives Claude Code full context to continue this project.
Read `docs/` for deeper specs on each phase.

---

## Project goal

Build **32 independent binary classifiers** (one per transformer layer) that predict —
in real time — whether a Qwen3.5-9B (4-bit quantized) agent is about to hallucinate
during a tool call, by probing the **residual stream** (hidden states) of every
transformer block simultaneously.

Each classifier intercepts the hidden state of its layer *before* the model emits its
tool call response and outputs a probability of hallucination (0 = correct call,
1 = hallucination). A probing study (AUROC vs layer index) identifies the most
discriminative layer, which then acts as a real-time guardrail.

---

## Target model

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3.5-9B` |
| Quantization | 4-bit NF4 (bitsandbytes double-quant) |
| Hidden size | 4096 |
| Num layers | 32 |
| Hook target | All 32 transformer blocks (`model.model.layers[0..31]`) |
| Pooling | Last-token hidden state of the prefill (`hidden[0, -1, :]`) |
| Attention | `sdpa` (PyTorch SDPA, O(n) memory) |

---

## Classifier architecture (per layer — 32 independent instances)

```
Input: hidden state of layer i, last token → R^4096
  │
  ├─ Linear(4096, 512)
  ├─ BatchNorm1d(512)
  ├─ ReLU()
  ├─ Dropout(0.3)
  ├─ Linear(512, 1)
  └─ Sigmoid()

Output: scalar in [0, 1]  (probability of hallucination)
Loss:   BCELoss with pos_weight = n_neg / n_pos  (class imbalance)
Optim:  AdamW(lr=1e-3, weight_decay=1e-4)
```

---

## Current status

| Phase | Status | Output |
|---|---|---|
| Phase 1 — Test Suite, Dataset & Activation Capture | ✅ Complete | `outputs/labeled_dataset.jsonl`, `outputs/activations/` |
| Phase 2 — Residual Stream Capture | ✅ Merged into Phase 1 | handled by `pipeline.py --capture_activations` |
| Phase 3 — Per-Layer Classifier Training | 🔲 Next | `outputs/classifiers/` |
| Phase 4 — Eval & Guardrail | 🔲 Pending | inference wrapper |

**Phases 1 and 2 are complete.** `pipeline.py --capture_activations` produces both
the labeled dataset and `X.npy (N, 32, 4096)` activations in a single pass.
The next task is Phase 3: train 32 independent MLP classifiers and plot AUROC vs layer.

---

## Repo structure

```
mcpsuite/
├── CLAUDE.md                   ← you are here
├── docs/
│   ├── roadmap.md              ← full 4-phase roadmap with design rationale
│   ├── phase1_complete.md      ← Phase 1 implementation notes + schema
│   ├── phase2_spec.md          ← Phase 2 spec (merged into Phase 1)
│   └── data_schema.md          ← JSONL schema + BFCL dataset layout
├── phase1/
│   ├── loader.py               ← loads BFCL, correlates Q + GT by id
│   ├── sampler.py              ← proportional sampling across categories
│   ├── evaluator.py            ← deterministic AST evaluator (no LLM judge)
│   ├── runner.py               ← Qwen 4-bit inference + 32-layer forward hooks
│   ├── pipeline.py             ← CLI orchestrator (phases 1-6 incl. activation save)
│   ├── test_evaluator.py       ← 25 unit tests (all passing)
│   ├── requirements.txt
│   ├── data/                   ← BFCL dataset (downloaded separately)
│   └── outputs/
│       ├── labeled_dataset.jsonl
│       ├── metrics.json
│       ├── splits/             ← train/val/test.jsonl
│       └── activations/
│           ├── train/          ← X.npy (N,32,4096), y.npy, meta.jsonl, shape.json
│           ├── val/
│           └── test/
└── phase2/                     ← TO BE CREATED (Phase 3 work)
    ├── dataset.py              ← PyTorch Dataset over X.npy / y.npy with layer selection
    └── train.py                ← 32 independent MLP classifiers + AUROC vs layer plot
```

---

## Key design decisions (do not change without discussion)

**Deterministic evaluation only — no LLM-as-judge.** The evaluator uses
AST-matching against BFCL ground truth. A model output is `label=1`
(hallucination) if the parsed function call does not match the ground truth
within type-coercion and set-matching rules.

**Hallucination taxonomy.** Eight types are tracked in the JSONL:
`NO_CALL_MADE`, `WRONG_FUNCTION`, `MISSING_ARGS`, `EXTRA_ARGS`,
`WRONG_ARG_VALUES`, `WRONG_ARG_NAMES`, `WRONG_CALL_COUNT`, `INFERENCE_ERROR`.
`INFERENCE_ERROR` samples must be **filtered before classifier training** — they
carry no meaningful hidden state (inference never completed).

**All-layer capture.** 32 forward hooks fire simultaneously during prefill.
Each hook captures `hidden[0, -1, :]` (last token, float16) and moves it to CPU
immediately. VRAM cost: ~0 (no accumulation); RAM cost: ~262 KB per sample.
Output: `X.npy` shape `(N, 32, 4096)`.

**Last-token pooling.** The residual stream vector is the hidden state of the
last input token (the final token before generation begins). This is the most
information-dense position for predicting what the model will generate next.

**32 independent classifiers.** One MLP per layer for a clean probing study.
AUROC vs layer index identifies the most discriminative layer. The best-layer
classifier is the candidate for the production guardrail.

**Class imbalance.** Dataset is ~8% hallucinations. Use
`pos_weight = n_neg / n_pos` in `BCELoss`. Do not oversample.

**Proportional sampling across BFCL categories.** Multi-turn categories
(`miss_func`, `miss_param`) have higher hallucination rates by design and are
included to increase positive samples. `multi_turn_long_context` is excluded
(too long even with truncation). Recommended weights: see `docs/roadmap.md`.

**`max_seq_len=3072` required on Kaggle T4.** Long multi-turn sequences (8000+
tokens) cause OOM during bitsandbytes dequant kernels, which corrupts the CUDA
context (`cudaErrorIllegalAddress` cascade). Truncating to 3072 prevents OOM.

**`attn_implementation="sdpa"`.** Reduces attention memory from O(n²) to O(n)
using PyTorch's built-in SDPA kernel. No extra packages required.

**Sequential dual-GPU model loading.** `bitsandbytes` NF4 init is not thread-safe.
Models are loaded one at a time with `torch.cuda.synchronize()` between loads;
only inference runs in parallel threads.

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

# Recommended full run (Kaggle T4 × 2, ~2000 samples, activations included)
cd phase1 && python pipeline.py \
    --data_dir ./data \
    --output   ./outputs/labeled_dataset.jsonl \
    --total    2000 \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --weights '{"simple":0.20,"multiple":0.15,"parallel":0.10,
                "parallel_multiple":0.05,"multi_turn_base":0.15,
                "multi_turn_miss_func":0.15,"multi_turn_miss_param":0.15,
                "multi_turn_long_context":0.0,"multi_turn_composite":0.05}'

# Quick smoke test (50 samples, no activations, single GPU)
cd phase1 && python pipeline.py --data_dir ./data --total 50
```

### New CLI flags added to pipeline.py

| Flag | Description |
|---|---|
| `--capture_activations` | Capture 32-layer hidden states and save X.npy / y.npy |
| `--max_seq_len N` | Truncate input to last N tokens (use 3072 on T4 16 GB) |
| `--num_gpus N` | Data-parallel inference across N GPUs |
| `--weights '{...}'` | Per-category sampling weights as JSON string |
| `--no_multi_turn` | Zero-weight all multi-turn categories (quick single-turn run) |

---

## What to build next — Phase 3

See `docs/roadmap.md` Phase 3 for the full spec. Summary:

1. Create `phase2/dataset.py` — PyTorch `Dataset` that memory-maps `X.npy` and
   `y.npy`, selects a specific layer by index, and filters `INFERENCE_ERROR` rows.

2. Create `phase2/train.py` — training loop for 32 independent MLP classifiers:
   - Input: `X[:, layer_idx, :]` shape `(N, 4096)`
   - Loss: `BCELoss(pos_weight=n_neg/n_pos)`
   - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
   - Early stopping: patience=10 on val AUROC
   - Output: `outputs/classifiers/layer_{i:02d}.pt` + `metrics.json` + AUROC plot

---

## BFCL dataset quick reference

- **Download**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard` on HuggingFace
- **Question files**: `data/BFCL_v3_{category}.json` — one JSON object per line
- **Ground truth files**: `data/possible_answer/BFCL_v3_{category}.json`
- **Correlation key**: `id` field (e.g. `"simple_42"`, `"multi_turn_base_7"`)
- **Categories used**: simple, multiple, parallel, parallel_multiple,
  multi_turn_base, multi_turn_miss_func, multi_turn_miss_param,
  multi_turn_composite
- **Excluded**: `multi_turn_long_context` (too long even with max_seq_len=3072)

See `docs/data_schema.md` for full field-by-field breakdown.
