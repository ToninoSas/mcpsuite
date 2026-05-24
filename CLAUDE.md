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
Loss:   BCELoss(reduction="none") with pos_weight = n_neg / n_pos  (class imbalance)
Optim:  AdamW(lr=1e-3, weight_decay=1e-4)
```

**Important**: `BCELoss` must use `reduction="none"` so per-sample weights are applied
before averaging. Using `reduction="mean"` silently ignores the `pos_weight`.

---

## Current status

| Phase | Status | Output |
|---|---|---|
| Phase 1 — Test Suite, Dataset & Activation Capture | ✅ Complete | `outputs/labeled_dataset.jsonl`, `outputs/activations/` |
| Phase 2 — Residual Stream Capture | ✅ Merged into Phase 1 | handled by `pipeline.py --capture_activations` |
| Phase 3 — Per-Layer Classifier Training | ✅ Complete | `classifiers_*/`, `metrics.json`, `metrics_per_layer.png` |
| Phase 4 — Eval & Guardrail | ✅ Complete | `eval_results*/`, `results.json`, `summary.png` |

**Tutte le fasi sono complete.**

### Risultati chiave (Phase 4)

| Esperimento | Test set | Best layer | AUROC | F1 |
|---|---|---|---|---|
| Single-turn classifier (1494 campioni) | Single-turn | 30 (test) / 13 (CV) | 0.831 | 0.463 |
| Mixed classifier (1494 single + 600 multi) | Mixed | 23 | 0.961 | 0.907 |
| Mixed classifier | Single-turn | 23 | 0.827 | 0.500 |

**Finding principale**: AUROC ~0.83 è il segnale genuino di allucinazione nel residual
stream, indipendente dal training set. Layer 23 è il layer ottimale stabile attraverso
esperimenti e distribuzioni di dati diverse.

Il mixed classifier su test mixed (0.96) è inflazionato da un confound strutturale
(single-turn vs multi-turn ha distribuzione di label opposta). Il risultato scientifico
rilevante è AUROC 0.83 su test single-turn.

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
├── phase1e2/                   ← inferenza single-turn + live tasks
│   ├── runner.py               ← Qwen 4-bit inference + 32-layer forward hooks
│   │                             (fix: skip system messages nei turni live)
│   ├── sampler.py              ← campionamento BFCL: `proportional_sample` (legacy,
│   │                             --total/--weights) e `exact_sample` (--counts)
│   └── pipeline.py             ← CLI orchestrator (--counts mutuamente esclusivo
│                                 con --total/--weights)
├── outputs/
│   ├── single_turn/            ← 1000 campioni (simple, multiple, parallel, parallel_multiple)
│   │   ├── labeled_dataset.jsonl
│   │   ├── metrics.json
│   │   ├── splits/
│   │   └── activations/train/ val/ test/
│   ├── single_turn2/           ← 494 campioni live (live_multiple, live_parallel,
│   │   └── ...                    live_parallel_multiple)
│   ├── single_turn_merged/     ← merge di single_turn + single_turn2
│   │   ├── metrics.json        ← 1494 campioni totali, 11.7% hallucination rate
│   │   └── activations/train/ val/ test/
│   └── multi_turn/             ← 600 campioni multi-turn
│       └── ...
└── phase3/                     ← training classificatori
    ├── dataset.py              ← ActivationDataset: memory-map X.npy, selezione layer
    ├── merge_activations.py    ← merge train/val/test da due sorgenti + merge metrics.json
    ├── train.py                ← 32 MLP con 5-fold StratifiedKFold CV + plot
    ├── plot_comparison.py      ← confronto AUROC+Accuracy tra due esperimenti
    └── requirements.txt
└── phase4/
    └── eval.py                 ← valutazione test set: AUROC, F1, bootstrap CI,
                                   soglia Youden, breakdown per categoria, plot summary
```

---

## Datasets

| Directory | Campioni | Categorie | Halluc. rate |
|---|---|---|---|
| `outputs/single_turn/` | 1000 | simple, multiple, parallel, parallel_multiple | 8.3% |
| `outputs/single_turn2/` | 494 | live_multiple, live_parallel, live_parallel_multiple | ~18% |
| `outputs/single_turn_merged/` | 1494 | tutte le sopra | 11.7% |
| `outputs/multi_turn/` | 600 | multi_turn_base, miss_func, miss_param, composite | ~99% |

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

**Class imbalance.** Dataset è ~8-12% hallucinations (single-turn). Use
`pos_weight = n_neg / n_pos` in `BCELoss(reduction="none")`. Do not oversample.

**CV pool = 85%, test = 15% held-out.** train (70%) + val (15%) concatenati
formano il CV pool per 5-fold StratifiedKFold. Il test set non viene mai toccato
fino a Phase 4.

**Confound multi-turn.** Il dataset multi-turn ha ~99% allucinazioni. Addestrare
su single+multi dà AUROC artificialmente alto (~0.96) perché il classificatore
impara la distinzione strutturale, non il segnale di allucinazione. Usare il
test set single-turn per misurare il segnale genuino.

**Live categories — fix system message.** Le categorie live (live_multiple,
live_parallel, live_parallel_multiple) incorporano un system message nei turni
della domanda. `build_prompt()` in `runner.py` deve saltare questi messaggi
system interni per evitare duplicati. Fix applicato.

**`max_seq_len=3072` required on Kaggle T4.** Long multi-turn sequences (8000+
tokens) cause OOM during bitsandbytes dequant kernels, which corrupts the CUDA
context (`cudaErrorIllegalAddress` cascade). Truncating to 3072 prevents OOM.

**Single GPU per live categories.** Le live tasks hanno prompt 2-5x più lunghi
degli standard single-turn. Con `num_gpus=2` i picchi di memoria (11 GB) causano
CUDA illegal memory access. Usare `num_gpus=1` (8.3 GB, stabile).

**`attn_implementation="sdpa"`.** Reduces attention memory from O(n²) to O(n)
using PyTorch's built-in SDPA kernel. No extra packages required.

**Sequential dual-GPU model loading.** `bitsandbytes` NF4 init is not thread-safe.
Models are loaded one at a time with `torch.cuda.synchronize()` between loads;
only inference runs in parallel threads.

---

## Commands

```bash
# Install deps (from phase1e2/)
pip install -r requirements.txt

# Download BFCL dataset
huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset --local-dir ./data

# Run unit tests (no GPU needed)
cd phase1e2 && python -m pytest test_evaluator.py test_sampler.py -v

# Single-turn inference — modalità esatta (raccomandata per nuovi esperimenti)
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn/labeled_dataset.jsonl \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --counts '{"simple":400,"multiple":200,"parallel":200,"parallel_multiple":200}'

# Single-turn inference — modalità proporzionale (legacy, backward-compatible)
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn/labeled_dataset.jsonl \
    --total    1000 \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --weights '{"simple":0.40,"multiple":0.20,"parallel":0.20,"parallel_multiple":0.20}'

# Live tasks inference (num_gpus=1 obbligatorio per stabilità memoria)
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn2/labeled_dataset.jsonl \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --capture_activations \
    --counts '{"live_multiple":350,"live_parallel":75,"live_parallel_multiple":75}'

# Merge attivazioni + metriche
cd phase3 && python merge_activations.py \
    --src_a  ../outputs/single_turn/activations \
    --src_b  ../outputs/single_turn2/activations \
    --out    ../outputs/single_turn_merged/activations \
    --metrics_a   ../outputs/single_turn/metrics.json \
    --metrics_b   ../outputs/single_turn2/metrics.json \
    --metrics_out ../outputs/single_turn_merged/metrics.json

# Training classificatori (single-turn)
cd phase3 && python train.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --out_dir         classifiers_single \
    --device          cuda

# Valutazione Phase 4 (best layer o tutti)
cd phase4 && python eval.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --classifiers_dir ../phase3/classifiers_single \
    --out_dir         eval_results_single \
    --layer           23

# Confronto due esperimenti
cd phase3 && python plot_comparison.py \
    --metrics_a classifiers_single/metrics.json   --label_a "Single-turn" \
    --metrics_b classifiers_mixed/metrics.json    --label_b "Mixed" \
    --out       comparison.png
```

---

## BFCL dataset quick reference

- **Download**: `gorilla-llm/Berkeley-Function-Calling-Leaderboard` on HuggingFace
- **Question files**: `data/BFCL_v3_{category}.json` — one JSON object per line
- **Ground truth files**: `data/possible_answer/BFCL_v3_{category}.json`
- **Correlation key**: `id` field (e.g. `"simple_42"`, `"multi_turn_base_7"`)
- **Categories usate**: simple, multiple, parallel, parallel_multiple,
  live_multiple, live_parallel, live_parallel_multiple,
  multi_turn_base, multi_turn_miss_func, multi_turn_miss_param, multi_turn_composite
- **Excluded**: `multi_turn_long_context` (too long even with max_seq_len=3072)

See `docs/data_schema.md` for full field-by-field breakdown.
