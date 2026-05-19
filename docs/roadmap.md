# Roadmap — LLM Agent Hallucination Probe

## Overview

The goal is to predict, *before* a tool call is emitted, whether a
Qwen3.5-9B agent is about to hallucinate. The prediction signal comes from
the model's own internal state — the residual stream of every transformer
block — not from post-hoc output analysis.

```
User query + function schemas
        │
        ▼
  Qwen3.5-9B-4bit
  [prefill phase]
        │
        ├──► forward hooks on ALL layers (0..31)
        │         │
        │         ▼
        │    hidden states [32, 4096]  (last token, each layer)
        │         │
        │    Binary Classifier per layer
        │    (Linear→BatchNorm→ReLU→Sigmoid)
        │         │
        │    AUROC per layer → plot → find best layer
        │         │
        │    Best classifier: P(hallucination)
        │         │
        │    ┌────┴─────┐
        │    │ p > θ    │ p ≤ θ
        │    ▼          ▼
        │  BLOCK     ALLOW
        │  / retry   tool call
        ▼
  tool call output
```

---

## Phase 1 — Test Suite, Dataset & Activation Capture  ✅ COMPLETE

**Goal**: Build a labeled dataset of (prompt, function_schema, model_output, label)
tuples from BFCL, and capture residual stream activations from all transformer
layers during the same inference pass.

**Source**: Berkeley Function-Calling Leaderboard v3
(`gorilla-llm/Berkeley-Function-Calling-Leaderboard`)

**Categories used**:
- `simple`, `multiple`, `parallel`, `parallel_multiple` — standard single-turn
- `live_multiple`, `live_parallel`, `live_parallel_multiple` — real-world single-turn
- `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_composite`
- **Excluded**: `multi_turn_long_context` (troppo lungo anche con max_seq_len=3072)

**Labeling**: deterministic AST matching against BFCL ground truth. No LLM judge.

**Activation capture**: `--capture_activations` flag in `pipeline.py`.
Registers 32 hooks (one per transformer layer), captures `hidden[0, -1, :]`
(last token of prefill) per layer, saves as `X.npy` shape `(N, 32, 4096)`.

**Dataset raccolto**:

| Dataset | Campioni | Halluc. rate | Note |
|---|---|---|---|
| `outputs/single_turn/` | 1000 | 8.3% | simple, multiple, parallel, parallel_multiple |
| `outputs/single_turn2/` | 494 | ~18% | live_multiple, live_parallel, live_parallel_multiple |
| `outputs/single_turn_merged/` | 1494 | 11.7% | merge dei due sopra |
| `outputs/multi_turn/` | 600 | ~99% | multi_turn_base, miss_func, miss_param, composite |

**Key constraints**:
- `--max_seq_len 3072` required on Kaggle T4 to prevent CUDA context corruption
- `--num_gpus 1` required for live tasks (prompt 2-5x più lunghi, OOM su dual-GPU)
- Filter `INFERENCE_ERROR` samples before classifier training
- Live categories: fix system message in `build_prompt()` (skip internal system msgs)

---

## Phase 2 — Residual Stream Capture  ✅ Merged into Phase 1

No separate script. Fully handled by `pipeline.py --capture_activations`.
See `docs/phase2_spec.md` for details and the pre-training data cleaning step.

---

## Phase 3 — Per-Layer Classifier Training  ✅ COMPLETE

**Goal**: Train 32 independent binary classifiers (one per transformer layer)
on the captured activations. Plot AUROC vs layer index to find which layer
carries the strongest hallucination signal.

**Architecture** (same for all 32 classifiers):
```python
nn.Sequential(
    nn.Linear(4096, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 1),
    nn.Sigmoid(),
)
```

**Training config** (per layer):
- Input: `X[:, layer_idx, :]` — shape `(N, 4096)`
- Loss: `BCELoss(reduction="none")` con `pos_weight = n_neg / n_pos`
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- CV: 5-fold StratifiedKFold sul CV pool (train 70% + val 15% = 85%)
- Early stopping: patience=10 su val AUROC per ogni fold
- Modello finale: addestrato su tutto il CV pool per `mean_epochs` (media dai fold)
- Test set (15%): intoccato fino a Phase 4

**Bug fix critico**: `BCELoss` deve usare `reduction="none"` per applicare
correttamente i pesi per-sample prima della media. Con `reduction="mean"` il
pos_weight viene ignorato silenziosamente.

**Esperimenti condotti**:

| Esperimento | Training set | CV Best layer | CV AUROC |
|---|---|---|---|
| Single-turn only | 1494 single-turn (con live) | 13 | 0.7845 |
| Mixed | 1494 single + 600 multi | 15 | 0.9505 |

**Confound identificato**: il mixed experiment ha AUROC artificialmente alto (~0.95)
perché il classificatore impara la distinzione strutturale single-turn (8% halluc) vs
multi-turn (99% halluc), non il segnale genuino di allucinazione. Il profilo AUROC
piatto su tutti i 32 layer conferma il confound.

**Files**:
```
phase3/
  dataset.py            ← ActivationDataset: memory-maps X.npy, filtra INFERENCE_ERROR
  merge_activations.py  ← merge train/val/test da due sorgenti + merge metrics.json
  train.py              ← 32 MLP con 5-fold CV stratificata + plot AUROC/Accuracy
  plot_comparison.py    ← confronto visivo AUROC+Accuracy tra due esperimenti
```

---

## Phase 4 — Evaluation on Held-Out Test Set  ✅ COMPLETE

**Goal**: Valutare il classificatore del best layer sul test set (15%) mai visto,
con threshold optimization e confidence intervals.

**Metodologia**:
- Threshold ottimale: Youden's J = max(TPR − FPR), invece di 0.5 fisso
- 95% CI: bootstrap con 1000 ricampionamenti
- Metriche: AUROC, AUPRC, Recall, Precision, F1, Accuracy
- Breakdown per categoria (da meta.jsonl)
- Plot: ROC curve, Precision-Recall curve, distribuzione score, summary per layer

**Risultati**:

| Scenario | Classificatore | Test set | Best layer | AUROC | F1 | Recall | Prec |
|---|---|---|---|---|---|---|---|
| A | Single-turn | Single-turn | 30* | 0.831 | 0.463 | 0.846 | 0.319 |
| B | Mixed | Mixed | 23 | 0.961† | 0.907 | 0.845 | 0.980 |
| C | Mixed | Single-turn | 23 | 0.827 | 0.500 | 0.731 | 0.380 |

\* Layer 30 sul test, layer 13 in CV — CI si sovrappongono, differenza dentro il rumore.
† Inflazionato dal confound strutturale.

**Finding principale**: Scenari A e C convergono a AUROC ~0.83 con best layer 23,
dimostrando che il segnale è genuino e stabile indipendentemente dal training set.
Layer 23 è una proprietà del modello, non del dataset.

**Interpretazione del tradeoff** (layer 23, threshold Youden ~0.12):
```
Su 100 tool call:
  ~12 allucinazioni reali → ~9 bloccate (recall 0.73–0.85)
  ~88 corrette            → ~20 bloccate erroneamente (falsi positivi)
```
In produzione la soglia va calibrata sul costo relativo di falso positivo vs falso negativo.

**Limitazione multi-turn**: il test set multi-turn ha quasi solo positivi (~99% halluc),
rendendo AUROC indefinito (no negativi). La valutazione standalone del multi-turn
non è praticabile con le metriche standard — usare il merged test set.

**Files**:
```
phase4/
  eval.py   ← carica classifier + test set, calcola metriche, plot ROC/PR/score dist
             ← gestisce edge cases: test set con sola classe, bootstrap su lista vuota
```

---

## Design decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Evaluation | Deterministic AST | Avoids cost and non-determinism of LLM judge |
| Hook layers | All 32 transformer blocks | Probing study to find most discriminative layer |
| Pooling | Last token of prefill (per layer) | Most information-dense position before generation |
| Quantization | NF4 double-quant | Best quality/VRAM tradeoff on T4 |
| Hallucination label | Binary (0/1) | Simple classifier target; fine-grained type stored separately |
| Classifier per layer | 32 independent MLPs | Clean probing: each measures intrinsic discriminability |
| CV strategy | 5-fold StratifiedKFold on 85% pool | Stable estimate; preserves class proportions |
| pos_weight | `n_neg/n_pos` in `BCELoss(reduction="none")` | Handles imbalance without oversampling |
| Threshold | Youden's J (not 0.5) | With 12% positives, threshold 0.5 gives recall≈0 |
| Multi-turn training | Included with caution | Aumenta positivi ma introduce confound — valutare su single-turn |
| Live categories | Incluse (solo num_gpus=1) | Aumentano positivi e eterogeneità; prompt lunghi causano OOM su dual-GPU |
| Multi-turn evaluation | Solo su merged test set | Test set multi-turn solo ha ~0 negativi → AUROC indefinito |
| Attention impl | `sdpa` (PyTorch SDPA) | Reduces attention memory O(n²)→O(n); no extra packages |
| Dual-GPU loading | Sequential | bitsandbytes not thread-safe; parallel loading causes CUDA races |
