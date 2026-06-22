# Roadmap — LLM Agent Hallucination Probe

## Overview

The goal is to predict, *before* a tool call is emitted, whether an LLM agent
is about to hallucinate. The prediction signal comes from the model's own
internal state — the residual stream of every transformer block — not from
post-hoc output analysis.

Il progetto è stato esteso al **multi-modello** per la tesi: lo stesso
pipeline è applicato a Qwen3.5-9B (segue affidabilmente un system prompt
custom con `<tool_call>`) e a Llama-3.1-8B-Instruct (richiede il template
nativo `apply_chat_template(tools=...)`).

```
User query + function schemas
        │
        ▼
  LLM 4-bit  (Qwen3.5-9B | Llama-3.1-8B-Instruct)
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

**Multi-model support**:

| Model | CLI flag | Template di tool calling |
|---|---|---|
| `Qwen/Qwen3.5-9B`                       | `--model Qwen/Qwen3.5-9B` | system prompt custom con `<tool_call>` (default) |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | `--model meta-llama/Meta-Llama-3.1-8B-Instruct --use_native_tools` | `apply_chat_template(tools=sample.functions)` |

Il registry `MODEL_CONFIGS` in `runner.py` definisce ogni modello supportato.
`build_prompt(sample, use_native_tools)` omette il system message custom quando
si usa il template nativo, evitando di confondere il modello con due specifiche
sovrapposte. Test suite: 28 test in `test_evaluator.py` (di cui 3 per il
formato `<function_name>{json}</function_name>` di Llama) + 10 test in
`test_sampler.py`.

**Dataset raccolto** (hallucination rate finali, post-fix parser +
`--use_native_tools` + strategia multi-turn B2 soglia 0.7):

| Regime | N | Categorie | Qwen | Llama |
|---|---|---|---|---|
| single-turn | 1494 | simple, multiple, parallel, parallel_multiple + live_* | 11.7% | 28.6% |
| multi-turn (B2 0.7) | 600 | multi_turn_base, miss_func, miss_param | 84.7% | 76.5% |
| merged | 2094 | unione single + multi | 32.6% | 42.4% |

I valori Llama pre-fix (single ~54%, merged ~67%, multi ~99.8%) erano inflazionati
dal mismatch di formato (parser fail → falsi `NO_CALL_MADE`/`WRONG_CALL_COUNT`);
risolti con `--use_native_tools` + fix del parser, e ri-etichettati con
`reevaluate.py` senza rifare l'inferenza.

**Key constraints**:
- `--max_seq_len 3072` required on Kaggle T4 to prevent CUDA context corruption
- `--num_gpus 1` required for live tasks (prompt 2-5x più lunghi, OOM su dual-GPU)
- `--use_native_tools` required for Llama (senza, label spurie da format mismatch)
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
- Loss: `BCELoss(reduction="none")` con pesi per-sample manuali
  (`pos_weight = n_neg / n_pos` applicato via `where(y==1, pos_w, 1.0)`)
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- CV: 5-fold StratifiedKFold sul CV pool (train 70% + val 15% = 85%)
- Early stopping: patience=10 su val AUROC per ogni fold
- Modello finale: addestrato su tutto il CV pool per `mean_epochs` (media dai fold)
- Test set (15%): intoccato fino a Phase 4

**Riproducibilità**: `train.py` accetta `--seed N` e applica
`set_seed(seed + layer_idx)` all'inizio del training di ogni layer. A parità di
seed CLI, layer X produce sempre lo stesso modello, indipendentemente
dall'ordine in cui i layer vengono trainati. `metrics.json` ha schema
`{"seed": N, "layers": [...]}`; `plot_comparison.py` supporta sia il nuovo
formato dict che il vecchio formato lista (backward-compatible).

**Bug fix critico**: `BCELoss` deve usare `reduction="none"` perché
l'architettura include `Sigmoid()` finale, impedendo l'uso di
`BCEWithLogitsLoss` con `pos_weight` nativo. I pesi vengono applicati
manualmente per-sample prima della media.

**Esperimenti condotti**: per ciascun modello (Qwen, Llama) si addestrano tre
classificatori — su single-turn, su multi-turn, su merged — e si valutano nella
matrice cross-distribution 2×2 + colonna merged (risultati finali in Phase 4).

I valori in-distribution finali (best layer per AUROC) sono: single-turn 0.82
(Qwen) / 0.86 (Llama), multi-turn 0.93 (Qwen) / 0.77 (Llama).

**Confound del merged**: addestrare/valutare sul merged dà AUROC inflazionato
perché il classificatore può imparare la distinzione strutturale single-turn vs
multi-turn (distribuzioni di classe opposte) anziché il segnale di allucinazione.
La misura genuina viene dalla valutazione per-regime (matrice 2×2).

**Files**:
```
phase3/
  dataset.py            ← ActivationDataset: memory-maps X.npy, filtra INFERENCE_ERROR
  merge_activations.py  ← merge train/val/test da due sorgenti + merge metrics.json
  train.py              ← 32 MLP con 5-fold CV stratificata + plot AUROC/Accuracy
                          + seed riproducibile per-layer
  plot_comparison.py    ← confronto visivo AUROC+Accuracy tra due esperimenti
                          (gestisce sia formato lista che dict)
```

---

## Phase 4 — Evaluation on Held-Out Test Set  ✅ COMPLETE

**Goal**: Valutare il classificatore del best layer sul test set (15%) mai visto,
con threshold optimization e confidence intervals.

**Metodologia**:
- Threshold ottimale: Youden's J = max(TPR − FPR), invece di 0.5 fisso
- 95% CI: bootstrap (1000 ricampionamenti) per **tutte** le metriche, non solo AUROC
- Metriche: AUROC, AUPRC, Recall, Precision, F1, Accuracy (con majority baseline)
- Breakdown per categoria (da meta.jsonl)
- Disegno **cross-distribution**: matrice 2×2 train×test (single/multi) + colonna merged

**Risultati finali — matrice 2×2 (AUROC del best layer)**:

| Train → Test | Tipo | Qwen (layer) | Llama (layer) |
|---|---|---|---|
| single → single | in-dist | 0.82 (19) | 0.86 (15) |
| multi → multi | in-dist | 0.93 (13) | 0.77 (20) |
| single → multi | transfer | 0.61 (10) | 0.61 (0) |
| multi → single | transfer | 0.72 (13) | 0.67 (2) |

**Colonna merged (test = merged)**: AUROC alto per entrambi (Qwen 0.92–0.94,
Llama 0.69–0.87) ma parzialmente inflazionato dal confound strutturale (vedi sotto).

**Finding principale (cross-model)**: il segnale di allucinazione è **specifico
della distribuzione**, non universale. È forte in-distribution (0.82–0.93, a
profondità intermedia layer 15–20) ma non generalizza tra regimi (transfer
0.61–0.72); nel transfer il best layer collassa su layer superficiali (correlato
strutturale, non semantico). Stesso pattern — diagonale forte, transfer debole,
asimmetria multi→single > single→multi — su **entrambi i modelli**: proprietà
generale, non peculiarità di un singolo modello.

**Confound del test merged**: il test merged mescola due regimi a distribuzione
di classe opposta, quindi un AUROC alto può riflettere la distinzione del regime
anziché l'allucinazione. L'effetto scala col divario di hallucination rate tra
regimi: forte in Qwen (11.7% vs 84.7%), debole in Llama (28.6% vs 76.5%). La
misura non contaminata è la matrice 2×2 per-regime.

**Implicazione applicativa**: un guardrail efficace dev'essere **specializzato
per regime**, non universale.

**Files**:
```
phase4/
  eval.py        ← metriche + 95% CI (tutte), Youden, majority baseline,
                   breakdown per categoria; produce results.json + summary.png +
                   metrics_per_layer.png + best_layer_detail.png (confusion matrix)
  plot_matrix.py ← heatmap 2×2, bar chart merged, griglia/curve AUROC-per-layer
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
| pos_weight | `n_neg/n_pos` manuale in `BCELoss(reduction="none")` | Handles imbalance; Sigmoid() finale impedisce BCEWithLogitsLoss |
| Per-layer seed | `set_seed(seed + layer_idx)` | Riproducibilità indipendente dall'ordine di training dei layer |
| Threshold | Youden's J (not 0.5) | Con classi sbilanciate, threshold 0.5 dà recall≈0 |
| Majority baseline | `max(frac_neg, frac_pos)` | Sui test a maggioranza positiva (multi-turn) `frac_neg` darebbe la baseline sbagliata |
| CI bootstrap | Su tutte le metriche, non solo AUROC | Classi minoritarie piccole → CI larghi, da riportare sempre |
| Multi-turn label | Aggregazione B2 (frac_failed ≥ 0.7) | La regola any-turn satura col n. di turni e gonfia i positivi |
| Valutazione | Cross-distribution (matrice 2×2) | Misura il segnale per-regime senza il confound del merged |
| Live categories | Incluse (solo num_gpus=1) | Aumentano positivi e eterogeneità; prompt lunghi causano OOM su dual-GPU |
| Attention impl | `sdpa` (PyTorch SDPA) | Reduces attention memory O(n²)→O(n); no extra packages |
| Dual-GPU loading | Sequential | bitsandbytes not thread-safe; parallel loading causes CUDA races |
| Multi-model template | Per-model via `MODEL_CONFIGS` + `--use_native_tools` | Qwen segue system prompt; Llama richiede `apply_chat_template(tools=)` nativo per non improvvisare il formato |
| Llama parser | step 1b `<function_name>` + step 2b parallel `{...};{...}` | Recupera falsi positivi dal formato nativo di Llama |
| Re-etichettatura | `reevaluate.py` | Applica fix dell'evaluator/strategia B2 ai dataset esistenti senza rifare l'inferenza |
