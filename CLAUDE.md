# CLAUDE.md — Hallucination Probe for LLM Agents

This file gives Claude Code full context to continue this project.
Read `docs/` for deeper specs on each phase.

---

## Project goal

Build **32 independent binary classifiers** (one per transformer layer) that predict —
in real time — whether an LLM agent is about to hallucinate during a tool call, by
probing the **residual stream** (hidden states) of every transformer block
simultaneously.

Each classifier intercepts the hidden state of its layer *before* the model emits its
tool call response and outputs a probability of hallucination (0 = correct call,
1 = hallucination). A probing study (AUROC vs layer index) identifies the most
discriminative layer, which then acts as a real-time guardrail.

Il progetto è stato esteso al **multi-modello** per la tesi: stesso pipeline,
stesso classificatore, applicato a due modelli con la stessa famiglia di
architettura (32 layer, hidden=4096) e taglio differente sui dati di training.

---

## Target models

| Model key | Model ID | Layers | Hidden | Quant | Tool template |
|---|---|---|---|---|---|
| `qwen`  | `Qwen/Qwen3.5-9B`                       | 32 | 4096 | NF4 4-bit | system prompt `<tool_call>` |
| `llama` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 32 | 4096 | NF4 4-bit | native `apply_chat_template(tools=)` |

| Field comune | Value |
|---|---|
| Pooling | Last-token hidden state of the prefill (`hidden[0, -1, :]`) |
| Hook target | All 32 transformer blocks (`model.model.layers[0..31]`) |
| Attention | `sdpa` (PyTorch SDPA, O(n) memory) |

Il registry `MODEL_CONFIGS` in [phase1e2/runner.py](phase1e2/runner.py) descrive
ogni modello supportato. Per attivare il template nativo in fase di inferenza:

```bash
# Qwen: usa system prompt personalizzato (default)
python pipeline.py --model "Qwen/Qwen3.5-9B" ...

# Llama: usa apply_chat_template(tools=...) nativo
python pipeline.py --model "meta-llama/Meta-Llama-3.1-8B-Instruct" --use_native_tools ...
```

**Perché due strade diverse**: Qwen segue affidabilmente le istruzioni del
system prompt e produce `<tool_call>{...}</tool_call>`. Llama-3.1, invece, è
fine-tuned sul proprio template nativo: senza `tools=` improvvisa formati
(p.es. `<function_name>{...}</function_name>` con punti nel nome → XML non
valido → parse fail → label spuria=1). Passare `tools=` al template fa sì
che Llama emetta il formato JSON canonico atteso.

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
Loss:   BCELoss(reduction="none") con pesi per-sample applicati manualmente:
          pos_weight = n_neg / n_pos
          weights = where(y==1, pos_weight, 1.0)
          loss = (BCE(pred, y) * weights).mean()
Optim:  AdamW(lr=1e-3, weight_decay=1e-4)
```

**Importante**: `BCELoss` usa `reduction="none"` perché l'architettura include
`Sigmoid()` finale (non possiamo usare `BCEWithLogitsLoss` con `pos_weight`
nativo). Con `reduction="mean"` i pesi sarebbero applicati dopo la media e
verrebbero silenziosamente diluiti.

**Riproducibilità**: `train.py` chiama `set_seed(seed + layer_idx)` all'inizio
di `train_one_layer()`. Risultato: layer X produce sempre lo stesso modello
indipendentemente dall'ordine in cui i layer vengono addestrati e da quanti
layer si decidono di addestrare in una sessione.

---

## Current status

| Phase | Stato | Output |
|---|---|---|
| Phase 1 — Test Suite, Dataset & Activation Capture | ✅ Completa | `outputs/*/labeled_dataset.jsonl`, `outputs/*/activations/` |
| Phase 2 — Residual Stream Capture | ✅ Integrata nella Phase 1 | gestita da `pipeline.py --capture_activations` |
| Phase 3 — Per-Layer Classifier Training | ✅ Completa | `classifiers_*/`, `metrics.json`, `metrics_per_layer.png` |
| Phase 4 — Eval & Cross-distribution | ✅ Completa | `eval_results*/`, `results.json`, heatmap/bar/grid |
| **Multi-modello (Qwen + Llama)** | ✅ Completo | matrice 2×2 + colonna merged, entrambi i modelli |

### Disegno sperimentale finale

Due famiglie di valutazione, sempre su test set held-out:
1. **Colonna merged**: test su distribuzione mista, training su single / multi / merged.
2. **Matrice di generalizzazione cross-distribution 2×2**: incrocia i due regimi
   (single-turn, multi-turn) in addestramento e valutazione. Diagonale =
   in-distribution; fuori diagonale = transfer.

### Risultati finali — matrice 2×2 (AUROC del best layer)

| Train → Test | Tipo | Qwen (best layer) | Llama (best layer) |
|---|---|---|---|
| single → single | in-dist | **0.82** (L19) | **0.86** (L15) |
| multi → multi | in-dist | **0.93** (L13) | **0.77** (L20) |
| single → multi | transfer | 0.61 (L10) | 0.61 (L0) |
| multi → single | transfer | 0.72 (L13) | 0.67 (L2) |

### Risultati finali — colonna merged (test = merged)

| Train | Qwen | Llama |
|---|---|---|
| single | 0.92 | 0.85 |
| multi | 0.92 | 0.69 |
| merged | 0.94 | 0.87 |

**Finding principale (cross-model)**: il segnale di allucinazione nel residual
stream è **rilevabile ma specifico della distribuzione**, non universale. Esiste
in-distribution in ciascun regime (AUROC 0.82–0.93, a profondità intermedia,
layer 15–20), ma **non generalizza** tra regimi (transfer 0.61–0.72), e nel
transfer il best layer collassa verso layer superficiali (correlato strutturale,
non semantico). Lo stesso pattern — diagonale forte, transfer debole, asimmetria
multi→single > single→multi — si manifesta su **entrambi i modelli**, suggerendo
una proprietà generale e non una peculiarità di un singolo modello.

**Confound del merged**: l'AUROC alto sul test merged (fino a 0.94) è in parte
inflazionato da un confound strutturale (il test mescola due regimi a
distribuzione di classe opposta). L'effetto è forte in Qwen (gap di hallucination
rate 11.7% vs 84.7%, prestazioni piatte e indifferenti al training set) e debole
in Llama (gap 28.6% vs 76.5%). Per questo la misura non contaminata è la matrice
2×2 per-regime, non il merged.

---

## Repo structure

```
mcpsuite/
├── CLAUDE.md                   ← you are here
├── docs/
│   ├── roadmap.md              ← roadmap completa + risultati per fase
│   ├── phase1_complete.md      ← note implementazione Phase 1 + schema
│   ├── phase2_spec.md          ← spec Phase 2 (integrata in Phase 1)
│   └── data_schema.md          ← schema JSONL + layout dataset BFCL
├── phase1e2/                   ← inferenza single/multi-turn + live tasks
│   ├── loader.py               ← carica + correla domande/ground-truth per id
│   ├── sampler.py              ← `proportional_sample` (--total/--weights, legacy)
│   │                             e `exact_sample` (--counts)
│   ├── runner.py               ← inferenza 4-bit + 32 forward hook
│   │                             • `MODEL_CONFIGS` registry (qwen, llama)
│   │                             • `use_native_tools` (template nativo Llama)
│   │                             • `_apply_template()` route tra Qwen-style
│   │                               e `apply_chat_template(tools=)` Llama
│   │                             • `build_prompt(..., use_native_tools)`
│   │                               skippa il system message interno quando
│   │                               è attivo il template nativo
│   ├── pipeline.py             ← orchestratore CLI
│   │                             • `--model` (model id o path locale)
│   │                             • `--use_native_tools` (richiesto per Llama)
│   │                             • `--counts` mutuamente esclusivo con
│   │                               `--total`/`--weights`
│   │                             • `metrics.json` include il campo `"model"`
│   ├── evaluator.py            ← valutazione deterministica AST
│   │                             • parser step 1b: `<function_name>{json}</function_name>`
│   │                               di Llama (nomi puntati come `game_rewards.get`)
│   │                             • parser step 2b: parallel Llama `{...}; {...}`
│   │                             • evaluate_multi_turn(aggregation_threshold)
│   │                               strategia B2 (label=1 se frac_failed ≥ soglia)
│   ├── reevaluate.py           ← ri-etichetta un dataset esistente con l'evaluator
│   │                             attuale, SENZA rifare l'inferenza (rigenera
│   │                             labeled_dataset/splits/metrics/activations y.npy)
│   │                             • flag `--multi_turn_threshold` (strategia B2)
│   ├── multi_turn_strategy_preview.py ← confronto strategie di aggregazione
│   │                             multi-turn (any / majority / threshold / ...)
│   ├── test_evaluator.py       ← 32 test unitari (include Llama-format)
│   └── test_sampler.py         ← 10 test unitari (proportional + exact_sample)
├── outputs/
│   ├── single_turn/            ← Qwen, 1000 campioni standard
│   ├── single_turn2/           ← Qwen, 494 campioni live
│   ├── single_turn_merged/     ← Qwen, 1494 campioni merge dei due (11.7% halluc)
│   ├── multi_turn/             ← Qwen, 600 campioni multi-turn (84.7% halluc, B2 0.7)
│   └── llama/                  ← Llama, esperimenti multi-modello
│       └── llama/
│           ├── standard/         ← single-turn standard
│           ├── live/             ← live tasks
│           ├── single_turn_merged/  ← merge standard+live
│           ├── multi-turn/       ← multi-turn
│           └── merged_all/       ← merge globale (single+multi)
├── phase3/                     ← training classificatori
│   ├── dataset.py              ← ActivationDataset: memory-map X.npy
│   │                             selezione layer, filtra INFERENCE_ERROR
│   ├── merge_activations.py    ← merge train/val/test e merge metrics.json
│   ├── train.py                ← 32 MLP con 5-fold StratifiedKFold CV + plot
│   │                             • `--seed` globale + `set_seed(seed + layer_idx)`
│   │                               per-layer (riproducibilità indipendente
│   │                               dall'ordine)
│   │                             • metrics.json: `{"seed": N, "layers": [...]}`
│   └── plot_comparison.py      ← confronto AUROC+Accuracy tra due esperimenti
│                                 (gestisce sia formato lista che dict)
├── phase4/
│   ├── eval.py                 ← valutazione test set: AUROC/AUPRC/Acc/Prec/Rec/F1
│   │                             tutte con 95% CI bootstrap, soglia Youden,
│   │                             majority baseline (max tra le due classi),
│   │                             breakdown per categoria. Produce results.json +
│   │                             summary.png + metrics_per_layer.png +
│   │                             best_layer_detail.png (confusion matrix)
│   └── plot_matrix.py          ← figure cross-distribution: heatmap 2×2 train×test,
│                                  bar chart colonna merged, griglia AUROC-per-layer,
│                                  curve AUROC sovrapposte tra modelli
└── data/                       ← dataset BFCL scaricato da HuggingFace
```

---

## Datasets

Hallucination rate finali (post-fix parser + `--use_native_tools` + strategia
multi-turn B2 soglia 0.7). Le tre distribuzioni esistono per entrambi i modelli:
single-turn (1494: simple/multiple/parallel/parallel_multiple + live), multi-turn
(600: base/miss_func/miss_param), merged (2094 = unione).

| Regime | N | Qwen | Llama |
|---|---|---|---|
| single-turn | 1494 | 11.7% | 28.6% |
| multi-turn (B2 0.7) | 600 | 84.7% | 76.5% |
| merged | 2094 | 32.6% | 42.4% |

I valori Llama pre-fix (single ~54%, merged ~67%) erano inflazionati dal mismatch
di formato (parser fail → falsi `NO_CALL_MADE`/`WRONG_CALL_COUNT`); risolti con
`--use_native_tools` + i fix del parser. I dataset esistenti sono stati ri-etichettati
con `reevaluate.py` senza rifare l'inferenza.

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

**Parser multi-format.** `_extract_calls_from_output()` prova in ordine:
1. Qwen `<tool_call>{json}</tool_call>`
1b. Llama `<function_name>{json}</function_name>` (regex relax, gestisce punti nel nome)
2. JSON fenced block (` ```json `)
2b. Llama parallel format `{...}; {...}; ...` (estrazione string-aware di tutti i top-level `{...}`)
3. JSON inline / array di tool calls
4. Python call-string con balanced-paren walker

**Native tool calling format per modello.** `MODEL_CONFIGS` definisce
`use_native_tools` per modello. Qwen=False (system prompt custom), Llama=True
(`apply_chat_template(tools=...)`). Il flag CLI `--use_native_tools` deve
essere abilitato esplicitamente per Llama: senza, il modello improvvisa il
formato di output → label spurie e hidden state non rappresentativi.

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

**Class imbalance.** Lo sbilanciamento dipende dal regime: single-turn a
maggioranza negativa (~12-29% positivi), multi-turn a maggioranza positiva
(~76-85%), merged intermedio. Si usa `pos_weight = n_neg / n_pos` con
`BCELoss(reduction="none")` applicato manualmente per-sample. `pos_weight` è
calcolato **una volta sul training set** (non per batch): equalizza il contributo
delle due classi qualunque sia la maggioritaria. Do not oversample.

**Seed reproducibility.** `set_seed(seed + layer_idx)` viene applicato
all'inizio del training di ogni layer. Allo stesso seed CLI corrisponde
sempre lo stesso modello per ogni layer, indipendentemente dall'ordine
di esecuzione. `StratifiedKFold(shuffle=True, random_state=seed)` per la CV.

**CV pool = 85%, test = 15% held-out.** train (70%) + val (15%) concatenati
formano il CV pool per 5-fold StratifiedKFold. Il test set non viene mai toccato
fino a Phase 4.

**Strategia di aggregazione multi-turn (B2, soglia 0.7).** Un sample multi-turn
ha più turni. La regola "any-turn" (label=1 se almeno un turno fallisce) satura
con il numero di turni e gonfia la classe positiva. Si usa invece una soglia di
maggioranza qualificata: label=1 se `frac_failed ≥ 0.7`. Riduce l'hallucination
rate multi-turn senza scartare dati (Qwen 98.8% → 84.7%, Llama 86.7% → 76.5%).
Parametro `aggregation_threshold` in `evaluate_multi_turn()`.

**Valutazione cross-distribution (matrice 2×2).** Il segnale va misurato
*dentro* ciascun regime (diagonale) e *tra* regimi (fuori diagonale). La
diagonale misura se il segnale esiste; la fuori-diagonale se generalizza.
Risultato: segnale **specifico della distribuzione** (transfer 0.61-0.72 ≪
in-distribution 0.82-0.93), confermato su entrambi i modelli.

**Confound del test merged.** Valutare su test merged dà AUROC alto ma
parzialmente spurio: il test mescola single-turn (poche allucinazioni) e
multi-turn (molte), quindi un classificatore può distinguere il *regime* anziché
l'allucinazione. La forza del confound scala col divario di hallucination rate
tra i regimi (forte in Qwen, debole in Llama). La misura pulita è la matrice 2×2.

**Majority baseline.** L'accuracy va sempre confrontata con la baseline della
classe maggioritaria, calcolata come `max(frac_neg, frac_pos)` — non come
`frac_neg`, che sarebbe errato sui test a maggioranza positiva (multi-turn).

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
# Output: 32 + 10 = 42 tests passed

# ── Qwen — single-turn inference, modalità esatta ───────────────────────────
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn/labeled_dataset.jsonl \
    --model    Qwen/Qwen3.5-9B \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --counts '{"simple":400,"multiple":200,"parallel":200,"parallel_multiple":200}'

# ── Qwen — live tasks (num_gpus=1 obbligatorio) ─────────────────────────────
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn2/labeled_dataset.jsonl \
    --model    Qwen/Qwen3.5-9B \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --capture_activations \
    --counts '{"live_multiple":350,"live_parallel":75,"live_parallel_multiple":75}'

# ── Llama-3.1 — single-turn (richiede --use_native_tools) ───────────────────
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/llama/standard/labeled_dataset.jsonl \
    --model    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --use_native_tools \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --counts '{"simple":400,"multiple":200,"parallel":200,"parallel_multiple":200}'

# ── Llama-3.1 — live (richiede --use_native_tools, num_gpus=1) ──────────────
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/llama/live/labeled_dataset.jsonl \
    --model    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --use_native_tools \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --capture_activations \
    --counts '{"live_multiple":350,"live_parallel":75,"live_parallel_multiple":75}'

# Merge attivazioni + metriche (model-agnostic)
cd phase3 && python merge_activations.py \
    --src_a  ../outputs/single_turn/activations \
    --src_b  ../outputs/single_turn2/activations \
    --out    ../outputs/single_turn_merged/activations \
    --metrics_a   ../outputs/single_turn/metrics.json \
    --metrics_b   ../outputs/single_turn2/metrics.json \
    --metrics_out ../outputs/single_turn_merged/metrics.json

# Training classificatori — fissare il seed per riproducibilità
cd phase3 && python train.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --out_dir         classifiers_single \
    --seed            42 \
    --device          cuda

# Valutazione Phase 4 (best layer o tutti)
cd phase4 && python eval.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --classifiers_dir ../phase3/classifiers_single \
    --out_dir         eval_results_single \
    --layer           23

# Ri-etichetta un dataset esistente senza rifare l'inferenza (es. dopo un fix
# del parser o per applicare la strategia multi-turn B2)
cd phase1e2 && python reevaluate.py \
    --exp_dir ../outputs/multi_turn \
    --multi_turn_threshold 0.7        # opzionale: strategia B2 (default = any-turn)

# Valutazione cross-distribution: stesso eval.py, classificatore e test set di
# regimi diversi (qui: classifier addestrato su single, test su multi)
cd phase4 && python eval.py \
    --activations_dir ../outputs/multi_turn/activations \
    --classifiers_dir ../phase3/classifiers_single \
    --out_dir         eval_results/multi-on-single

# Figure cross-distribution (heatmap 2×2, bar chart merged, griglia/curve AUROC)
# — vedi le funzioni in phase4/plot_matrix.py (best_from_results,
#   plot_train_test_matrix, plot_fixed_test_bars, plot_auroc_per_layer_grid,
#   plot_auroc_curves), richiamabili dai results.json già prodotti da eval.py.

# Confronto due esperimenti (es. Qwen vs Llama)
cd phase3 && python plot_comparison.py \
    --metrics_a classifiers_qwen/metrics.json    --label_a "Qwen3.5-9B" \
    --metrics_b classifiers_llama/metrics.json   --label_b "Llama-3.1-8B" \
    --out       comparison_qwen_vs_llama.png
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
