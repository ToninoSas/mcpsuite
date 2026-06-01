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
| Phase 4 — Eval & Guardrail | ✅ Completa | `eval_results*/`, `results.json`, `summary.png` |
| **Multi-modello (Llama-3.1-8B-Instruct)** | 🟡 Inferenza ed eval in corso | `outputs/llama/...` |

### Risultati Qwen (consolidati)

| Esperimento | Test set | Best layer | AUROC | F1 |
|---|---|---|---|---|
| Single-turn classifier (1494 campioni) | Single-turn | 30 (test) / 13 (CV) | 0.831 | 0.463 |
| Mixed classifier (1494 single + 600 multi) | Mixed | 23 | 0.961 | 0.907 |
| Mixed classifier | Single-turn | 23 | 0.827 | 0.500 |

**Finding principale**: AUROC ~0.83 è il segnale genuino di allucinazione nel
residual stream, indipendente dal training set. Layer 23 è il layer ottimale
stabile attraverso esperimenti e distribuzioni di dati diverse.

Il mixed classifier su test mixed (0.96) è inflazionato da un confound
strutturale (single-turn vs multi-turn ha distribuzione di label opposta). Il
risultato scientifico rilevante è AUROC 0.83 su test single-turn.

### Risultati Llama (preliminari, in revisione)

Le prime esecuzioni Llama sono state condotte **senza** `--use_native_tools` e
hanno prodotto hallucination rate ~54% su single-turn, parzialmente inflazionato
da fallimenti di parsing del formato non-nativo. Il fix è stato applicato in
codice (commit `6f59bd6`) e le ri-esecuzioni con template nativo sono in corso.
I valori target attesi sono ~20–30% hallucination rate single-turn, in linea
con Qwen.

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
│   │                             • parser step 1b per il formato
│   │                               `<function_name>{json}</function_name>` di Llama
│   │                               (gestisce nomi puntati come `game_rewards.get`)
│   ├── test_evaluator.py       ← 32 test unitari (include 7 Llama-format)
│   └── test_sampler.py         ← 10 test unitari (proportional + exact_sample)
├── outputs/
│   ├── single_turn/            ← Qwen, 1000 campioni standard
│   ├── single_turn2/           ← Qwen, 494 campioni live
│   ├── single_turn_merged/     ← Qwen, 1494 campioni merge dei due (11.7% halluc)
│   ├── multi_turn/             ← Qwen, 600 campioni multi-turn (~99% halluc)
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
│   └── eval.py                 ← valutazione test set: AUROC, F1, bootstrap CI,
│                                  soglia Youden, breakdown per categoria, plot
└── data/                       ← dataset BFCL scaricato da HuggingFace
```

---

## Datasets

### Qwen

| Directory | Campioni | Categorie | Halluc. rate |
|---|---|---|---|
| `outputs/single_turn/` | 1000 | simple, multiple, parallel, parallel_multiple | 8.3% |
| `outputs/single_turn2/` | 494 | live_multiple, live_parallel, live_parallel_multiple | ~18% |
| `outputs/single_turn_merged/` | 1494 | tutte le sopra | 11.7% |
| `outputs/multi_turn/` | 600 | multi_turn_base, miss_func, miss_param, composite | ~99% |

### Llama (pre-fix `--use_native_tools` — in attesa di re-run)

| Directory | Campioni | Halluc. rate (attuale) |
|---|---|---|
| `outputs/llama/llama/standard/` | 1000 | 50.0% |
| `outputs/llama/llama/live/` | 494 | 62.5% |
| `outputs/llama/llama/single_turn_merged/` | 1494 | 54.2% |
| `outputs/llama/llama/multi-turn/` | 600 | 99.8% |
| `outputs/llama/llama/merged_all/` | 2094 | 67.2% |

I valori ≥50% sono influenzati dal mismatch di formato (parser fail → `NO_CALL_MADE`).
La metrica genuina si avrà dopo il re-run con `--use_native_tools`.

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

**Class imbalance.** Dataset è ~8-12% hallucinations (single-turn). Use
`pos_weight = n_neg / n_pos` con `BCELoss(reduction="none")` applicato
manualmente per-sample. Do not oversample.

**Seed reproducibility.** `set_seed(seed + layer_idx)` viene applicato
all'inizio del training di ogni layer. Allo stesso seed CLI corrisponde
sempre lo stesso modello per ogni layer, indipendentemente dall'ordine
di esecuzione. `StratifiedKFold(shuffle=True, random_state=seed)` per la CV.

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
