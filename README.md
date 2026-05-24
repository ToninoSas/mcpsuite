# BFCL Hallucination Probe

Pipeline completa per predire — prima che venga emessa — se Qwen3.5-9B allucinera
durante una tool call, intercettando il residual stream di tutti i 32 layer del
transformer durante il prefill.

---

## Stato del progetto

| Fase | Stato | Output |
|---|---|---|
| Phase 1 — Dataset & pipeline di inferenza | ✅ Completa | `outputs/*/labeled_dataset.jsonl` |
| Phase 2 — Cattura residual stream | ✅ Integrata nella Phase 1 | `outputs/*/activations/` |
| Phase 3 — Addestramento classificatori per layer | ✅ Completa | `classifiers_*/metrics.json` |
| Phase 4 — Valutazione sul test set | ✅ Completa | `eval_results*/results.json` |

### Risultati principali

| Esperimento | Test set | Best layer | AUROC | F1 |
|---|---|---|---|---|
| Single-turn (1494 campioni) | Single-turn | 23 | **0.831** | 0.463 |
| Mixed (single + multi-turn) | Mixed | 23 | 0.961* | 0.907 |
| Mixed classifier | Single-turn | 23 | **0.827** | 0.500 |

*AUROC 0.96 è inflazionato da un confound strutturale (single vs multi-turn). Il
segnale genuino di allucinazione è **AUROC ~0.83**, stabile al layer 23.

---

## Requisiti di sistema

| Componente | Minimo | Raccomandato |
|---|---|---|
| Python | 3.10 | 3.11 / 3.12 |
| GPU VRAM | 16 GB | 2 × 16 GB (Kaggle T4) |
| RAM | 16 GB | 32 GB |
| Disco | 25 GB | 50 GB |

---

## Step 1 — Ambiente Python

```bash
python -m venv .venv
source .venv/bin/activate

cd phase1e2
pip install -r requirements.txt
```

---

## Step 2 — Scarica il dataset BFCL

```bash
huggingface-cli login

huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset \
    --local-dir ./data
```

---

## Step 3 — Verifica i test unitari

```bash
cd phase1e2
python -m pytest test_evaluator.py -v
# Output atteso: 25 passed
```

---

## Step 4 — Inferenza e cattura attivazioni

Sono disponibili due modalità di campionamento, mutuamente esclusive:

- **`--counts` (esatta, raccomandata)** — specifica direttamente quanti sample prelevare da ogni categoria.
- **`--total` + `--weights` (proporzionale, legacy)** — distribuisce un budget totale in base a pesi normalizzati.

### Single-turn — modalità esatta (raccomandata)

```bash
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn/labeled_dataset.jsonl \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --counts '{"simple":400,"multiple":200,"parallel":200,"parallel_multiple":200}'
```

### Live tasks — modalità esatta (raccomandata)

```bash
# num_gpus=1 obbligatorio: le live tasks hanno prompt lunghi che causano OOM su dual-GPU
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn2/labeled_dataset.jsonl \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --capture_activations \
    --counts '{"live_multiple":350,"live_parallel":75,"live_parallel_multiple":75}'
```

### Modalità proporzionale (legacy, backward-compatible)

```bash
# Single-turn
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn/labeled_dataset.jsonl \
    --total    1000 \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --weights '{"simple":0.40,"multiple":0.20,"parallel":0.20,"parallel_multiple":0.20}'

# Live tasks
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn2/labeled_dataset.jsonl \
    --total    500 \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --capture_activations \
    --weights '{"live_multiple":0.70,"live_parallel":0.15,"live_parallel_multiple":0.15}'
```

> **Nota**: `--counts` e `--total`/`--weights` sono mutuamente esclusivi. La modalità esatta
> è raccomandata per nuovi esperimenti perché elimina ambiguità sul numero finale di sample
> per categoria. Per riprodurre un esperimento vecchio con `--counts`, leggi i conteggi
> effettivi dal report di `proportional_sample` e passali con lo stesso `--seed`.

---

## Step 5 — Merge attivazioni e metriche

```bash
cd phase3 && python merge_activations.py \
    --src_a  ../outputs/single_turn/activations \
    --src_b  ../outputs/single_turn2/activations \
    --out    ../outputs/single_turn_merged/activations \
    --metrics_a   ../outputs/single_turn/metrics.json \
    --metrics_b   ../outputs/single_turn2/metrics.json \
    --metrics_out ../outputs/single_turn_merged/metrics.json
```

---

## Step 6 — Training classificatori (Phase 3)

```bash
cd phase3 && python train.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --out_dir         classifiers_single \
    --k_folds         5 \
    --device          cuda
```

Output: `classifiers_single/layer_XX.pt` + `metrics.json` + `metrics_per_layer.png`

---

## Step 7 — Valutazione sul test set (Phase 4)

```bash
# Layer singolo (best layer)
cd phase4 && python eval.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --classifiers_dir ../phase3/classifiers_single \
    --out_dir         eval_results_single \
    --layer           23

# Tutti i layer (produce anche summary.png)
cd phase4 && python eval.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --classifiers_dir ../phase3/classifiers_single \
    --out_dir         eval_results_single
```

Output per layer: AUROC con 95% CI bootstrap, F1, Recall, Precision alla soglia
ottimale di Youden, breakdown per categoria, grafici ROC + PR + distribuzione score.

---

## Struttura del progetto

```
mcpsuite/
├── CLAUDE.md              ← contesto completo per Claude Code
├── README.md              ← questo file
├── docs/
│   ├── roadmap.md         ← roadmap dettagliata 4 fasi + risultati
│   ├── phase1_complete.md ← note implementazione Phase 1
│   ├── phase2_spec.md     ← spec Phase 2 (integrata in Phase 1)
│   └── data_schema.md     ← schema JSONL e layout dataset BFCL
├── phase1e2/
│   ├── loader.py          ← carica e correla domande + ground truth per ID
│   ├── sampler.py         ← campionamento proporzionale (`proportional_sample`) o per conteggi esatti (`exact_sample`)
│   ├── evaluator.py       ← valutazione deterministica AST (no LLM-judge)
│   ├── runner.py          ← inferenza Qwen 4-bit + 32 forward hook
│   ├── pipeline.py        ← orchestratore CLI
│   └── test_evaluator.py  ← 25 test unitari
├── phase3/
│   ├── dataset.py             ← ActivationDataset: memory-map X.npy
│   ├── merge_activations.py   ← merge attivazioni + metrics.json da due sorgenti
│   ├── train.py               ← 32 MLP con 5-fold CV stratificata + plot
│   └── plot_comparison.py     ← confronto visivo tra due esperimenti
├── phase4/
│   └── eval.py                ← valutazione test set con bootstrap CI e soglia Youden
└── outputs/
    ├── single_turn/           ← 1000 campioni standard single-turn
    ├── single_turn2/          ← 494 campioni live tasks
    ├── single_turn_merged/    ← merge dei due (1494 campioni, 11.7% halluc)
    └── multi_turn/            ← 600 campioni multi-turn (~99% halluc)
```
