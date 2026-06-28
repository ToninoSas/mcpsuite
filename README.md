# Prevedere le allucinazioni nel tool calling di un agente LLM tramite l’analisi del residual stream
## Progetto di tesi triennale in Informatica

### Abstract
Abbiamo addestrato e valutato dei classificatori leggeri (MLP) sullo stato interno di Qwen3.5-9B e LLama-3.1-8B-Instruct mentre facevano inferenza su tasks di tool calling, da benchmark BFCL. Lo scopo era capire se le allucinazioni di tool calling erano possibili da prevedere. Si è dimostrato che la codifica dell’incertezza, all’interno del modello, avviene nei layer intermedi ed è quindi possibile prevedere queste allucinazioni con un sistema di protezione (guard rail) specializzato.

### Cosa rappresenta questa repo
Pipeline completa per prevedere le allucinazioni di un Agente LLM durante una tool call,
intercettando il residual stream di tutti i layer del transformer durante il prefill.

Il progetto è **multi-modello**: supporta Qwen3.5-9B e Llama-3.1-8B-Instruct
attraverso lo stesso pipeline. Abbiamo preso due modelli per verificare che i risultati non dipendano da un singolo modello.

---

## Stato del progetto

| Fase | Stato | Output |
|---|---|---|
| Phase 1 — Dataset & pipeline di inferenza (multi-modello) | ✅ Completa | `outputs/*/labeled_dataset.jsonl` |
| Phase 2 — Cattura residual stream | ✅ Integrata nella Phase 1 | `outputs/*/activations/` |
| Phase 3 — Addestramento classificatori per layer | ✅ Completa | `classifiers_*/metrics.json` |
| Phase 4 — Valutazione cross-distribution | ✅ Completa | `eval_results*/results.json` + heatmap/bar/grid |
| Estensione multi-modello (Qwen + Llama) | ✅ Completa | matrice 2×2 + colonna merged |

### Risultati principali — matrice 2×2 (AUROC del best layer)

| Train → Test | Tipo | Qwen (layer) | Llama (layer) |
|---|---|---|---|
| single → single | in-distribution | **0.82** (19) | **0.86** (15) |
| multi → multi | in-distribution | **0.93** (13) | **0.77** (20) |
| single → multi | transfer | 0.61 (10) | 0.61 (0) |
| multi → single | transfer | 0.72 (13) | 0.67 (2) |

**Finding principale (cross-model)**: il segnale di allucinazione nel residual
stream è **rilevabile ma specifico della distribuzione**. È forte in-distribution
(AUROC 0.82–0.93, a profondità intermedia) ma non generalizza tra regimi (transfer
0.61–0.72). Lo stesso pattern su entrambi i modelli → proprietà generale.
Implicazione: un guardrail efficace dev'essere specializzato per regime.

Le valutazioni su test **merged** danno AUROC alto (fino a 0.94) ma parzialmente
inflazionato da un confound strutturale (il test mescola due regimi a
distribuzione di classe opposta); la misura non contaminata è la matrice 2×2.

---

## Modelli supportati

| Key | Model ID | Flag CLI richiesti |
|---|---|---|
| Qwen  | `Qwen/Qwen3.5-9B`                       | `--model Qwen/Qwen3.5-9B` |
| Llama | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `--model meta-llama/Meta-Llama-3.1-8B-Instruct --use_native_tools` |

Il registry `MODEL_CONFIGS` in [phase1e2/runner.py](phase1e2/runner.py)
specifica per ogni modello se usare il template nativo (Llama) o il system
prompt custom (Qwen). Phase 3 e Phase 4 sono **model-agnostic**: leggono
direttamente `X.npy`/`y.npy` indipendentemente da quale modello li ha prodotti.

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
python -m pytest test_evaluator.py test_sampler.py -v
# Output atteso: 28 + 10 = 38 passed
```

I 28 test di `test_evaluator.py` includono 3 casi specifici per il formato
`<function_name>{json}</function_name>` di Llama (con nomi puntati tipo
`game_rewards.get`).

---

## Step 4 — Inferenza e cattura attivazioni

Sono disponibili due modalità di campionamento, mutuamente esclusive:

- **`--counts` (esatta, raccomandata)** — specifica direttamente quanti sample
  prelevare da ogni categoria.
- **`--total` + `--weights` (proporzionale, legacy)** — distribuisce un budget
  totale in base a pesi normalizzati.

### Qwen — single-turn (modalità esatta)

```bash
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn/labeled_dataset.jsonl \
    --model    Qwen/Qwen3.5-9B \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --counts '{"simple":400,"multiple":200,"parallel":200,"parallel_multiple":200}'
```

### Qwen — live tasks

```bash
# num_gpus=1 obbligatorio: le live tasks hanno prompt lunghi (OOM su dual-GPU)
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn2/labeled_dataset.jsonl \
    --model    Qwen/Qwen3.5-9B \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --capture_activations \
    --counts '{"live_multiple":350,"live_parallel":75,"live_parallel_multiple":75}'
```

### Llama-3.1 — single-turn (richiede `--use_native_tools`)

```bash
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/llama/standard/labeled_dataset.jsonl \
    --model    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --use_native_tools \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --counts '{"simple":400,"multiple":200,"parallel":200,"parallel_multiple":200}'
```

### Llama-3.1 — live tasks

```bash
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/llama/live/labeled_dataset.jsonl \
    --model    meta-llama/Meta-Llama-3.1-8B-Instruct \
    --use_native_tools \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --capture_activations \
    --counts '{"live_multiple":350,"live_parallel":75,"live_parallel_multiple":75}'
```

> **Importante per Llama**: senza `--use_native_tools` il modello riceve il
> system prompt custom (pensato per Qwen) e improvvisa il formato di output,
> generando label spurie da fallimento di parsing. Il flag attiva
> `apply_chat_template(tools=sample.functions)` che produce il prompt nel
> formato su cui Llama è stato addestrato.

### Modalità proporzionale (legacy, backward-compatible)

```bash
cd phase1e2 && python pipeline.py \
    --data_dir ./data \
    --output   ../outputs/single_turn/labeled_dataset.jsonl \
    --model    Qwen/Qwen3.5-9B \
    --total    1000 \
    --num_gpus 2 \
    --max_seq_len 3072 \
    --capture_activations \
    --weights '{"simple":0.40,"multiple":0.20,"parallel":0.20,"parallel_multiple":0.20}'
```

> **Nota**: `--counts` e `--total`/`--weights` sono mutuamente esclusivi. La
> modalità esatta è raccomandata per nuovi esperimenti perché elimina
> ambiguità sul numero finale di sample per categoria.

---

## Step 5 — Merge attivazioni e metriche

Il merge è **model-agnostic**: combina due qualunque sorgenti di attivazioni
con la stessa struttura di file.

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

Il training è **deterministico**: stesso `--seed` ⟹ stessi pesi. Internamente
ogni layer X riceve seed `seed + X`, quindi modello e ordine di esecuzione
sono indipendenti.

```bash
cd phase3 && python train.py \
    --activations_dir ../outputs/single_turn_merged/activations \
    --out_dir         classifiers_single \
    --k_folds         5 \
    --seed            42 \
    --device          cuda
```

Output:
- `classifiers_single/layer_XX.pt` (32 pesi MLP)
- `classifiers_single/metrics.json` — `{"seed": 42, "layers": [...]}`
- `classifiers_single/metrics_per_layer.png`

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
ottimale di Youden, breakdown per categoria, grafici ROC + PR + distribuzione
score.

---

## Step 8 — Confronto tra esperimenti (es. Qwen vs Llama)

```bash
cd phase3 && python plot_comparison.py \
    --metrics_a classifiers_qwen/metrics.json    --label_a "Qwen3.5-9B" \
    --metrics_b classifiers_llama/metrics.json   --label_b "Llama-3.1-8B" \
    --out       comparison_qwen_vs_llama.png
```

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
│   ├── sampler.py         ← `proportional_sample` o `exact_sample`
│   ├── evaluator.py       ← valutazione deterministica AST (no LLM-judge)
│   │                        + parser per il formato Llama <function_name>
│   ├── runner.py          ← inferenza 4-bit + 32 forward hook
│   │                        MODEL_CONFIGS (qwen, llama) + use_native_tools
│   ├── pipeline.py        ← orchestratore CLI (--model, --use_native_tools)
│   ├── test_evaluator.py  ← 28 test unitari
│   └── test_sampler.py    ← 10 test unitari
├── phase3/
│   ├── dataset.py             ← ActivationDataset: memory-map X.npy
│   ├── merge_activations.py   ← merge attivazioni + metrics.json
│   ├── train.py               ← 32 MLP con 5-fold CV + seed riproducibile
│   └── plot_comparison.py     ← confronto visivo tra due esperimenti
├── phase4/
│   └── eval.py                ← test set: bootstrap CI, soglia Youden, plot
└── outputs/
    ├── single_turn/           ← Qwen, 1000 campioni standard
    ├── single_turn2/          ← Qwen, 494 campioni live
    ├── single_turn_merged/    ← Qwen, merge dei due (1494, 11.7% halluc)
    ├── multi_turn/            ← Qwen, 600 campioni multi-turn (84.7% halluc, B2 0.7)
    └── llama/llama/           ← Llama, esperimenti multi-modello
        ├── standard/
        ├── live/
        ├── single_turn_merged/
        ├── multi-turn/
        └── merged_all/
```
