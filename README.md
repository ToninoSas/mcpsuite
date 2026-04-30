# BFCL Hallucination Probe

Pipeline completa per costruire un dataset labellato di allucinazioni su tool call
(Berkeley Function-Calling Leaderboard), catturare il residual stream di tutti i
layer del modello e addestrare classificatori per layer per predire le allucinazioni
prima che vengano emesse.

---

## Stato del progetto

| Fase | Stato | Output |
|---|---|---|
| Phase 1 — Dataset & pipeline di inferenza | ✅ Completa | `outputs/labeled_dataset.jsonl` |
| Phase 2 — Cattura residual stream | ✅ Integrata nella Phase 1 | `outputs/activations/` |
| Phase 3 — Addestramento classificatori per layer | 🔲 Prossima | `outputs/classifiers/` |
| Phase 4 — Valutazione & guardrail real-time | 🔲 In attesa | inference wrapper |

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
source .venv/bin/activate        # Windows: .venv\Scripts\activate

cd phase1
pip install -r requirements.txt
```

Se hai una GPU NVIDIA:
```bash
pip install bitsandbytes --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2 — Scarica il dataset BFCL

```bash
huggingface-cli login

huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset \
    --local-dir ./phase1/data
```

Struttura attesa:
```
phase1/data/
  BFCL_v3_simple.json
  BFCL_v3_multiple.json
  BFCL_v3_parallel.json
  BFCL_v3_parallel_multiple.json
  BFCL_v3_multi_turn_base.json
  BFCL_v3_multi_turn_miss_func.json
  BFCL_v3_multi_turn_miss_param.json
  BFCL_v3_multi_turn_composite.json
  possible_answer/
    BFCL_v3_simple.json
    ...
```

---

## Step 3 — Verifica i test unitari

```bash
cd phase1
python -m pytest test_evaluator.py -v
```

Output atteso: `25 passed`

---

## Step 4 — Avvia la pipeline completa

### Run raccomandato (Kaggle T4 × 2, ~2000 sample, cattura attivazioni)

```bash
cd phase1
python pipeline.py \
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
```

### Test rapido (50 sample, no attivazioni, singola GPU)

```bash
cd phase1
python pipeline.py --data_dir ./data --total 50
```

### Parametri principali

| Parametro | Default | Descrizione |
|---|---|---|
| `--data_dir` | `./data` | Cartella con i file BFCL |
| `--output` | `./outputs/labeled_dataset.jsonl` | File JSONL di output |
| `--total` | `2000` | Numero totale di sample |
| `--model` | `Qwen/Qwen3.5-9B` | Modello HuggingFace o path locale |
| `--num_gpus` | `1` | GPU da usare in parallelo |
| `--max_seq_len` | `None` | Tronca l'input agli ultimi N token (usa 3072 su T4 16 GB) |
| `--capture_activations` | off | Cattura gli hidden state di tutti i layer |
| `--weights '{...}'` | pesi default | Pesi di campionamento per categoria (JSON) |
| `--no_multi_turn` | off | Esclude tutte le categorie multi-turn |
| `--max_new_tokens` | `512` | Token massimi generati per risposta |
| `--seed` | `42` | Seme per riproducibilità |

---

## Output

```
phase1/outputs/
  labeled_dataset.jsonl       ← dataset completo labellato
  metrics.json                ← accuracy, hallucination rate, timing, per-category stats
  splits/
    train.jsonl               ← 70% stratificato per categoria
    val.jsonl                 ← 15%
    test.jsonl                ← 15%
  activations/                ← prodotto solo con --capture_activations
    train/
      X.npy                   ← float16, shape (N_train, 32, 4096)
      y.npy                   ← int8,    shape (N_train,)
      meta.jsonl              ← id, category, hallucination_type per riga
      shape.json              ← {"X_shape": [N, 32, 4096], "num_layers": 32, ...}
    val/   (stessa struttura)
    test/  (stessa struttura)
```

`X[i, j, :]` = hidden state dell'ultimo token del prefill, layer `j`, sample `i`.

### Struttura di ogni record JSONL

```jsonc
{
  "id":                    "simple_42",
  "category":              "simple",
  "question":              [[{"role": "user", "content": "..."}]],
  "functions":             [{"name": "...", "description": "...", "parameters": {...}}],
  "ground_truth":          [{"function_name": {"param": [val1, val2]}}],
  "execution_result_type": [],
  "model_raw_output":      "function_name(param=val)",
  "label":                 0,                  // 0=corretto, 1=allucinazione
  "hallucination_type":    null,               // null se label=0
  "eval_details":          { ... }             // diagnostica completa
}
```

### Valori di `hallucination_type`

| Tipo | Significato |
|---|---|
| `null` | Nessuna allucinazione (label=0) |
| `NO_CALL_MADE` | Il modello non ha prodotto nessuna tool call |
| `WRONG_FUNCTION` | Nome funzione sbagliato |
| `MISSING_ARGS` | Argomenti obbligatori mancanti |
| `EXTRA_ARGS` | Argomenti extra non previsti dallo schema |
| `WRONG_ARG_VALUES` | Argomenti giusti ma valori fuori dal set accettabile |
| `WRONG_ARG_NAMES` | Nomi degli argomenti errati |
| `WRONG_CALL_COUNT` | Numero di chiamate errato (parallel/multiple) |
| `INFERENCE_ERROR` | Errore durante l'inferenza (OOM, CUDA) — filtrare prima del training |

---

## Note tecniche

### Cattura del residual stream

Con `--capture_activations` la pipeline registra 32 forward hook (uno per layer)
durante il prefill. Ogni hook cattura `hidden[0, -1, :]` (ultimo token, float16)
e lo sposta subito su CPU. Costo VRAM: ~0; costo RAM: ~262 KB per sample.

### Doppia GPU

I modelli vengono caricati **sequenzialmente** (bitsandbytes NF4 non è thread-safe).
Solo l'inferenza gira in parallelo tra le GPU. Usa sempre `--max_seq_len 3072` con
multi-turn per evitare OOM che corrompono il contesto CUDA.

### Bilanciamento delle classi

Il dataset ha circa l'8% di allucinazioni. Durante il training (Phase 3) si usa
`pos_weight = n_neg / n_pos` in `BCELoss`. I sample `INFERENCE_ERROR` vanno
filtrati prima del training perché non hanno hidden state significativi.

---

## Struttura del progetto

```
mcpsuite/
├── CLAUDE.md              ← contesto completo per Claude Code
├── README.md              ← questo file
├── docs/
│   ├── roadmap.md         ← roadmap dettagliata 4 fasi
│   ├── phase1_complete.md ← note implementazione Phase 1
│   ├── phase2_spec.md     ← spec Phase 2 (integrata in Phase 1)
│   └── data_schema.md     ← schema JSONL e layout dataset BFCL
└── phase1/
    ├── loader.py          ← carica e correla domande + ground truth per ID
    ├── sampler.py         ← campionamento proporzionale tra categorie
    ├── evaluator.py       ← valutazione deterministica AST (no LLM-judge)
    ├── runner.py          ← inferenza Qwen 4-bit + 32 forward hook
    ├── pipeline.py        ← orchestratore CLI (fasi 1-6)
    ├── test_evaluator.py  ← 25 test unitari (tutti passano)
    ├── requirements.txt
    ├── data/              ← dataset BFCL (da scaricare)
    └── outputs/           ← generato automaticamente
```
