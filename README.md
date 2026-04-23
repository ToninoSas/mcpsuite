# BFCL Hallucination Probe — Fase 1

Pipeline completa per costruire un dataset labellato di allucinazioni
su tool call, partendo dal Berkeley Function-Calling Leaderboard (BFCL).

---

## Requisiti di sistema

| Componente | Minimo | Raccomandato |
|---|---|---|
| Python | 3.10 | 3.11 / 3.12 |
| GPU VRAM | 12 GB | 24 GB |
| RAM | 16 GB | 32 GB |
| Disco | 20 GB | 40 GB |

> Se hai meno di 16 GB di VRAM usa il backend `llama_cpp` con un GGUF Q4_K_M
> anziché `transformers`. Vedi la sezione "Backend alternativo" in fondo.

---

## Step 1 — Ambiente Python

```bash
# Crea un venv (o conda)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

Se hai una GPU NVIDIA e vuoi bitsandbytes con CUDA:
```bash
# Controlla la versione CUDA installata
nvcc --version

# Installa il wheel corretto (es. CUDA 12.1)
pip install bitsandbytes --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2 — Scarica il dataset BFCL

```bash
# Assicurati di essere autenticato (serve un account HuggingFace gratuito)
huggingface-cli login

# Scarica tutto il dataset nella cartella data/
huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset \
    --local-dir ./data
```

Dopo il download la struttura sarà:
```
data/
  BFCL_v3_simple.json
  BFCL_v3_multiple.json
  BFCL_v3_parallel.json
  BFCL_v3_parallel_multiple.json
  BFCL_v3_multi_turn_base.json
  BFCL_v3_multi_turn_miss_func.json
  BFCL_v3_multi_turn_miss_param.json
  BFCL_v3_multi_turn_long_context.json
  BFCL_v3_multi_turn_composite.json
  possible_answer/
    BFCL_v3_simple.json
    ...
```

---

## Step 3 — Verifica i test unitari

```bash
python -m pytest test_evaluator.py -v
```

Output atteso: `25 passed in 0.Xs`

---

## Step 4 — Avvia la pipeline completa

```bash
python pipeline.py \
    --data_dir  ./data \
    --output    ./outputs/labeled_dataset.jsonl \
    --total     2000 \
    --model     Qwen/Qwen3.5-9B \
    --backend   transformers \
    --seed      42
```

### Parametri principali

| Parametro | Default | Descrizione |
|---|---|---|
| `--data_dir` | `./data` | Cartella con i file BFCL scaricati |
| `--output` | `./outputs/labeled_dataset.jsonl` | File JSONL di output |
| `--total` | `2000` | Numero totale di sample da processare |
| `--model` | `Qwen/Qwen3.5-9B` | Modello HuggingFace o path locale |
| `--backend` | `transformers` | `transformers` oppure `llama_cpp` |
| `--max_new_tokens` | `512` | Token massimi generati per risposta |
| `--seed` | `42` | Seme per riproducibilità |
| `--skip_inference` | off | Salta l'inferenza (ri-valuta output esistenti) |

### Esempio con budget ridotto (test rapido)

```bash
python pipeline.py \
    --data_dir ./data \
    --output   ./outputs/test_run.jsonl \
    --total    100 \
    --model    Qwen/Qwen3.5-9B
```

---

## Output

```
outputs/
  labeled_dataset.jsonl       ← dataset completo
  splits/
    train.jsonl               ← 70% stratificato per categoria
    val.jsonl                 ← 15%
    test.jsonl                ← 15%
```

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
| `WRONG_CALL_COUNT` | Numero di chiamate errato (categorie parallel/multiple) |
| `INFERENCE_ERROR` | Errore durante l'inferenza |

---

## Re-valutazione senza ri-eseguire l'inferenza

Se hai già un `labeled_dataset.jsonl` con gli output del modello e vuoi
solo modificare la logica di valutazione:

```bash
python pipeline.py \
    --data_dir     ./data \
    --output       ./outputs/labeled_dataset.jsonl \
    --skip_inference
```

Il pipeline leggerà i `model_raw_output` già salvati e ricalcolerà i label.

---

## Backend alternativo — llama-cpp (GGUF)

Per GPU con meno VRAM o CPU-only:

```bash
# Installa llama-cpp-python con supporto CUDA (se disponibile)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# Scarica il GGUF di Qwen/Qwen3.5-9B Q4_K_M (~4.4 GB)
hf download Qwen/Qwen3.5-9B-GGUF \
    qwen3.5-9B-q4_k_m.gguf \
    --local-dir ./models

# Avvia la pipeline con backend llama_cpp
python pipeline.py \
    --backend  llama_cpp \
    --model    ./models/qwen3.5-9B-q4_k_m.gguf \
    --total    2000
```

> Nota: con llama.cpp gli hidden states non sono accessibili direttamente.
> Per la Fase 2 (cattura del residual stream) è necessario il backend
> `transformers`.

---

## Struttura del progetto

```
phase1/
  loader.py          ← carica e correla domande + ground truth per ID
  sampler.py         ← campionamento proporzionale tra categorie
  evaluator.py       ← valutazione deterministica AST (no LLM-judge)
  runner.py          ← inferenza Qwen 4-bit + forward hook (Fase 2)
  pipeline.py        ← orchestratore principale (CLI)
  test_evaluator.py  ← 25 test unitari sul valutatore
  requirements.txt
  data/              ← dataset BFCL (da scaricare)
  outputs/           ← generato automaticamente
```
