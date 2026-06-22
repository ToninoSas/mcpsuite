# Phase 1 — Implementation Notes

## Status: ✅ Complete (32/32 evaluator tests + 10/10 sampler tests passing)

The pipeline supports **two models** through the same code path:

- `Qwen/Qwen3.5-9B` — system-prompt-driven tool calling with `<tool_call>` tag
- `meta-llama/Meta-Llama-3.1-8B-Instruct` — native `apply_chat_template(tools=)`
  via the `--use_native_tools` flag

---

## What was built

### `loader.py`

Loads BFCL `.json` files and correlates questions with ground truth by `id`.

Two GT formats exist in the dataset:
- **AST format** (in `possible_answer/`): `{"id": "simple_0", "ground_truth": [{"func_name": {"param": [val1, val2]}}]}`
- **Exec format** (inline in question file): `{"id": "exec_simple_0", "ground_truth": ["func_name(param=val)"], "execution_result_type": ["exact_match"]}`

The loader handles both transparently. `_build_gt_index()` returns both
the GT dict and the `execution_result_type` per id.

`BFCLSample` carries an optional `hidden_vec` field (numpy array,
shape `(num_layers, hidden_size)` float16) populated during inference
when `capture_activations=True`.

### `sampler.py`

Due modalità di campionamento, mutuamente esclusive:

- `proportional_sample(samples, total, weights, seed)` — *legacy*: distribuisce
  un budget totale `total` in base ai pesi per categoria. Fix di compatibilità
  garantito.
- `exact_sample(samples, counts, seed)` — *raccomandata*: prende esattamente
  `counts[category]` campioni per ciascuna categoria specificata. Elimina
  ambiguità nel report finale di campioni effettivi per categoria.

Entrambe usano `random.sample` con seed fisso per riproducibilità. Lo split
stratificato train/val/test (70/15/15) preserva la distribuzione di categoria
all'interno di ogni split.

### `evaluator.py`

The heart of Phase 1. Fully deterministic, no external API calls.

**Parser pipeline** (tries in order, returns on first success):
1. Qwen native `<tool_call>{...}</tool_call>` tag
1b. **Llama-style `<function_name>{json}</function_name>`** (regex relax che
    accetta nomi puntati come `game_rewards.get` — XML non valido ma comune nel
    fallback Llama). Gestisce tre sotto-casi:
   - JSON con `name` + `arguments`
   - JSON con `name` solo, tutti gli altri campi diventano arguments
   - JSON senza `name`, il tag stesso diventa il nome funzione
2. JSON fenced block (` ```json ``` `)
2b. **Llama parallel format** `{...}; {...}; ...` — oggetti JSON top-level
    separati da `;` (formato nativo di Llama per le parallel call). Estrazione
    string-aware di tutti i top-level `{...}` (`_extract_top_level_braces`).
    Senza questo, lo step 3 estraeva solo il primo oggetto → `WRONG_CALL_COUNT`
    sistematico sulle categorie parallel.
3. JSON inline (finds first `[` or `{` and extracts balanced structure)
4. Python call-string with balanced-parenthesis extractor

**Critical bug fixed**: `re.sub(r"^.*?(?=\w+\s*\()", ...)` exhibited a
Python regex engine quirk where zero-length matches at position 0 caused
`re.finditer` to return a second match at position 1, stripping the first
character of the function name. Fixed by removing the `re.sub` entirely and
using a `re.finditer` on `\b([A-Za-z_]\w*(?:\.\w+)*)\s*\(` with explicit
balanced-paren walking.

**Matching rules**:
- Function name: case-insensitive comparison
- Arg values: type coercion (`"20"==20`, `"true"==True`, hyphen/underscore
  normalized), list-of-acceptable-values semantics from BFCL GT format
- Optional params: GT encodes optional by including `""` or `None` in the
  acceptable values list
- Parallel/multiple: bipartite greedy matching — each predicted call must
  match exactly one GT entry; count mismatch detected first

**Hallucination taxonomy**:
```
label=0  →  hallucination_type=None
label=1  →  hallucination_type ∈ {
    NO_CALL_MADE       model produced no parseable tool call
    WRONG_FUNCTION     function name does not match any GT entry
    MISSING_ARGS       required argument absent from predicted call
    EXTRA_ARGS         predicted call has args not in function schema
    WRONG_ARG_VALUES   arg name correct but value outside acceptable set
    WRONG_ARG_NAMES    arg present but under wrong key name
    WRONG_CALL_COUNT   # predicted calls ≠ # GT calls (parallel/multiple)
    INFERENCE_ERROR    runner crashed during inference (OOM, CUDA error)
}
```

**Note**: `INFERENCE_ERROR` samples must be **filtered out before classifier
training** — they carry no meaningful residual stream signal. Already handled
by `ActivationDataset(filter_inference_errors=True)`.

**Multi-turn — aggregazione del label.** `evaluate_multi_turn()` valuta ogni
turno e aggrega in un label di sample. Il parametro `aggregation_threshold`
controlla la regola: `None` = any-turn (label=1 se almeno un turno fallisce),
`0.7` = strategia B2 (label=1 se `frac_failed ≥ 0.7`). Per i turni in cui il GT è
vuoto (scenari miss_func/miss_param dove non si deve chiamare), un no-call
corretto è `label=0`; una call non attesa è `WRONG_FUNCTION`.

### `reevaluate.py`

Ri-applica l'evaluator a un `labeled_dataset` esistente **senza rifare
l'inferenza**: rigenera `labeled_dataset.jsonl`, `splits/`, `metrics.json` e
`activations/{split}/y.npy`+`meta.jsonl` (X.npy resta invariato). Usato per
propagare fix del parser o la strategia multi-turn (`--multi_turn_threshold`)
ai dataset già prodotti. Esempio: il fix del formato parallel di Llama ha
recuperato ~218 falsi positivi senza un nuovo run GPU.

### `runner.py`

Wraps HuggingFace `transformers` + `bitsandbytes` (NF4 4-bit). Supporta
qualsiasi modello con architettura transformer decoder caricabile via
`AutoModelForCausalLM`.

**`MODEL_CONFIGS` registry** (chiave → metadata):

```python
MODEL_CONFIGS = {
    "qwen": {
        "model_id":         "Qwen/Qwen3.5-9B",
        "n_layers":         32, "hidden_size": 4096,
        "use_native_tools": False,   # system prompt custom con <tool_call>
    },
    "llama": {
        "model_id":         "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "n_layers":         32, "hidden_size": 4096,
        "use_native_tools": True,    # apply_chat_template(tools=...)
    },
}
```

**`RunnerConfig`** key fields:
```python
model_name_or_path: str  = "Qwen/Qwen3.5-9B"
max_seq_len:        int | None = None       # truncate input to last N tokens (3072 on T4 16GB)
attn_implementation: str | None = "sdpa"   # PyTorch SDPA for memory-efficient attention
use_native_tools:   bool  = False           # passa tools= al chat template nativo
```

**`_apply_template()`** — instrada tra il template nativo e quello Qwen-style:
- Se `tools is not None` → `apply_chat_template(messages, tools=tools, ...)`
- Altrimenti → `apply_chat_template(messages, enable_thinking=False, ...)`
  con fallback `TypeError` per modelli che non supportano `enable_thinking`.

**`build_prompt(sample, use_native_tools)`** — costruisce la lista messaggi:
- `use_native_tools=False`: prepende il system message con gli schemi funzione
- `use_native_tools=True`: omette il system message (sarà generato dal template
  nativo a partire dalle tools); skippa anche eventuali system message
  *interni* ai turni della domanda (categorie live)

**`generate_with_hidden_state()`** — captures the last token of the prefill
from every transformer layer simultaneously:

```python
def generate_with_hidden_state(self, messages, tools=None) -> tuple[str, np.ndarray | None]:
    # Registers one forward hook per layer (32 total for Qwen3.5-9B / Llama-3.1-8B)
    # Each hook fires during prefill (seq_len > 1), captures hidden[0, -1, :]
    # and immediately moves it to CPU float16 (VRAM not accumulated)
    # Returns (raw_output_string, activations)
    # activations: numpy float16, shape (num_layers, hidden_size) = (32, 4096)
    #              None if any hook failed to fire
```

Key design points:
- Hooks detect prefill vs decode by checking `seq_len > 1`
- Each captured tensor is moved to CPU immediately → only ~262 KB per sample in RAM
- `attn_implementation="sdpa"` reduces attention memory from O(n²) to O(n)
- `max_seq_len` truncates from the left (keeps most recent context) before tokenization
- Quando `use_native_tools=True`, `tools=sample.functions` viene passato al
  template; altrimenti il system message custom è già nel prompt

**Dual-GPU parallelism** (`run_inference_parallel`):
- Models are loaded **sequentially** (bitsandbytes is not thread-safe)
- `torch.cuda.synchronize()` between loads prevents CUDA context race conditions
- All samples (single-turn and multi-turn) are divided equally across GPUs
- Requires `max_seq_len` to be set when multi-turn samples are present, otherwise
  OOM on long sequences corrupts the CUDA context (`cudaErrorIllegalAddress` cascade)

**OOM recovery** in `run_inference_on_samples`:
- `torch.cuda.OutOfMemoryError` is caught separately from generic exceptions
- `torch.cuda.synchronize()` is called before `empty_cache()` to wait for
  pending kernels and prevent cascading illegal memory access on next sample

### `pipeline.py`

CLI orchestrator. Runs steps 1-6 in sequence:
1. `load_all()` — load corpus
2. `exact_sample()` / `proportional_sample()` — sample con conteggi esatti o pesi
3. `run_inference_on_samples()` / `run_inference_parallel()` — inferenza,
   optionally with activation capture
4. `evaluate()` per sample — assign labels + hallucination type
5. Write `labeled_dataset.jsonl` + `splits/`
6. If `--capture_activations`: write `activations/{split}/X.npy` + `y.npy` +
   `meta.jsonl` + `shape.json`

**CLI flags principali**:
```
--model PATH            model id HuggingFace o path locale (default Qwen/Qwen3.5-9B)
--use_native_tools      attiva il template nativo (richiesto per Llama-3.1)
--capture_activations   capture hidden states from all layers during inference
--max_seq_len N         truncate input sequences to last N tokens (use 3072 on T4)
--num_gpus N            data-parallel inference across N GPUs
--counts JSON           conteggi esatti per categoria (mutuamente esclusivo con --total/--weights)
--total N               numero totale di sample
--weights JSON          pesi per categoria
--no_multi_turn         zero-weight all multi-turn categories (quick single-turn run)
--checkpoint_dir DIR    abilita ripresa automatica da checkpoint
--seed N                seed di campionamento e split (default 42)
```

**Metrics**: at the end of every run the pipeline prints and saves `metrics.json`:
- Campo `"model"`: model id o path usato (tracciabilità multi-modello)
- Overall accuracy and hallucination rate
- Per-category breakdown (N, accuracy, hallucination rate, top hallucination type)
- Wall-clock timing per phase (load, sample, inference, eval, save, activations)
- Inference throughput (samples/s)

`shape.json` in `activations/{split}/` include anch'esso il campo `model` per
permettere a Phase 3/Phase 4 di tracciare la provenienza dei dati.

---

## Activation output schema

```
outputs/activations/
  train/
    X.npy       float16, shape (N_train, 32, 4096)
    y.npy       int8,    shape (N_train,)
    meta.jsonl  one JSON per row: id, category, hallucination_type
    shape.json  {"X_shape": [N, 32, 4096], "num_layers": 32, "hidden_size": 4096, "model": "...", ...}
  val/   (same structure)
  test/  (same structure)
```

`X[i, j, :]` is the last-token hidden state of layer `j` during the prefill
of sample `i`. Shape `(32, 4096)` per sample.

---

## Known limitations

- Multi-turn evaluation applies single-turn AST logic per turn. State-based
  evaluation (comparing backend state after execution) is not implemented —
  only response-based matching. This may inflate false negatives on multi-turn.

- `INFERENCE_ERROR` samples (OOM, CUDA crash) have `label=1` but carry no
  meaningful hidden state. Filter them before classifier training
  (automatico via `ActivationDataset`).

- `execution_result_type: "real_time"` samples are evaluated with AST matching,
  which may not be accurate. Consider filtering these.

- With `max_seq_len=3072` on multi-turn, long conversation history is truncated
  from the left. The model loses early context but the most recent turns are kept.

- Tool-call format varies per modello. Il parser gestisce 4 formati (Qwen
  `<tool_call>`, Llama `<function_name>`, JSON fenced/inline, Python call-string);
  formati nuovi non riconosciuti producono `NO_CALL_MADE`. Per Llama-3.1
  è **obbligatorio** usare `--use_native_tools` per evitare format mismatch
  sistematici.
