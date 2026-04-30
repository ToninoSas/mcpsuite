# Phase 1 — Implementation Notes

## Status: ✅ Complete (25/25 tests passing)

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

Proportional sampling with configurable per-category weights defined in
`DEFAULT_WEIGHTS`. Uses `random.sample` with a fixed seed for reproducibility.
Stratified train/val/test split (70/15/15) preserves category distribution
within each split.

### `evaluator.py`

The heart of Phase 1. Fully deterministic, no external API calls.

**Parser pipeline** (tries in order, returns on first success):
1. Qwen native `<tool_call>{...}</tool_call>` tag
2. JSON fenced block (` ```json ``` `)
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
training** — they carry no meaningful residual stream signal.

### `runner.py`

Wraps HuggingFace `transformers` + `bitsandbytes` (NF4 4-bit).

**`RunnerConfig`** key fields:
```python
max_seq_len: int | None = None        # truncate input to last N tokens (3072 on T4 16GB)
attn_implementation: str | None = "sdpa"  # PyTorch SDPA for memory-efficient attention
```

**`generate_with_hidden_state()`** — captures the last token of the prefill
from every transformer layer simultaneously:

```python
def generate_with_hidden_state(self, messages) -> tuple[str, np.ndarray | None]:
    # Registers one forward hook per layer (32 total for Qwen3.5-9B)
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
2. `proportional_sample()` — sample with configurable category weights
3. `run_inference_on_samples()` / `run_inference_parallel()` — Qwen inference,
   optionally with activation capture
4. `evaluate()` per sample — assign labels + hallucination type
5. Write `labeled_dataset.jsonl` + `splits/`
6. If `--capture_activations`: write `activations/{split}/X.npy` + `y.npy` +
   `meta.jsonl` + `shape.json`

**New CLI flags**:
```
--capture_activations   capture hidden states from all layers during inference
--max_seq_len N         truncate input sequences to last N tokens (use 3072 on T4)
--num_gpus N            data-parallel inference across N GPUs
--weights '{...}'       per-category sampling weights as JSON string
--no_multi_turn         zero-weight all multi-turn categories (quick single-turn run)
```

**Metrics**: at the end of every run the pipeline prints and saves `metrics.json`:
- Overall accuracy and hallucination rate
- Per-category breakdown (N, accuracy, hallucination rate, top hallucination type)
- Wall-clock timing per phase (load, sample, inference, eval, save, activations)
- Inference throughput (samples/s)

---

## Activation output schema

```
outputs/activations/
  train/
    X.npy       float16, shape (N_train, 32, 4096)
    y.npy       int8,    shape (N_train,)
    meta.jsonl  one JSON per row: id, category, hallucination_type
    shape.json  {"X_shape": [N, 32, 4096], "num_layers": 32, "hidden_size": 4096, ...}
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
  meaningful hidden state. Filter them before classifier training.

- `execution_result_type: "real_time"` samples are evaluated with AST matching,
  which may not be accurate. Consider filtering these.

- With `max_seq_len=3072` on multi-turn, long conversation history is truncated
  from the left. The model loses early context but the most recent turns are kept.

- Qwen's tool-call format may vary between model versions. The parser handles
  4 formats; new variants fall back to `NO_CALL_MADE`.
