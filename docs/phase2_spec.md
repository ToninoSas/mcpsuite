# Phase 2 — Residual Stream Capture

## Status: ✅ Merged into Phase 1

---

## What changed from the original spec

Phase 2 (activation capture) is **no longer a separate script**.
It is fully integrated into `phase1/pipeline.py` via the
`--capture_activations` flag, which triggers step 1.6 of the pipeline.

There is no `capture.py` to write. Running the pipeline with
`--capture_activations` produces both the labeled dataset and the
activation files in a single pass.

---

## What was redesigned: all-layer capture

The original spec captured only the **last transformer layer**.
The current implementation captures the **last token of the prefill
from every layer** simultaneously.

**Why**: intermediate layers carry more localised semantic information.
By capturing all 32 layers we can run a probing study (AUROC vs layer index)
to find which layer is most discriminative for hallucination detection,
rather than assuming the last layer is always best.

**How** (`generate_with_hidden_state` in `runner.py`):
- Registers one `register_forward_hook` per layer (32 hooks total)
- Each hook fires when `seq_len > 1` (prefill), ignores `seq_len == 1` (decode)
- Captures `hidden[0, -1, :]` — last token — and moves it to CPU float16 immediately
- VRAM cost: ~0 (one tensor at a time; previous layer already freed)
- CPU RAM cost per sample: 32 × 4096 × 2 bytes = 262 KB

---

## Storage layout (produced by pipeline.py step 1.6)

```
outputs/activations/
  train/
    X.npy       float16, shape (N_train, 32, 4096)
    y.npy       int8,    shape (N_train,)
    meta.jsonl  one JSON per row: id, category, hallucination_type
    shape.json  {"X_shape": [N, 32, 4096], "num_layers": 32, "hidden_size": 4096, ...}
  val/
    X.npy / y.npy / meta.jsonl / shape.json
  test/
    X.npy / y.npy / meta.jsonl / shape.json
```

`X[i, j, :]` = last-token hidden state of layer `j` during prefill of sample `i`.

Expected sizes for 2000 samples (70/15/15 split):
- Train X.npy: 1400 × 32 × 4096 × 2 bytes ≈ **368 MB**
- Val X.npy:   300  × 32 × 4096 × 2 bytes ≈ **79 MB**
- Test X.npy:  300  × 32 × 4096 × 2 bytes ≈ **79 MB**

---

## How to run

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

`--max_seq_len 3072` is required on Kaggle T4 (16 GB VRAM) to prevent OOM
on long multi-turn sequences. Without it, OOM corrupts the CUDA context and
causes `cudaErrorIllegalAddress` on subsequent samples.

---

## What still needs to be built (for Phase 3)

```
phase2/
  dataset.py   ← PyTorch Dataset over X.npy / y.npy with layer selection
  train.py     ← 32 independent MLP classifiers, one per layer; AUROC plot
```

See `docs/roadmap.md` Phase 3 for the full classifier spec.

---

## Pre-training data cleaning

Before training classifiers, filter out `INFERENCE_ERROR` samples:

```python
import json, numpy as np

meta = [json.loads(l) for l in open("outputs/activations/train/meta.jsonl")]
valid = [i for i, m in enumerate(meta) if m["hallucination_type"] != "INFERENCE_ERROR"]

X = np.load("outputs/activations/train/X.npy", mmap_mode="r")[valid]
y = np.load("outputs/activations/train/y.npy",  mmap_mode="r")[valid]
```

`INFERENCE_ERROR` records have `label=1` but the model never ran —
the hidden state was never captured and they carry no signal.
