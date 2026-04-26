# Phase 2 — Residual Stream Capture (Spec)

## Status: 🔲 Next to implement

---

## Goal

Re-run every sample from `phase1/outputs/labeled_dataset.jsonl` through
Qwen3.5-9B-4bit, capture the hidden state from the final transformer block
at the end of the prefill pass, and write the resulting activation matrix
to disk alongside the binary labels.

The output of Phase 2 is the training data for the binary classifier in Phase 3.

---

## The hook (already implemented)

`runner.py::TransformersRunner.generate_with_hidden_state()` already has the
full implementation. It:

1. Registers a `register_forward_hook` on `self.model.model.layers[-1]`
2. Runs `model.generate()` — the hook fires during prefill
3. Removes the hook
4. Returns `(raw_output_string, hidden_state_tensor)`

The returned tensor has shape `[1, seq_len, hidden_size]` where
`hidden_size = 4096` for Qwen3.5-9B.

No changes needed to `runner.py` for Phase 2.

---

## What to build: `phase2/capture.py`

```
phase2/
  capture.py      ← main script: reads JSONL, runs hooks, writes .npy
  dataset.py      ← PyTorch Dataset wrapping .npy files
  train.py        ← classifier training (Phase 3)
```

### `capture.py` logic

```python
# Pseudocode for capture.py

runner = TransformersRunner(config)
runner.load()

for split in ["train", "val", "test"]:
    records = load_jsonl(f"phase1/outputs/splits/{split}.jsonl")
    
    X = []   # will become np.ndarray (N, 4096) float16
    y = []   # will become np.ndarray (N,) int8
    meta = []
    
    for record in records:
        messages = build_prompt_from_record(record)
        
        raw_output, hidden = runner.generate_with_hidden_state(messages)
        
        # Pool: last token of the prefill sequence
        # hidden shape: [1, seq_len, 4096]
        vec = hidden[0, -1, :].to(torch.float16).numpy()   # shape (4096,)
        
        X.append(vec)
        y.append(record["label"])
        meta.append({
            "id": record["id"],
            "category": record["category"],
            "hallucination_type": record["hallucination_type"],
        })
    
    out_dir = Path(f"phase2/outputs/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "X.npy", np.stack(X))    # (N, 4096) float16
    np.save(out_dir / "y.npy", np.array(y, dtype=np.int8))
    write_jsonl(out_dir / "meta.jsonl", meta)
```

### Important: use `generate_with_hidden_state`, not `generate`

Phase 2 needs to capture activations during the **same forward pass**
as inference. Do not run inference separately and then try to replay the
prompt — you will get slightly different hidden states due to KV cache
differences.

### Checkpoint / resume

Phase 2 may take hours. `capture.py` should checkpoint progress:
- Write activations in batches of N (e.g. 100)
- Track processed IDs in a `progress.json` file
- On restart, skip already-processed IDs

---

## What to build: `phase2/dataset.py`

```python
class ActivationDataset(torch.utils.data.Dataset):
    """
    Memory-maps .npy files so the full matrix is never loaded into RAM.
    Supports float32 upcasting from the float16 storage.
    """
    def __init__(self, split_dir: str, dtype=torch.float32):
        self.X = np.load(f"{split_dir}/X.npy", mmap_mode="r")  # (N, 4096)
        self.y = np.load(f"{split_dir}/y.npy", mmap_mode="r")  # (N,)
        self.dtype = dtype

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=self.dtype)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, label
```

---

## Storage layout

```
phase2/
  outputs/
    train/
      X.npy            float16, shape (N_train, 4096)
      y.npy            int8,    shape (N_train,)
      meta.jsonl       one JSON per line: id, category, hallucination_type
      progress.json    checkpoint: set of processed IDs
    val/
      X.npy
      y.npy
      meta.jsonl
    test/
      X.npy
      y.npy
      meta.jsonl
```

Expected sizes (for 2000 total samples, 70/15/15 split, per pooling strategy):
- Train X.npy: ~1400 × 4096 × 2 bytes ≈ **11.5 MB**
- Val X.npy:   ~300 × 4096 × 2 bytes  ≈ **2.5 MB**
- Test X.npy:  ~300 × 4096 × 2 bytes  ≈ **2.5 MB**

This is small enough to keep in git-lfs or just in the repo.

---

## Ablation: pooling strategies

Two strategies implemented in `capture.py`, stored in separate subdirectories:

| Strategy | Code | Notes |
|---|---|---|
| Last token | `hidden[0, -1, :]` | Primary; most "decisive" position |
| Mean of last 20 tokens | `hidden[0, -20:, :].mean(dim=0)` | Ablation; smooths over final context window |

```
outputs/train/last_token/X.npy
outputs/train/mean_last20/X.npy
```

---

## GPU memory note

Running `generate_with_hidden_state()` with bitsandbytes NF4:
- Qwen3.5-9B-4bit: ~5-6 GB model weights
- Hidden state per sample: `seq_len × 4096 × 2` bytes ≈ negligible
- Safe on 12 GB VRAM with batch_size=1

If VRAM is tight, clear CUDA cache between samples:
```python
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```

---

## CLI for capture.py

```bash
python phase2/capture.py \
    --splits_dir  phase1/outputs/splits \
    --output_dir  phase2/outputs \
    --model       Qwen/Qwen3.5-9B \
    --pooling     last_token          # last_token | mean_last20
    --batch_size  1
```
