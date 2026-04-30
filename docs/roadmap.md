# Roadmap — LLM Agent Hallucination Probe

## Overview

The goal is to predict, *before* a tool call is emitted, whether a
Qwen3.5-9B agent is about to hallucinate. The prediction signal comes from
the model's own internal state — the residual stream of every transformer
block — not from post-hoc output analysis.

```
User query + function schemas
        │
        ▼
  Qwen3.5-9B-4bit
  [prefill phase]
        │
        ├──► forward hooks on ALL layers (0..31)
        │         │
        │         ▼
        │    hidden states [32, 4096]  (last token, each layer)
        │         │
        │    Binary Classifier per layer
        │    (Linear→BatchNorm→ReLU→Sigmoid)
        │         │
        │    AUROC per layer → plot → find best layer
        │         │
        │    Best classifier: P(hallucination)
        │         │
        │    ┌────┴─────┐
        │    │ p > θ    │ p ≤ θ
        │    ▼          ▼
        │  BLOCK     ALLOW
        │  / retry   tool call
        ▼
  tool call output
```

---

## Phase 1 — Test Suite, Dataset & Activation Capture  ✅ COMPLETE

**Goal**: Build a labeled dataset of (prompt, function_schema, model_output, label)
tuples from BFCL, and capture residual stream activations from all transformer
layers during the same inference pass.

**Source**: Berkeley Function-Calling Leaderboard v3
(`gorilla-llm/Berkeley-Function-Calling-Leaderboard`)

**Categories** (proportional sampling, recommended weights for full run):
- `simple` (20%)
- `multiple` (15%)
- `parallel` (10%)
- `parallel_multiple` (5%)
- `multi_turn_base` (15%)
- `multi_turn_miss_func` (15%) ← high hallucination rate by design
- `multi_turn_miss_param` (15%) ← high hallucination rate by design
- `multi_turn_long_context` (0%) ← excluded: too long even with truncation
- `multi_turn_composite` (5%)

**Labeling**: deterministic AST matching against BFCL ground truth. No LLM judge.

**Activation capture**: `--capture_activations` flag in `pipeline.py`.
Registers 32 hooks (one per transformer layer), captures `hidden[0, -1, :]`
(last token of prefill) per layer, saves as `X.npy` shape `(N, 32, 4096)`.

**Key constraints**:
- `--max_seq_len 3072` required on Kaggle T4 to prevent CUDA context corruption
- Filter `INFERENCE_ERROR` samples before classifier training
- Multi-turn included (except `long_context`) — higher hallucination rate

**Output**:
```
outputs/
  labeled_dataset.jsonl
  metrics.json
  splits/train.jsonl, val.jsonl, test.jsonl
  activations/
    train/  X.npy (N,32,4096)  y.npy  meta.jsonl  shape.json
    val/    ...
    test/   ...
```

---

## Phase 2 — Residual Stream Capture  ✅ Merged into Phase 1

No separate script. Fully handled by `pipeline.py --capture_activations`.
See `docs/phase2_spec.md` for details and the pre-training data cleaning step.

---

## Phase 3 — Per-Layer Classifier Training  🔲 NEXT

**Goal**: Train 32 independent binary classifiers (one per transformer layer)
on the captured activations. Plot AUROC vs layer index to find which layer
carries the strongest hallucination signal.

**Architecture** (same for all 32 classifiers):
```python
nn.Sequential(
    nn.Linear(4096, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 1),
    nn.Sigmoid(),
)
```

**Training config** (per layer):
- Input: `X[:, layer_idx, :]` — shape `(N, 4096)`
- Loss: `BCELoss` with `pos_weight = n_neg / n_pos` (handles class imbalance)
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- Early stopping: patience=10 on val AUROC
- Batch size: 256

**Output per layer**:
- `outputs/classifiers/layer_{i:02d}.pt` — saved model weights
- `outputs/classifiers/metrics.json` — AUROC, F1, precision, recall per layer

**Key output**: plot of `AUROC vs layer index (0-31)`. The peak layer is
the candidate for the production guardrail.

**Files to build**:
```
phase2/
  dataset.py   ← ActivationDataset: memory-maps X.npy, selects layer by index
  train.py     ← training loop for all 32 layers, saves metrics + plot
```

**Pre-training step**: filter `INFERENCE_ERROR` from X/y before training.
See `docs/phase2_spec.md` for the filtering snippet.

---

## Phase 4 — Evaluation & Guardrail  🔲 PENDING

**Goal**: Integrate the best-layer classifier as a real-time guardrail.

**Evaluation**:
- AUROC, AUPRC on held-out test set
- Per-hallucination-type breakdown
- Latency benchmark (32 hooks + classifier overhead in ms)
- False positive analysis

**Guardrail**:
```python
class HallucinationGuardrail:
    def __init__(self, runner, classifier, best_layer, threshold=0.5):
        ...
    def generate_safe(self, messages):
        output, activations = runner.generate_with_hidden_state(messages)
        # activations: (32, 4096) — select best layer
        vec = torch.tensor(activations[best_layer], dtype=torch.float32)
        prob = classifier(vec.unsqueeze(0))
        if prob > threshold:
            # block / retry / flag
            ...
        return output, prob
```

---

## Design decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Evaluation | Deterministic AST | Avoids cost and non-determinism of LLM judge |
| Hook layers | All 32 transformer blocks | Probing study to find most discriminative layer |
| Pooling | Last token of prefill (per layer) | Most information-dense position before generation |
| Quantization | NF4 double-quant | Best quality/VRAM tradeoff on T4 |
| Hallucination label | Binary (0/1) | Simple classifier target; fine-grained type stored separately |
| Classifier per layer | 32 independent MLPs | Clean probing: each measures intrinsic discriminability of that layer |
| Multi-turn | Included (except long_context) | Higher hallucination rate; `max_seq_len=3072` prevents OOM |
| INFERENCE_ERROR | Filtered before training | No real signal: inference never ran for these samples |
| Attention impl | `sdpa` (PyTorch SDPA) | Reduces attention memory O(n²)→O(n); no extra packages |
| Dual-GPU loading | Sequential | bitsandbytes not thread-safe; parallel loading causes CUDA races |
