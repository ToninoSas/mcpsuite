# Roadmap — LLM Agent Hallucination Probe

## Overview

The goal is to predict, *before* a tool call is emitted, whether a
Qwen2.5-7B agent is about to hallucinate. The prediction signal comes from
the model's own internal state — the residual stream of the final transformer
block — not from post-hoc output analysis.

```
User query + function schemas
        │
        ▼
  Qwen2.5-7B-4bit
  [prefill phase]
        │
        ├──► forward hook on layers[-1]
        │         │
        │         ▼
        │    hidden state [1, seq, 3584]
        │         │
        │    last-token pool → R^3584
        │         │
        │    Binary Classifier
        │    (Linear→ReLU→Sigmoid)
        │         │
        │    P(hallucination)
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

## Phase 1 — Test Suite & Dataset  ✅ COMPLETE

**Goal**: Build a labeled dataset of (prompt, function_schema, model_output,
label) tuples from the BFCL benchmark.

**Source**: Berkeley Function-Calling Leaderboard v3
(`gorilla-llm/Berkeley-Function-Calling-Leaderboard`)

**Categories** (proportional sampling):
- `simple` (25%) — one function, one call
- `multiple` (20%) — multiple functions available, one correct
- `parallel` (10%) — one function, multiple simultaneous calls
- `parallel_multiple` (5%) — multiple functions, multiple calls
- `multi_turn_base` (15%) — multi-turn conversation, complete info
- `multi_turn_miss_func` (8%) — requested function unavailable
- `multi_turn_miss_param` (8%) — underspecified parameters
- `multi_turn_long_context` (5%) — long conversation history
- `multi_turn_composite` (4%) — combined multi-turn challenges

**Labeling strategy**: deterministic AST matching against ground truth.
No LLM-as-judge. A call is `label=1` if it does not match within
type-coercion and set-matching rules.

**Files**: `phase1/`

**Output**: `phase1/outputs/labeled_dataset.jsonl` + `splits/`

---

## Phase 2 — Residual Stream Capture  🔲 NEXT

**Goal**: Re-run the labeled samples through Qwen, capture the hidden state
from the final transformer block at the end of the prefill pass, and store
the activations alongside their labels.

**Hook target**: `model.model.layers[-1]` — fires once per forward pass
during prefill.

**Pooling**: last-token hidden state (index `[-1]` on the sequence dim).
Mean pooling is a secondary ablation.

**Storage format**:
```
outputs/activations/
  train/
    X.npy          shape (N_train, 3584)  dtype float16
    y.npy          shape (N_train,)       dtype int8
    meta.jsonl     id, category, hallucination_type per row
  val/   (same)
  test/  (same)
```

**Implementation note**: `runner.py::TransformersRunner.generate_with_hidden_state()`
is already implemented. Phase 2 only needs `capture.py` to iterate the JSONL
and call it, and `dataset.py` to wrap the .npy files in a PyTorch Dataset.

**Key decision**: capture activations from the *prefill* of the full prompt
(system + functions + user query), not from the generation pass. The model
has already "decided" what to generate by this point.

---

## Phase 3 — Binary Classifier Training  🔲 PENDING

**Goal**: Train a shallow MLP on the captured activations.

**Architecture** (fixed, per user spec):
```python
nn.Sequential(
    nn.Linear(3584, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 1),
    nn.Sigmoid(),
)
```

**Training config**:
- Loss: `BCELoss` with `pos_weight` to handle class imbalance
- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- Scheduler: cosine annealing or ReduceLROnPlateau on val AUROC
- Early stopping: patience=10 epochs on val AUROC
- Batch size: 256 (activations fit in RAM, no GPU needed for training)
- Epochs: up to 100 (early stopping will kick in)

**Metrics to track**:
- Primary: AUROC on val set
- Secondary: F1, precision, recall at threshold θ=0.5
- Threshold tuning: optimize θ on val set for target precision/recall

**Ablations to run**:
1. Last-token vs mean-pool
2. Layer -1 vs layer -2 vs layer -4
3. With vs without BatchNorm
4. Dropout rate (0.1, 0.3, 0.5)

---

## Phase 4 — Evaluation & Deployment  🔲 PENDING

**Goal**: Integrate the trained classifier as a real-time guardrail.

**Evaluation**:
- AUROC, AUPRC on held-out test set
- Per-hallucination-type breakdown (which types are hardest to catch?)
- Latency benchmark (hook + classifier overhead in ms)
- False positive analysis (what correct calls get blocked?)

**Guardrail integration**:
```python
class HallucinationGuardrail:
    def __init__(self, runner, classifier, threshold=0.5):
        ...
    def generate_safe(self, messages):
        output, hidden = runner.generate_with_hidden_state(messages)
        prob = classifier(pool(hidden))
        if prob > threshold:
            # Option A: block and return error
            # Option B: retry with modified prompt
            # Option C: flag for human review
            ...
        return output, prob
```

**Active learning loop**:
- Flag high-confidence wrong predictions for manual review
- Add reviewed samples back to training set
- Re-train monthly or when error rate exceeds threshold

---

## Design decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Evaluation | Deterministic AST | User requirement; avoids cost and non-determinism of LLM judge |
| Hook layer | Last transformer block | Most context-integrated representation; closest to output |
| Pooling | Last-token (ablate mean) | Last token is the "summary" token before generation |
| Quantization | NF4 double-quant | Best quality/VRAM tradeoff; preserves hidden state fidelity |
| Hallucination label | Binary (0/1) | Simple classifier target; fine-grained type stored for analysis |
| Hidden dim input | 3584 (Qwen2.5-7B) | Fixed by model architecture |
| Classifier width | 512 | User specification |
| Activation | ReLU | User specification |
| Output activation | Sigmoid | User specification; appropriate for binary BCE |
