"""
Phase 3 — Per-layer classifier training.

Trains 32 independent binary MLP classifiers (one per transformer layer) on the
activation files produced by phase1/pipeline.py --capture_activations.
Saves per-layer weights, metrics, and an AUROC-vs-layer-index plot.

Usage:
    python train.py \
        --activations_dir ../phase1/outputs/activations \
        --out_dir         ../phase1/outputs/classifiers

    # train only a subset of layers (for quick experiments):
    python train.py --layers 0 8 16 23 31

Output:
    out_dir/
        layer_00.pt  ...  layer_31.pt   ← model weights
        metrics.json                    ← AUROC, F1, precision, recall per layer
        auroc_per_layer.png             ← the probing curve
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import ActivationDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_classifier(hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_size, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    )


# ---------------------------------------------------------------------------
# Training loop (single layer)
# ---------------------------------------------------------------------------

def train_one_layer(
    layer_idx: int,
    activations_dir: Path,
    out_dir: Path,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | str = "cpu",
) -> dict:
    train_ds = ActivationDataset(activations_dir / "train", layer=layer_idx)
    val_ds   = ActivationDataset(activations_dir / "val",   layer=layer_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_classifier(hidden_size=train_ds.X.shape[1]).to(device)
    criterion = nn.BCELoss(reduction="mean")
    # pos_weight not used inside BCELoss directly — we handle it via weight tensor
    pos_w = train_ds.pos_weight.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auroc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch).squeeze(1)
            # manual pos_weight: scale loss for positives
            weights = torch.where(y_batch == 1, pos_w, torch.ones_like(y_batch))
            loss = (criterion(pred, y_batch) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- validate ---
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                probs = model(X_batch.to(device)).squeeze(1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_batch.numpy())

        probs  = np.concatenate(all_probs)
        labels = np.concatenate(all_labels).astype(int)

        if labels.sum() == 0 or labels.sum() == len(labels):
            auroc = 0.5
        else:
            auroc = roc_auc_score(labels, probs)

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # --- final evaluation on val set with best model ---
    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            probs = model(X_batch.to(device)).squeeze(1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds  = (probs >= 0.5).astype(int)

    metrics = {
        "layer": layer_idx,
        "auroc":     float(roc_auc_score(labels, probs)) if labels.sum() > 0 else 0.5,
        "auprc":     float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0,
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "n_train":   len(train_ds),
        "n_val":     len(val_ds),
        "pos_train": train_ds.class_counts["pos"],
        "pos_val":   val_ds.class_counts["pos"],
    }

    # save model
    torch.save(best_state, out_dir / f"layer_{layer_idx:02d}.pt")
    return metrics


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_auroc(layer_metrics: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available, skipping plot")
        return

    layers = [m["layer"] for m in layer_metrics]
    aurocs = [m["auroc"]  for m in layer_metrics]

    best_idx = int(np.argmax(aurocs))
    best_layer = layers[best_idx]
    best_auroc = aurocs[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, aurocs, marker="o", linewidth=1.5, markersize=4, color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random baseline")
    ax.axvline(best_layer, color="tomato", linestyle="--", linewidth=1,
               label=f"best layer {best_layer} (AUROC={best_auroc:.3f})")
    ax.scatter([best_layer], [best_auroc], color="tomato", zorder=5, s=60)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUROC (val set)")
    ax.set_title("Hallucination probe — AUROC per transformer layer")
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-layer hallucination classifiers")
    parser.add_argument("--activations_dir", default="../phase1/outputs/activations",
                        help="Root dir with train/ val/ test/ subdirs")
    parser.add_argument("--out_dir",         default="../phase1/outputs/classifiers")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to train (default: all 0-31)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    out_dir         = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # detect number of layers from shape.json
    shape_path = activations_dir / "train" / "shape.json"
    if shape_path.exists():
        shape_info = json.loads(shape_path.read_text())
        num_layers = shape_info.get("num_layers", 32)
    else:
        num_layers = 32

    layers_to_train = args.layers if args.layers is not None else list(range(num_layers))

    print(f"Device : {args.device}")
    print(f"Layers : {layers_to_train}")
    print(f"Out dir: {out_dir}")
    print()

    all_metrics: list[dict] = []
    t_start = time.time()

    for layer_idx in layers_to_train:
        t0 = time.time()
        print(f"[layer {layer_idx:02d}/{num_layers-1}] training...", end=" ", flush=True)
        m = train_one_layer(
            layer_idx=layer_idx,
            activations_dir=activations_dir,
            out_dir=out_dir,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        elapsed = time.time() - t0
        print(f"AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}  ({elapsed:.1f}s)")
        all_metrics.append(m)

    # sort by layer index before saving
    all_metrics.sort(key=lambda x: x["layer"])

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # summary table
    print("\n--- AUROC per layer ---")
    print(f"{'Layer':>6}  {'AUROC':>7}  {'AUPRC':>7}  {'F1':>6}  {'pos_val':>7}")
    for m in all_metrics:
        print(f"  {m['layer']:>4}   {m['auroc']:.4f}   {m['auprc']:.4f}  {m['f1']:.4f}  {m['pos_val']:>7}")

    best = max(all_metrics, key=lambda x: x["auroc"])
    print(f"\nBest layer: {best['layer']}  AUROC={best['auroc']:.4f}")
    print(f"Total training time: {(time.time() - t_start)/60:.1f} min")

    # plot
    if len(all_metrics) > 1:
        plot_auroc(all_metrics, out_dir / "auroc_per_layer.png")


if __name__ == "__main__":
    main()
