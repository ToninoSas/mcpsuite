"""
Phase 3 — Per-layer classifier training with k-fold cross-validation.

Strategia:
  - CV pool: train (70%) + val (15%) concatenati → 85% del dataset totale
  - 5-fold StratifiedKFold sul CV pool → cv_auroc_mean ± std per layer selection
  - Modello finale: addestrato sull'intero CV pool per mean_epochs (media dai fold)
  - Test set (15%) usato solo in Phase 4

Output:
    out_dir/
        layer_XX.pt         ← pesi del modello finale
        metrics.json        ← cv_auroc_mean, cv_auroc_std, mean_epochs per layer
        auroc_per_layer.png
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from dataset import ActivationDataset


# ─────────────────────────────────────────────────────────────────────────────
# Architettura
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_size, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training con early stopping (usato nei fold CV)
# ─────────────────────────────────────────────────────────────────────────────

def train_on_tensors(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    epochs:       int   = 100,
    patience:     int   = 10,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> tuple[dict, dict, int]:
    """
    Addestra un MLP con early stopping su val AUROC.
    Restituisce (metriche_val, state_dict_migliore, best_epoch).
    """
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_w = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False,
    )

    model     = build_classifier(hidden_size=X_train.shape[1]).to(device)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auroc = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred    = model(X_b).squeeze(1)
            weights = torch.where(y_b == 1, pos_w, torch.ones_like(y_b))
            loss    = (criterion(pred, y_b) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        probs_list, labels_list = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                probs_list.append(model(X_b.to(device)).squeeze(1).cpu().numpy())
                labels_list.append(y_b.numpy())

        probs  = np.concatenate(probs_list)
        labels = np.concatenate(labels_list).astype(int)
        auroc  = roc_auc_score(labels, probs) if 0 < labels.sum() < len(labels) else 0.5

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            probs_list.append(model(X_b.to(device)).squeeze(1).cpu().numpy())
            labels_list.append(y_b.numpy())

    probs  = np.concatenate(probs_list)
    labels = np.concatenate(labels_list).astype(int)
    preds  = (probs >= 0.5).astype(int)

    metrics = {
        "auroc":     float(roc_auc_score(labels, probs)) if labels.sum() > 0 else 0.5,
        "auprc":     float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0,
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
    }
    return metrics, best_state, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation (sul CV pool 85%)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    X: torch.Tensor,
    y: torch.Tensor,
    k:     int = 5,
    seed:  int = 42,
    device: str = "cpu",
    **train_kwargs,
) -> dict:
    """
    k-fold StratifiedKFold sul CV pool (train+val concatenati).
    Restituisce mean/std AUROC e media degli epoch ottimali tra i fold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_aurocs      = []
    fold_best_epochs = []

    for tr_idx, vl_idx in skf.split(X.numpy(), y.numpy().astype(int)):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]
        metrics, _, best_epoch = train_on_tensors(
            X_tr, y_tr, X_vl, y_vl, device=device, **train_kwargs
        )
        fold_aurocs.append(metrics["auroc"])
        fold_best_epochs.append(best_epoch)

    return {
        "cv_auroc_mean": float(np.mean(fold_aurocs)),
        "cv_auroc_std":  float(np.std(fold_aurocs)),
        "fold_aurocs":   fold_aurocs,
        "mean_epochs":   int(np.mean(fold_best_epochs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training finale su tutto il CV pool (nessun early stopping)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs:     int,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    device:       str   = "cpu",
) -> dict:
    """
    Addestra il modello finale su tutti i dati per n_epochs fissi.
    n_epochs = media degli epoch ottimali dai fold CV.
    """
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    pos_w = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, device=device)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size, shuffle=True,
    )

    model     = build_classifier(hidden_size=X.shape[1]).to(device)
    criterion = nn.BCELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(n_epochs):
        model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred    = model(X_b).squeeze(1)
            weights = torch.where(y_b == 1, pos_w, torch.ones_like(y_b))
            loss    = (criterion(pred, y_b) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline per singolo layer
# ─────────────────────────────────────────────────────────────────────────────

def train_one_layer(
    layer_idx:       int,
    activations_dir: Path,
    out_dir:         Path,
    k_folds:         int   = 5,
    epochs:          int   = 100,
    patience:        int   = 10,
    batch_size:      int   = 256,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    device:          str   = "cpu",
) -> dict:
    # CV pool = train (70%) + val (15%) = 85%
    train_ds = ActivationDataset(activations_dir / "train", layer=layer_idx)
    val_ds   = ActivationDataset(activations_dir / "val",   layer=layer_idx)

    X_cv = torch.cat([train_ds.X.to(torch.float32), val_ds.X.to(torch.float32)], dim=0)
    y_cv = torch.cat([train_ds.y, val_ds.y], dim=0)

    train_kwargs = dict(
        epochs=epochs, patience=patience,
        batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, device=device,
    )

    # 1) CV sul CV pool → stima robusta per layer selection
    cv = cross_validate(X_cv, y_cv, k=k_folds, **train_kwargs)

    # 2) Modello finale: tutto il CV pool per mean_epochs (no early stopping)
    final_state = train_final_model(
        X_cv, y_cv,
        n_epochs=max(cv["mean_epochs"], 1),
        batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, device=device,
    )
    torch.save(final_state, out_dir / f"layer_{layer_idx:02d}.pt")

    return {
        "layer":         layer_idx,
        "cv_auroc_mean": cv["cv_auroc_mean"],
        "cv_auroc_std":  cv["cv_auroc_std"],
        "fold_aurocs":   cv["fold_aurocs"],
        "mean_epochs":   cv["mean_epochs"],
        "n_cv_pool":     len(y_cv),
        "pos_cv_pool":   int(y_cv.sum().item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc(layer_metrics: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib non disponibile — salto")
        return

    layers   = [m["layer"]         for m in layer_metrics]
    cv_means = [m["cv_auroc_mean"] for m in layer_metrics]
    cv_stds  = [m["cv_auroc_std"]  for m in layer_metrics]
    best_idx = int(np.argmax(cv_means))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.errorbar(
        layers, cv_means, yerr=cv_stds,
        fmt="o-", linewidth=1.5, markersize=4,
        color="steelblue", ecolor="lightsteelblue", elinewidth=1.5, capsize=3,
        label="CV AUROC mean ± std (85% pool)",
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random baseline")
    ax.axvline(
        layers[best_idx], color="tomato", linestyle="--", linewidth=1.2,
        label=f"best layer {layers[best_idx]}  "
              f"(CV={cv_means[best_idx]:.3f}±{cv_stds[best_idx]:.3f})",
    )
    ax.scatter([layers[best_idx]], [cv_means[best_idx]],
               color="tomato", zorder=5, s=70)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUROC")
    ax.set_title("Hallucination probe — AUROC per transformer layer")
    ax.set_xticks(layers)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] salvato → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--out_dir",         required=True)
    parser.add_argument("--layers",     type=int, nargs="+", default=None)
    parser.add_argument("--k_folds",    type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    out_dir         = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_path = activations_dir / "train" / "shape.json"
    num_layers = json.loads(shape_path.read_text()).get("num_layers", 32) if shape_path.exists() else 32
    layers_to_train = args.layers if args.layers else list(range(num_layers))

    print(f"Device    : {args.device}")
    print(f"Layers    : {layers_to_train}")
    print(f"CV folds  : {args.k_folds}")
    print(f"CV pool   : train (70%) + val (15%) = 85%  [test 15% intoccato fino a Phase 4]")
    print(f"Out dir   : {out_dir}")
    print()

    all_metrics: list[dict] = []
    t_start = time.time()

    for layer_idx in layers_to_train:
        t0 = time.time()
        print(f"[layer {layer_idx:02d}/{num_layers-1}] CV...", end=" ", flush=True)
        m = train_one_layer(
            layer_idx=layer_idx,
            activations_dir=activations_dir,
            out_dir=out_dir,
            k_folds=args.k_folds,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        elapsed = time.time() - t0
        print(
            f"CV={m['cv_auroc_mean']:.4f}±{m['cv_auroc_std']:.4f}  "
            f"epochs={m['mean_epochs']}  "
            f"({elapsed:.1f}s)"
        )
        all_metrics.append(m)

    all_metrics.sort(key=lambda x: x["layer"])

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'Layer':>6}  {'CV AUROC':>10}  {'±std':>6}  {'Epochs':>6}")
    print("─" * 38)
    for m in all_metrics:
        print(
            f"  {m['layer']:>4}   {m['cv_auroc_mean']:>8.4f}  "
            f"{m['cv_auroc_std']:>6.4f}  {m['mean_epochs']:>6}"
        )

    best = max(all_metrics, key=lambda x: x["cv_auroc_mean"])
    print(f"\nBest layer: {best['layer']}  "
          f"CV={best['cv_auroc_mean']:.4f}±{best['cv_auroc_std']:.4f}  "
          f"epochs={best['mean_epochs']}")
    print(f"Tempo totale: {(time.time()-t_start)/60:.1f} min")

    if len(all_metrics) > 1:
        plot_auroc(all_metrics, out_dir / "auroc_per_layer.png")


if __name__ == "__main__":
    main()
