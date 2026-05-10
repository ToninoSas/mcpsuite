"""
Phase 4 — Valutazione del classificatore sul test set (15% held-out).

Per ogni layer specificato (default: tutti) carica il classifier addestrato,
calcola le predizioni sul test set e riporta:
  - AUROC, AUPRC con 95% CI bootstrap
  - Soglia ottimale (Youden's J = max sensitivity + specificity - 1)
  - Precision, Recall, F1, Accuracy alla soglia ottimale
  - Breakdown per categoria (da meta.jsonl)
  - Plot ROC + Precision-Recall curve + distribuzione degli score

Uso:
    python eval.py \
        --activations_dir  ../outputs/augmented/activations \
        --classifiers_dir  classifiers_single_augmented \
        --out_dir          eval_results \
        --layer            13

    # tutti i layer (per confronto):
    python eval.py \
        --activations_dir  ../outputs/augmented/activations \
        --classifiers_dir  classifiers_single_augmented \
        --out_dir          eval_results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "phase3"))
from dataset import ActivationDataset


# ─────────────────────────────────────────────────────────────────────────────
# Architettura (identica a train.py)
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
# Inferenza
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model: nn.Module,
    X: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size].to(device)
            probs.append(model(batch).squeeze(1).cpu().numpy())
    return np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# Soglia ottimale — Youden's J
# ─────────────────────────────────────────────────────────────────────────────

def optimal_threshold_youden(labels: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_fn,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), size=len(labels))
        y, p = labels[idx], probs[idx]
        if 0 < y.sum() < len(y):
            scores.append(metric_fn(y, p))
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Metriche complete a una soglia
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    bootstrap_n: int = 1000,
) -> dict:
    preds = (probs >= threshold).astype(int)

    auroc = float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else 0.5
    auprc = float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0

    auroc_lo, auroc_hi = bootstrap_ci(labels, probs, roc_auc_score, n=bootstrap_n)
    auprc_lo, auprc_hi = bootstrap_ci(labels, probs, average_precision_score, n=bootstrap_n)

    majority_baseline = float((labels == 0).mean())

    return {
        "n_samples":        int(len(labels)),
        "n_pos":            int(labels.sum()),
        "n_neg":            int((labels == 0).sum()),
        "majority_baseline": round(majority_baseline, 4),
        "threshold":        round(threshold, 4),
        "auroc":            round(auroc, 4),
        "auroc_ci_95":      [round(auroc_lo, 4), round(auroc_hi, 4)],
        "auprc":            round(auprc, 4),
        "auprc_ci_95":      [round(auprc_lo, 4), round(auprc_hi, 4)],
        "precision":        round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":           round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1":               round(float(f1_score(labels, preds, zero_division=0)), 4),
        "accuracy":         round(float((preds == labels).mean()), 4),
        "tp": int(((preds == 1) & (labels == 1)).sum()),
        "fp": int(((preds == 1) & (labels == 0)).sum()),
        "tn": int(((preds == 0) & (labels == 0)).sum()),
        "fn": int(((preds == 0) & (labels == 1)).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Breakdown per categoria
# ─────────────────────────────────────────────────────────────────────────────

def per_category_metrics(
    meta: list[dict],
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict:
    categories: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        cat = m.get("category", "unknown")
        categories.setdefault(cat, []).append(i)

    result = {}
    for cat, idx in sorted(categories.items()):
        idx = np.array(idx)
        y, p = labels[idx], probs[idx]
        preds = (p >= threshold).astype(int)
        n_pos = int(y.sum())
        result[cat] = {
            "n":         len(y),
            "n_pos":     n_pos,
            "auroc":     round(float(roc_auc_score(y, p)), 4) if 0 < n_pos < len(y) else None,
            "recall":    round(float(recall_score(y, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "f1":        round(float(f1_score(y, preds, zero_division=0)), 4),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_eval(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    layer: int,
    metrics: dict,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib non disponibile — salto")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── ROC curve ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    ax = axes[0]
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"AUROC = {metrics['auroc']:.3f} "
                  f"(95% CI {metrics['auroc_ci_95'][0]:.3f}–{metrics['auroc_ci_95'][1]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Layer {layer}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Precision-Recall curve ───────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(labels, probs)
    ax = axes[1]
    ax.plot(rec, prec, color="darkorange", linewidth=2,
            label=f"AUPRC = {metrics['auprc']:.3f} "
                  f"(95% CI {metrics['auprc_ci_95'][0]:.3f}–{metrics['auprc_ci_95'][1]:.3f})")
    ax.axhline(metrics["n_pos"] / metrics["n_samples"], color="gray",
               linestyle="--", linewidth=1, label="random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — Layer {layer}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Score distribution ───────────────────────────────────────────────────
    ax = axes[2]
    ax.hist(probs[labels == 0], bins=30, alpha=0.6, color="steelblue",
            density=True, label=f"Negativi (n={metrics['n_neg']})")
    ax.hist(probs[labels == 1], bins=30, alpha=0.6, color="tomato",
            density=True, label=f"Positivi (n={metrics['n_pos']})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"soglia = {threshold:.3f}")
    ax.set_xlabel("Score (probabilità allucinazione)")
    ax.set_ylabel("Densità")
    ax.set_title(f"Distribuzione Score — Layer {layer}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Phase 4 — Layer {layer}  |  "
        f"Recall={metrics['recall']:.3f}  Precision={metrics['precision']:.3f}  "
        f"F1={metrics['f1']:.3f}  (soglia={threshold:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] salvato → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione singolo layer
# ─────────────────────────────────────────────────────────────────────────────

def eval_layer(
    layer_idx:       int,
    activations_dir: Path,
    classifiers_dir: Path,
    out_dir:         Path,
    device:          str,
    bootstrap_n:     int,
) -> dict | None:
    weights_path = classifiers_dir / f"layer_{layer_idx:02d}.pt"
    if not weights_path.exists():
        print(f"[layer {layer_idx:02d}] ⚠  {weights_path} non trovato — salto")
        return None

    test_ds = ActivationDataset(activations_dir / "test", layer=layer_idx)
    if len(test_ds) == 0:
        print(f"[layer {layer_idx:02d}] ⚠  test set vuoto — salto")
        return None

    model = build_classifier(hidden_size=test_ds.X.shape[1]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    probs  = predict(model, test_ds.X, device)
    labels = test_ds.y.numpy().astype(int)
    meta   = test_ds.meta

    threshold = optimal_threshold_youden(labels, probs)
    metrics   = compute_metrics(labels, probs, threshold, bootstrap_n)
    per_cat   = per_category_metrics(meta, labels, probs, threshold)

    result = {
        "layer":       layer_idx,
        "metrics":     metrics,
        "per_category": per_cat,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"layer_{layer_idx:02d}.json", "w") as f:
        json.dump(result, f, indent=2)

    plot_eval(
        labels, probs, threshold, layer_idx, metrics,
        out_dir / f"layer_{layer_idx:02d}.png",
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir",  required=True)
    parser.add_argument("--classifiers_dir",  required=True)
    parser.add_argument("--out_dir",          required=True)
    parser.add_argument("--layer",   type=int, default=None,
                        help="Layer da valutare (default: tutti)")
    parser.add_argument("--bootstrap_n", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    activations_dir = Path(args.activations_dir)
    classifiers_dir = Path(args.classifiers_dir)
    out_dir         = Path(args.out_dir)

    shape_path = activations_dir / "train" / "shape.json"
    num_layers = json.loads(shape_path.read_text()).get("num_layers", 32) if shape_path.exists() else 32
    layers = [args.layer] if args.layer is not None else list(range(num_layers))

    print(f"Device          : {args.device}")
    print(f"Test set        : {activations_dir / 'test'}")
    print(f"Classifiers     : {classifiers_dir}")
    print(f"Layer(s)        : {layers}")
    print(f"Bootstrap n     : {args.bootstrap_n}")
    print()

    all_results = []
    for layer_idx in layers:
        print(f"[layer {layer_idx:02d}] valutazione...", end=" ", flush=True)
        result = eval_layer(
            layer_idx, activations_dir, classifiers_dir,
            out_dir, args.device, args.bootstrap_n,
        )
        if result is None:
            continue
        m = result["metrics"]
        print(
            f"AUROC={m['auroc']:.4f} [{m['auroc_ci_95'][0]:.3f}–{m['auroc_ci_95'][1]:.3f}]  "
            f"Recall={m['recall']:.3f}  Prec={m['precision']:.3f}  "
            f"F1={m['f1']:.3f}  thr={m['threshold']:.3f}"
        )
        all_results.append(result)

    if not all_results:
        print("Nessun risultato prodotto.")
        return

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # riepilogo testuale
    print(f"\n{'Layer':>6}  {'AUROC':>7}  {'95% CI':>15}  "
          f"{'AUPRC':>7}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'Thr':>6}")
    print("─" * 80)
    for r in all_results:
        m = r["metrics"]
        ci = m["auroc_ci_95"]
        print(
            f"  {r['layer']:>4}   {m['auroc']:>5.4f}  "
            f"[{ci[0]:.3f}–{ci[1]:.3f}]  "
            f"{m['auprc']:>5.4f}  {m['recall']:>5.3f}  "
            f"{m['precision']:>5.3f}  {m['f1']:>5.3f}  {m['threshold']:>4.3f}"
        )

    if len(all_results) > 1:
        best = max(all_results, key=lambda r: r["metrics"]["auroc"])
        print(f"\nBest layer: {best['layer']}  "
              f"AUROC={best['metrics']['auroc']:.4f}  "
              f"F1={best['metrics']['f1']:.4f}")

    print(f"\nRisultati salvati in {out_dir}/")


if __name__ == "__main__":
    main()
