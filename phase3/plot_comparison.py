"""
Confronta i risultati di due esperimenti di training su un unico grafico.

Uso tipico:
    python plot_comparison.py \
        --metrics_a classifiers_mixed/metrics.json   --label_a "Mixed (single+multi)" \
        --metrics_b classifiers_single/metrics.json  --label_b "Single-turn only" \
        --out       comparison.png
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def load_metrics(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def plot_comparison(
    metrics_a: list[dict],
    metrics_b: list[dict],
    label_a:   str,
    label_b:   str,
    out_path:  Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def extract(metrics, key):
        return [m[key] for m in metrics]

    layers_a = extract(metrics_a, "layer")
    layers_b = extract(metrics_b, "layer")

    auroc_a  = extract(metrics_a, "cv_auroc_mean")
    auroc_b  = extract(metrics_b, "cv_auroc_mean")
    auroc_sa = extract(metrics_a, "cv_auroc_std")
    auroc_sb = extract(metrics_b, "cv_auroc_std")

    acc_a    = extract(metrics_a, "cv_accuracy_mean")
    acc_b    = extract(metrics_b, "cv_accuracy_mean")
    acc_sa   = extract(metrics_a, "cv_accuracy_std")
    acc_sb   = extract(metrics_b, "cv_accuracy_std")

    best_a = int(np.argmax(auroc_a))
    best_b = int(np.argmax(auroc_b))

    # baseline accuracy = classe maggioritaria per ogni esperimento
    def majority_baseline(metrics):
        n_pos = metrics[0]["pos_cv_pool"]
        n_tot = metrics[0]["n_cv_pool"]
        return (n_tot - n_pos) / n_tot

    maj_a = majority_baseline(metrics_a)
    maj_b = majority_baseline(metrics_b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # ── subplot 1: AUROC ─────────────────────────────────────────────────────
    ax1.errorbar(
        layers_a, auroc_a, yerr=auroc_sa,
        fmt="o-", linewidth=1.5, markersize=4,
        color="steelblue", ecolor="lightsteelblue", elinewidth=1.5, capsize=3,
        label=f"{label_a}  (best layer {layers_a[best_a]}: {auroc_a[best_a]:.3f})",
    )
    ax1.errorbar(
        layers_b, auroc_b, yerr=auroc_sb,
        fmt="s--", linewidth=1.5, markersize=4,
        color="darkorange", ecolor="moccasin", elinewidth=1.5, capsize=3,
        label=f"{label_b}  (best layer {layers_b[best_b]}: {auroc_b[best_b]:.3f})",
    )
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random baseline")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("CV AUROC per layer", fontsize=10)

    # ── subplot 2: Accuracy ──────────────────────────────────────────────────
    ax2.errorbar(
        layers_a, acc_a, yerr=acc_sa,
        fmt="o-", linewidth=1.5, markersize=4,
        color="steelblue", ecolor="lightsteelblue", elinewidth=1.5, capsize=3,
        label=f"{label_a}  (majority baseline {maj_a:.2f})",
    )
    ax2.errorbar(
        layers_b, acc_b, yerr=acc_sb,
        fmt="s--", linewidth=1.5, markersize=4,
        color="darkorange", ecolor="moccasin", elinewidth=1.5, capsize=3,
        label=f"{label_b}  (majority baseline {maj_b:.2f})",
    )
    ax2.axhline(maj_a, color="steelblue",  linestyle=":", linewidth=1, alpha=0.6)
    ax2.axhline(maj_b, color="darkorange", linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("CV Accuracy per layer", fontsize=10)

    layers_all = sorted(set(layers_a) | set(layers_b))
    ax2.set_xticks(layers_all)

    fig.suptitle("Hallucination probe — Confronto esperimenti", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Salvato → {out_path}")

    # riepilogo testuale
    print(f"\n{'Layer':>6}  {'AUROC_A':>9}  {'AUROC_B':>9}  {'Δ AUROC':>8}  "
          f"{'Acc_A':>7}  {'Acc_B':>7}  {'Δ Acc':>7}")
    print("─" * 65)
    for i, layer in enumerate(layers_a):
        d_auroc = auroc_a[i] - auroc_b[i]
        d_acc   = acc_a[i]   - acc_b[i]
        print(
            f"  {layer:>4}   {auroc_a[i]:>7.4f}   {auroc_b[i]:>7.4f}  "
            f"{d_auroc:>+8.4f}  {acc_a[i]:>5.4f}  {acc_b[i]:>5.4f}  {d_acc:>+7.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_a", required=True, help="metrics.json esperimento A")
    parser.add_argument("--metrics_b", required=True, help="metrics.json esperimento B")
    parser.add_argument("--label_a",   default="Esperimento A")
    parser.add_argument("--label_b",   default="Esperimento B")
    parser.add_argument("--out",       default="comparison.png")
    args = parser.parse_args()

    metrics_a = load_metrics(args.metrics_a)
    metrics_b = load_metrics(args.metrics_b)

    plot_comparison(
        metrics_a, metrics_b,
        args.label_a, args.label_b,
        Path(args.out),
    )


if __name__ == "__main__":
    main()
