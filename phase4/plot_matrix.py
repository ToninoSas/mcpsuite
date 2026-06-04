"""
plot_matrix.py — Heatmap della matrice di generalizzazione train × test.

Ogni cella della heatmap è un esperimento di Phase 4: un classificatore
addestrato sulla distribuzione `train` e valutato sul test set della
distribuzione `test`. Il colore = AUROC del best layer; l'annotazione riporta
AUROC, 95% CI e l'indice del best layer.

Serve a rispondere alla domanda di generalizzazione cross-distribution:
  - diagonale (train==test) = performance in-distribution
  - fuori diagonale         = transfer tra distribuzioni

Uso (CLI) — una --cell per esperimento, formato "train,test,path/results.json":
    python plot_matrix.py \
        --cell single,single,eval-results/single-on-single/results.json \
        --cell single,multi, eval-results/multi-on-single/results.json \
        --cell multi,single, eval-results/single-on-multi/results.json \
        --cell multi,multi,  eval-results/multi-on-multi/results.json \
        --row_order single multi \
        --col_order single multi \
        --title "Llama-3.1-8B — matrice train×test" \
        --out matrix_llama.png

Uso (import, es. in cella Kaggle):
    from plot_matrix import best_from_results, plot_train_test_matrix
    cells = {
        ("single", "single"): best_from_results(".../single-on-single/results.json"),
        ...
    }
    plot_train_test_matrix(cells, ["single","multi"], ["single","multi"],
                           out_path="matrix.png", title="...")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Estrazione del best layer da un results.json di Phase 4
# ─────────────────────────────────────────────────────────────────────────────

def best_from_results(results_path: str | Path) -> dict:
    """
    Legge un results.json (lista di dict per-layer) e restituisce le metriche
    del best layer (per AUROC):  {"auroc", "ci", "layer", "f1", "accuracy"}.
    """
    data = json.loads(Path(results_path).read_text())
    if isinstance(data, dict) and "layers" in data:
        data = data["layers"]
    best = max(data, key=lambda r: r["metrics"]["auroc"])
    m = best["metrics"]
    return {
        "auroc":    m["auroc"],
        "ci":       m.get("auroc_ci_95", [m["auroc"], m["auroc"]]),
        "layer":    best["layer"],
        "f1":       m.get("f1"),
        "accuracy": m.get("accuracy"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_test_matrix(
    cells:     dict[tuple[str, str], dict],
    row_order: list[str],
    col_order: list[str],
    out_path:  str | Path,
    title:     str = "Matrice di generalizzazione train × test",
    metric:    str = "auroc",
    vmin:      float = 0.5,
    vmax:      float = 1.0,
) -> None:
    """
    cells:     {(train_label, test_label): {"auroc", "ci", "layer", ...}}
    row_order: etichette delle righe (train set), dall'alto in basso
    col_order: etichette delle colonne (test set), da sinistra a destra
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_rows, n_cols = len(row_order), len(col_order)
    grid = np.full((n_rows, n_cols), np.nan)
    for i, tr in enumerate(row_order):
        for j, te in enumerate(col_order):
            cell = cells.get((tr, te))
            if cell is not None:
                grid[i, j] = cell[metric]

    fig, ax = plt.subplots(figsize=(2.4 * n_cols + 2.5, 2.0 * n_rows + 2.0))

    im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="equal")

    # ticks ed etichette
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([f"test:\n{c}" for c in col_order], fontsize=11)
    ax.set_yticklabels([f"train:\n{r}" for r in row_order], fontsize=11)
    ax.set_xlabel("Test set (distribuzione di valutazione)", fontsize=11)
    ax.set_ylabel("Train set (distribuzione di addestramento)", fontsize=11)

    # bordo spesso sulle celle diagonali (in-distribution)
    for i, tr in enumerate(row_order):
        for j, te in enumerate(col_order):
            if tr == te:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="black",
                                           linewidth=3))

    # annotazioni in ogni cella
    for i, tr in enumerate(row_order):
        for j, te in enumerate(col_order):
            cell = cells.get((tr, te))
            if cell is None:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=14, color="gray")
                continue
            val = cell[metric]
            ci  = cell.get("ci", [val, val])
            lay = cell.get("layer")
            # testo scuro/chiaro a seconda del valore (verde chiaro → testo scuro)
            text_color = "black"
            txt = (
                f"{val:.3f}\n"
                f"[{ci[0]:.2f}–{ci[1]:.2f}]\n"
                f"layer {lay}"
            )
            tag = "in-dist" if tr == te else "transfer"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold")
            ax.text(j, i + 0.34, tag, ha="center", va="center",
                    fontsize=8, color="dimgray", style="italic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric.upper(), fontsize=11)

    ax.set_title(title, fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] matrice salvata → {out_path}")

    # riepilogo testuale
    print(f"\n{'train→test':>16}  {'AUROC':>7}  {'95% CI':>15}  {'layer':>6}  tipo")
    print("─" * 62)
    for tr in row_order:
        for te in col_order:
            cell = cells.get((tr, te))
            if cell is None:
                continue
            ci = cell.get("ci", [0, 0])
            tag = "in-dist" if tr == te else "transfer"
            print(f"  {tr+'→'+te:>14}  {cell['auroc']:>6.3f}  "
                  f"[{ci[0]:.3f}–{ci[1]:.3f}]  {cell['layer']:>6}  {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", action="append", required=True,
                   help="train,test,path/results.json  (ripetibile)")
    p.add_argument("--row_order", nargs="+", required=True,
                   help="ordine etichette train (righe)")
    p.add_argument("--col_order", nargs="+", required=True,
                   help="ordine etichette test (colonne)")
    p.add_argument("--out",   default="matrix.png")
    p.add_argument("--title", default="Matrice di generalizzazione train × test")
    p.add_argument("--metric", default="auroc")
    args = p.parse_args()

    cells: dict[tuple[str, str], dict] = {}
    for spec in args.cell:
        parts = [x.strip() for x in spec.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--cell malformato: {spec!r} (atteso train,test,path)")
        train, test, path = parts
        cells[(train, test)] = best_from_results(path)

    plot_train_test_matrix(
        cells, args.row_order, args.col_order,
        out_path=args.out, title=args.title, metric=args.metric,
    )


if __name__ == "__main__":
    main()
