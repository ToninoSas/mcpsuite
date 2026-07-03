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


def per_layer_curve(results_path: str | Path) -> tuple[list, list, list, list]:
    """
    Estrae da un results.json la curva AUROC-per-layer completa:
    restituisce (layers, auroc, ci_lo, ci_hi), ordinati per layer.
    """
    data = json.loads(Path(results_path).read_text())
    if isinstance(data, dict) and "layers" in data:
        data = data["layers"]
    data = sorted(data, key=lambda r: r["layer"])
    layers = [r["layer"] for r in data]
    auroc  = [r["metrics"]["auroc"] for r in data]
    ci     = [r["metrics"].get("auroc_ci_95", [r["metrics"]["auroc"]] * 2) for r in data]
    lo     = [c[0] for c in ci]
    hi     = [c[1] for c in ci]
    return layers, auroc, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Curva AUROC-per-layer (una o più valutazioni sovrapposte)
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc_curves(
    curves:   dict[str, str],
    out_path: str | Path,
    title:    str | None = None,
    ymin:     float = 0.4,
    ymax:     float = 1.0,
) -> None:
    """
    Sovrappone le curve AUROC-per-layer di una o più valutazioni.
    Utile per confrontare la stessa valutazione tra modelli (es. merged→merged
    di Qwen vs Llama) o per mostrare una singola curva.

    curves: {etichetta: path/results.json}  (ordine preservato in legenda)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

    fig, ax = plt.subplots(figsize=(13, 6))

    for k, (label, path) in enumerate(curves.items()):
        if not Path(path).exists():
            print(f"⚠  {path} non trovato — skip")
            continue
        layers, auroc, lo, hi = per_layer_curve(path)
        color = palette[k % len(palette)]
        best = int(np.argmax(auroc))

        ax.plot(layers, auroc, "o-", color=color, linewidth=2, markersize=4,
                label=f"{label}  (best layer {layers[best]}: {auroc[best]:.2f})")
        ax.fill_between(layers, lo, hi, color=color, alpha=0.13)
        ax.axvline(layers[best], color=color, linestyle="--", linewidth=1.0, alpha=0.5)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random baseline (0.5)")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    if title:
        ax.set_title(title, fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] curve AUROC-per-layer salvate → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Griglia AUROC-per-layer (2×2): mostra la gobba in-distribution vs il collasso
# nel transfer
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc_per_layer_grid(
    cells_paths: dict[tuple[str, str], str],
    row_order:   list[str],
    col_order:   list[str],
    out_path:    str | Path,
    title:       str | None = None,
) -> None:
    """
    Griglia di curve AUROC-per-layer, una per cella della matrice train×test.
    Layout: righe = training set, colonne = test set (come la heatmap).
    Le celle diagonali (in-distribution) sono colorate diversamente da quelle
    di transfer, così si vede la gobba a mid-depth vs il collasso a layer shallow.

    cells_paths: {(train, test): path/results.json}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_rows, n_cols = len(row_order), len(col_order)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for i, tr in enumerate(row_order):
        for j, te in enumerate(col_order):
            ax = axes[i][j]
            path = cells_paths.get((tr, te))
            if path is None or not Path(path).exists():
                ax.set_visible(False)
                continue

            layers, auroc, lo, hi = per_layer_curve(path)
            is_diag = (tr == te)
            color = "steelblue" if is_diag else "tomato"

            ax.plot(layers, auroc, "o-", color=color, markersize=3, linewidth=1.6)
            ax.fill_between(layers, lo, hi, color=color, alpha=0.15)
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

            best = int(np.argmax(auroc))
            ax.axvline(layers[best], color="black", linestyle="--",
                       linewidth=1.0, alpha=0.6)
            ax.text(0.97, 0.06,
                    f"best layer {layers[best]}  (AUROC {auroc[best]:.2f})",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="lightgray", alpha=0.85))

            tag = "in-distribution" if is_diag else "transfer"
            ax.set_title(f"train: {tr}  $\\rightarrow$  test: {te}   ({tag})",
                         fontsize=10, fontweight="bold" if is_diag else "normal")
            ax.set_ylim(0.4, 1.0)
            ax.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel("Layer index")
            if j == 0:
                ax.set_ylabel("AUROC")

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] griglia AUROC-per-layer salvata → {out_path}")


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
            delta = (ci[1] - ci[0]) / 2   # semi-ampiezza del CI → formato ±
            # testo scuro/chiaro a seconda del valore (verde chiaro → testo scuro)
            text_color = "black"
            txt = (
                f"{val:.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]\n"
                f"layer {lay}"
            )
            tag = "in-dist" if tr == te else "transfer"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=11, color=text_color, fontweight="bold")
            ax.text(j, i + 0.34, tag, ha="center", va="center",
                    fontsize=8, color="dimgray", style="italic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric.upper(), fontsize=11)

    # title=None o "" → nessun titolo (utile quando il nome del modello è dato
    # dalla sub-caption LaTeX in una figura affiancata)
    if title:
        ax.set_title(title, fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] matrice salvata → {out_path}")

    # riepilogo testuale
    print(f"\n{'train→test':>16}  {'AUROC (±CI)':>14}  {'layer':>6}  tipo")
    print("─" * 52)
    for tr in row_order:
        for te in col_order:
            cell = cells.get((tr, te))
            if cell is None:
                continue
            ci = cell.get("ci", [cell["auroc"], cell["auroc"]])
            delta = (ci[1] - ci[0]) / 2
            tag = "in-dist" if tr == te else "transfer"
            print(f"  {tr+'→'+te:>14}  {cell['auroc']:.2f} ± {delta:.2f}  "
                  f"{cell['layer']:>6}  {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# Bar chart — valutazioni a test fisso, varia il training set (es. colonna merged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fixed_test_bars(
    cells:       dict[tuple[str, str], dict],
    group_order: list[str],
    bar_order:   list[str],
    out_path:    str | Path,
    title:       str = "Valutazioni su test merged",
    test_label:  str = "merged",
    ymin:        float = 0.5,
    ymax:        float | None = None,
) -> None:
    """
    Bar chart raggruppato per confrontare valutazioni con lo STESSO test set
    (es. la colonna merged) al variare del training set, per più modelli.

    cells:       {(group, train): {"auroc", "ci", "layer"}}
                 group = etichetta del gruppo (es. modello "Qwen"/"Llama")
                 train = distribuzione di training (es. "single"/"multi"/"merged")
    group_order: gruppi sull'asse x (es. ["Qwen", "Llama"])
    bar_order:   barre dentro ogni gruppo (es. ["single", "multi", "merged"])
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_groups = len(group_order)
    n_bars   = len(bar_order)
    width    = 0.8 / n_bars
    x        = np.arange(n_groups)

    # palette per training set
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    colors  = {b: palette[i % len(palette)] for i, b in enumerate(bar_order)}

    # Limite superiore automatico: lascia spazio sopra l'estremo superiore del CI
    # più alto, così l'etichetta "valore / Lxx" (2 righe) non sfora il bordo.
    if ymax is None:
        upper_cis = [c.get("ci", [c["auroc"], c["auroc"]])[1] for c in cells.values()]
        max_top = max(upper_cis) if upper_cis else 1.0
        ymax = min(max_top + 0.08, 1.10)   # margine per le 2 righe di testo

    fig, ax = plt.subplots(figsize=(2.6 * n_groups + 3.0, 6))

    for j, bar in enumerate(bar_order):
        vals, los, his, layers = [], [], [], []
        for g in group_order:
            cell = cells.get((g, bar))
            if cell is None:
                vals.append(np.nan); los.append(0); his.append(0); layers.append(None)
            else:
                v  = cell["auroc"]
                ci = cell.get("ci", [v, v])
                vals.append(v)
                los.append(v - ci[0])
                his.append(ci[1] - v)
                layers.append(cell.get("layer"))
        offset = (j - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=[los, his], capsize=4,
                      color=colors[bar], label=f"train: {bar}",
                      edgecolor="black", linewidth=0.6, error_kw={"elinewidth": 1.2})
        # annota valore + best layer sopra l'estremo superiore del CI
        for xi, v, hi_off, lay in zip(x + offset, vals, his, layers):
            if np.isnan(v):
                continue
            ax.text(xi, v + hi_off + 0.012, f"{v:.2f}\nL{lay}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=12)
    ax.set_ylabel("AUROC (best layer)", fontsize=11)
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"{title}  —  test set: {test_label}", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] bar chart salvato → {out_path}")

    # riepilogo testuale
    print(f"\n{'gruppo':>10}  {'train':>8}  {'AUROC':>7}  {'95% CI':>15}  {'layer':>6}")
    print("─" * 56)
    for g in group_order:
        for b in bar_order:
            cell = cells.get((g, b))
            if cell is None:
                continue
            ci = cell.get("ci", [0, 0])
            print(f"  {g:>8}  {b:>8}  {cell['auroc']:>6.3f}  "
                  f"[{ci[0]:.3f}–{ci[1]:.3f}]  {cell['layer']:>6}")


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
