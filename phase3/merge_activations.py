"""
Unisce due directory di attivazioni e (opzionalmente) i relativi metrics.json.

Strategia:
  - train, val, test : merge di src_a + src_b (con shuffle) per tutti gli split

Uso:
    python merge_activations.py \
        --src_a  ../outputs_single/activations \
        --src_b  ../outputs_live/activations \
        --out    ../outputs_augmented/activations \
        --metrics_a ../outputs_single/metrics.json \
        --metrics_b ../outputs_live/metrics.json \
        --metrics_out ../outputs_augmented/metrics.json
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path


def merge_split(src_a: Path, src_b: Path, split: str, out_dir: Path, seed: int = 42) -> dict:
    """Merge train/val/test split da due sorgenti con shuffle riproducibile."""
    a = src_a / split
    b = src_b / split

    X_a = np.load(a / "X.npy", mmap_mode="r")
    X_b = np.load(b / "X.npy", mmap_mode="r")
    y_a = np.load(a / "y.npy", mmap_mode="r")
    y_b = np.load(b / "y.npy", mmap_mode="r")

    meta_a = [json.loads(l) for l in (a / "meta.jsonl").read_text().splitlines() if l.strip()]
    meta_b = [json.loads(l) for l in (b / "meta.jsonl").read_text().splitlines() if l.strip()]

    assert X_a.shape[1:] == X_b.shape[1:], (
        f"Shape incompatibili: {X_a.shape[1:]} vs {X_b.shape[1:]}"
    )

    X    = np.concatenate([X_a, X_b], axis=0)
    y    = np.concatenate([y_a, y_b], axis=0)
    meta = meta_a + meta_b

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y, meta = X[idx], y[idx], [meta[i] for i in idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X.astype(np.float16))
    np.save(out_dir / "y.npy", y.astype(np.int8))
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    shape_info = {
        "X_shape":     list(X.shape),
        "X_dtype":     "float16",
        "y_dtype":     "int8",
        "num_layers":  X.shape[1],
        "hidden_size": X.shape[2],
        "n_samples":   X.shape[0],
        "n_pos":       n_pos,
        "n_neg":       n_neg,
        "pos_rate":    round(n_pos / len(y), 4),
        "n_from_a":    len(y_a),
        "n_from_b":    len(y_b),
        "sources":     {"a": str(src_a), "b": str(src_b)},
    }
    with open(out_dir / "shape.json", "w") as f:
        json.dump(shape_info, f, indent=2)

    return shape_info


def merge_metrics(path_a: str, path_b: str, out_path: str) -> dict:
    """
    Unisce due metrics.json prodotti dalla pipeline.

    Somma i contatori assoluti (total, n_correct, n_halluc, splits, timings,
    hallucination_types, per_category) e ricalcola le rate derivate.
    Le categorie presenti in una sola sorgente vengono incluse così come sono.
    """
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())

    def sum_type_counts(ta: dict, tb: dict) -> dict:
        merged: dict[str, int] = {}
        for k, v in {**ta, **tb}.items():
            merged[k] = ta.get(k, 0) + tb.get(k, 0)
        return dict(sorted(merged.items(), key=lambda x: -x[1]))

    total     = a["total"]     + b["total"]
    n_correct = a["n_correct"] + b["n_correct"]
    n_halluc  = a["n_halluc"]  + b["n_halluc"]

    merged_types = sum_type_counts(
        a.get("hallucination_types", {}),
        b.get("hallucination_types", {}),
    )

    per_cat: dict[str, dict] = {}
    all_cats = set(a.get("per_category", {}).keys()) | set(b.get("per_category", {}).keys())
    for cat in sorted(all_cats):
        ca = a.get("per_category", {}).get(cat, {})
        cb = b.get("per_category", {}).get(cat, {})
        if not ca:
            per_cat[cat] = cb
            continue
        if not cb:
            per_cat[cat] = ca
            continue
        n     = ca["n"]     + cb["n"]
        nc    = ca["n_correct"] + cb["n_correct"]
        nh    = ca["n_halluc"]  + cb["n_halluc"]
        per_cat[cat] = {
            "n":                 n,
            "n_correct":         nc,
            "n_halluc":          nh,
            "accuracy":          round(nc / n, 4) if n else 0.0,
            "hallucination_rate": round(nh / n, 4) if n else 0.0,
            "hallucination_types": sum_type_counts(
                ca.get("hallucination_types", {}),
                cb.get("hallucination_types", {}),
            ),
        }

    splits_a = a.get("splits", {})
    splits_b = b.get("splits", {})
    all_splits = set(splits_a) | set(splits_b)
    merged_splits = {s: splits_a.get(s, 0) + splits_b.get(s, 0) for s in sorted(all_splits)}

    tim_a = a.get("timings_sec", {})
    tim_b = b.get("timings_sec", {})
    all_tim = set(tim_a) | set(tim_b)
    merged_tim = {k: round(tim_a.get(k, 0.0) + tim_b.get(k, 0.0), 2) for k in sorted(all_tim)}

    total_inf_time = merged_tim.get("inference", 0.0)
    sps = round(total / total_inf_time, 3) if total_inf_time > 0 else 0.0

    result = {
        "total":              total,
        "n_correct":          n_correct,
        "n_halluc":           n_halluc,
        "accuracy":           round(n_correct / total, 4) if total else 0.0,
        "hallucination_rate": round(n_halluc  / total, 4) if total else 0.0,
        "hallucination_types": merged_types,
        "per_category":       per_cat,
        "splits":             merged_splits,
        "timings_sec":        merged_tim,
        "inference_samples_per_sec": sps,
        "sources": {"a": str(path_a), "b": str(path_b)},
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Metriche unite → {out_path}")
    print(
        f"  Totale: {total}  |  Allucinazioni: {n_halluc} "
        f"({result['hallucination_rate']*100:.1f}%)  |  Corretti: {n_correct} "
        f"({result['accuracy']*100:.1f}%)"
    )
    print(f"  Categorie: {', '.join(per_cat.keys())}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_a", required=True, help="Sorgente A (es. single-turn)")
    parser.add_argument("--src_b", required=True, help="Sorgente B (es. multi-turn)")
    parser.add_argument("--out",   required=True, help="Directory di output")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--metrics_a",   default=None, help="metrics.json sorgente A")
    parser.add_argument("--metrics_b",   default=None, help="metrics.json sorgente B")
    parser.add_argument("--metrics_out", default=None, help="Percorso metrics.json unito")
    args = parser.parse_args()

    src_a = Path(args.src_a)
    src_b = Path(args.src_b)
    out   = Path(args.out)

    print(f"Sorgente A : {src_a}")
    print(f"Sorgente B : {src_b}")
    print(f"Output     : {out}")
    print()

    for split in ("train", "val", "test"):
        src_split_a = src_a / split
        src_split_b = src_b / split
        if not (src_split_a / "X.npy").exists():
            print(f"[{split}] ⚠  {src_split_a} non trovato — salto")
            continue
        if not (src_split_b / "X.npy").exists():
            print(f"[{split}] ⚠  {src_split_b} non trovato — salto")
            continue
        print(f"[{split}] merge in corso...", end=" ", flush=True)
        info = merge_split(src_a, src_b, split, out / split, seed=args.seed)
        print(
            f"{info['n_samples']} sample  "
            f"(A={info['n_from_a']} + B={info['n_from_b']})  "
            f"pos={info['n_pos']} ({info['pos_rate']*100:.1f}%)  "
            f"neg={info['n_neg']}"
        )

    y_train = np.load(out / "train" / "y.npy")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print()
    print(f"pos_weight (train): {n_neg / n_pos:.2f}  → da passare a BCELoss nel training")

    if args.metrics_a and args.metrics_b:
        metrics_out = args.metrics_out or str(out.parent / "metrics.json")
        print()
        merge_metrics(args.metrics_a, args.metrics_b, metrics_out)


if __name__ == "__main__":
    main()
