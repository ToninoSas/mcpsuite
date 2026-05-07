"""
Unisce due directory di attivazioni (es. single-turn e multi-turn).

Strategia:
  - train, val, test : merge di src_a + src_b (con shuffle) per tutti gli split

Uso:
    python merge_activations.py \
        --src_a  ../phase1/outputs_single/activations \
        --src_b  ../phase1/outputs_multi/activations \
        --out    ../phase1/outputs_merged/activations
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_a", required=True, help="Sorgente A (es. single-turn)")
    parser.add_argument("--src_b", required=True, help="Sorgente B (es. multi-turn)")
    parser.add_argument("--out",   required=True, help="Directory di output")
    parser.add_argument("--seed",  type=int, default=42)
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


if __name__ == "__main__":
    main()
