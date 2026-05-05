"""
Unisce due directory di attivazioni (es. single-turn e multi-turn).

Strategia:
  - train : merge di src_a + src_b (con shuffle)
  - val   : copia da src_a (proporzioni originali intatte)
  - test  : copia da src_a (proporzioni originali intatte)

Uso:
    python merge_activations.py \
        --src_a  ../phase1/outputs_single/activations \
        --src_b  ../phase1/outputs_multi/activations \
        --out    ../phase1/outputs_merged/activations

    # Per usare src_b come sorgente per val/test:
    python merge_activations.py ... --val_src b
"""

from __future__ import annotations

import argparse
import json
import shutil
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Merge (solo train)
# ─────────────────────────────────────────────────────────────────────────────

def merge_train(src_a: Path, src_b: Path, out_dir: Path, seed: int = 42) -> dict:
    a = src_a / "train"
    b = src_b / "train"

    X_a = np.load(a / "X.npy", mmap_mode="r")
    X_b = np.load(b / "X.npy", mmap_mode="r")
    y_a = np.load(a / "y.npy", mmap_mode="r")
    y_b = np.load(b / "y.npy", mmap_mode="r")

    meta_a = [json.loads(l) for l in (a / "meta.jsonl").read_text().splitlines() if l.strip()]
    meta_b = [json.loads(l) for l in (b / "meta.jsonl").read_text().splitlines() if l.strip()]

    assert X_a.shape[1:] == X_b.shape[1:], (
        f"Shape incompatibili: {X_a.shape[1:]} vs {X_b.shape[1:]}"
    )

    X = np.concatenate([X_a, X_b], axis=0)
    y = np.concatenate([y_a, y_b], axis=0)
    meta = meta_a + meta_b

    # Shuffle riproducibile: evita che il modello veda tutti i single-turn
    # prima di tutti i multi-turn durante l'addestramento
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


# ─────────────────────────────────────────────────────────────────────────────
# Copia (val e test)
# ─────────────────────────────────────────────────────────────────────────────

def copy_split(src: Path, split: str, out_dir: Path) -> dict:
    """Copia X.npy, y.npy, meta.jsonl, shape.json senza modifiche."""
    src_split = src / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("X.npy", "y.npy", "meta.jsonl", "shape.json"):
        shutil.copy2(src_split / fname, out_dir / fname)

    shape_info = json.loads((out_dir / "shape.json").read_text())
    # Aggiunge nota sulla provenienza
    shape_info["copied_from"] = str(src)
    with open(out_dir / "shape.json", "w") as f:
        json.dump(shape_info, f, indent=2)

    y = np.load(out_dir / "y.npy")
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return {"n_samples": len(y), "n_pos": n_pos, "n_neg": n_neg,
            "pos_rate": round(n_pos / len(y), 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_a",   required=True, help="Sorgente A (es. single-turn)")
    parser.add_argument("--src_b",   required=True, help="Sorgente B (es. multi-turn)")
    parser.add_argument("--out",     required=True, help="Directory di output")
    parser.add_argument("--val_src", default="a", choices=["a", "b"],
                        help="Sorgente per val e test (default: a)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    src_a    = Path(args.src_a)
    src_b    = Path(args.src_b)
    out      = Path(args.out)
    eval_src = src_a if args.val_src == "a" else src_b

    print(f"Sorgente A (train+)   : {src_a}")
    print(f"Sorgente B (train+)   : {src_b}")
    print(f"Sorgente val/test     : {eval_src}  (--val_src {args.val_src})")
    print(f"Output                : {out}")
    print()

    # ── train: merge ──────────────────────────────────────────────────────────
    print("[train] merge in corso...", end=" ", flush=True)
    info = merge_train(src_a, src_b, out / "train", seed=args.seed)
    print(
        f"{info['n_samples']} sample  "
        f"(A={info['n_from_a']} + B={info['n_from_b']})  "
        f"pos={info['n_pos']} ({info['pos_rate']*100:.1f}%)  "
        f"neg={info['n_neg']}"
    )

    # ── val / test: copia dalla sorgente scelta ───────────────────────────────
    for split in ("val", "test"):
        src_split = eval_src / split
        if not (src_split / "X.npy").exists():
            print(f"[{split}] ⚠  {src_split} non trovato — salto")
            continue
        print(f"[{split}] copia da {args.val_src}...", end=" ", flush=True)
        info = copy_split(eval_src, split, out / split)
        print(
            f"{info['n_samples']} sample  "
            f"pos={info['n_pos']} ({info['pos_rate']*100:.1f}%)  "
            f"neg={info['n_neg']}"
        )

    # ── riepilogo pos_weight per il training ──────────────────────────────────
    y_train = np.load(out / "train" / "y.npy")
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print()
    print(f"pos_weight (train): {n_neg / n_pos:.2f}  → da passare a BCELoss nel training")


if __name__ == "__main__":
    main()
