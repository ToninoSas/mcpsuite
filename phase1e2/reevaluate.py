"""
reevaluate.py — Ri-applica l'evaluator a un dataset già etichettato.

Quando si introduce un fix o un'estensione del parser/comparator in
evaluator.py, i `labeled_dataset.jsonl` esistenti possono diventare obsoleti
senza che cambi nulla negli `model_raw_output`. Questo script:

  1. Legge `labeled_dataset.jsonl` (o `.json`) di un esperimento
  2. Ri-valuta ogni sample con l'evaluator attuale
  3. Rigenera `labeled_dataset.jsonl`, `splits/{train,val,test}.jsonl`,
     `metrics.json`
  4. Aggiorna `activations/{split}/y.npy` e `meta.jsonl` (X.npy resta
     invariato — è l'ordine e gli id che servono per mantenere
     l'allineamento riga-per-riga con i nuovi label)

Uso tipico:
  python reevaluate.py --exp_dir ../outputs/llama/llama/standard
  python reevaluate.py --exp_dir ../outputs/llama/llama/standard --dry_run

Output:
  - File aggiornati in-place dentro `exp_dir`
  - Report con conteggio di label cambiati e diff di hallucination rate
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluator import evaluate, evaluate_multi_turn
from loader import MULTI_TURN_CATEGORIES


# ─────────────────────────────────────────────────────────────────────────────
# Lettura dataset
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset(exp_dir: Path) -> tuple[Path, list[dict]]:
    """
    Trova labeled_dataset.json[l] in exp_dir e lo carica.
    Supporta tre formati:
      a) JSONL (un record per riga) — standard pipeline.py
      b) JSON array top-level         — `[ {...}, {...} ]`
      c) Pretty-printed concatenato   — `{...}{...}{...}` con indentazione
    """
    candidates = [
        exp_dir / "labeled_dataset.jsonl",
        exp_dir / "labeled_dataset.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Nessun labeled_dataset.[json|jsonl] trovato in {exp_dir}"
        )

    text = path.read_text(encoding="utf-8")
    records: list[dict] = []

    # Tentativo a) JSONL puro
    try:
        records = [json.loads(l) for l in text.splitlines() if l.strip()]
        if records and all(isinstance(r, dict) for r in records):
            return path, records
    except json.JSONDecodeError:
        records = []

    # Tentativo b) array top-level
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return path, [r for r in parsed if isinstance(r, dict)]
    except json.JSONDecodeError:
        pass

    # Tentativo c) JSON concatenati (pretty-printed o no) con raw_decode
    decoder = json.JSONDecoder()
    idx = 0
    n   = len(text)
    while idx < n:
        # salta whitespace tra record
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            records.append(obj)
        idx = end

    return path, records


# ─────────────────────────────────────────────────────────────────────────────
# Re-evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _reevaluate_record(rec: dict) -> dict:
    """
    Ri-applica evaluate() / evaluate_multi_turn() a un record.
    Restituisce un nuovo dict con label e hallucination_type aggiornati.
    """
    category = rec.get("category", "")
    raw_output = rec.get("model_raw_output", "")
    gt = rec.get("ground_truth", [])

    if category in MULTI_TURN_CATEGORIES:
        # multi-turn: raw_output è list[str], gt è list[list[...]]
        turn_outputs = raw_output if isinstance(raw_output, list) else [raw_output]
        per_turn_gt  = gt if (gt and isinstance(gt[0], list)) else [gt]
        result = evaluate_multi_turn(turn_outputs, per_turn_gt, category)
    else:
        # single-turn (incluse live_*)
        if isinstance(raw_output, list):
            raw_output = raw_output[0] if raw_output else ""
        result = evaluate(raw_output, gt, category=category)

    new_rec = dict(rec)
    new_rec["label"]              = result.label
    new_rec["hallucination_type"] = result.hallucination_type
    new_rec["eval_details"]       = result.details
    return new_rec


# ─────────────────────────────────────────────────────────────────────────────
# Scrittura output
# ─────────────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _rewrite_splits(exp_dir: Path, records: list[dict]) -> None:
    """Aggiorna splits/*.jsonl mantenendo l'assegnazione di ogni id allo split."""
    splits_dir = exp_dir / "splits"
    if not splits_dir.exists():
        print(f"  [splits] {splits_dir} non esiste — skip")
        return

    by_id = {r["id"]: r for r in records}
    for split_file in splits_dir.glob("*.jsonl"):
        old_records = [
            json.loads(l)
            for l in split_file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        new_records = [by_id.get(r["id"], r) for r in old_records]
        _write_jsonl(split_file, new_records)
        print(f"  [splits] aggiornato {split_file.name} ({len(new_records)} record)")


def _rewrite_activations(exp_dir: Path, records: list[dict]) -> None:
    """
    Aggiorna activations/{split}/y.npy e meta.jsonl mantenendo l'ordine
    delle righe (cioè la corrispondenza con X.npy).
    """
    acts_base = exp_dir / "activations"
    if not acts_base.exists():
        print(f"  [activations] {acts_base} non esiste — skip")
        return

    by_id = {r["id"]: r for r in records}

    for split_dir in sorted(acts_base.iterdir()):
        if not split_dir.is_dir():
            continue
        meta_path = split_dir / "meta.jsonl"
        y_path    = split_dir / "y.npy"
        if not (meta_path.exists() and y_path.exists()):
            print(f"  [activations] {split_dir.name}: meta/y mancanti — skip")
            continue

        old_meta = [
            json.loads(l)
            for l in meta_path.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        new_meta, new_y, missing = [], [], 0
        for m in old_meta:
            rec = by_id.get(m["id"])
            if rec is None:
                # id non più presente nel dataset rie-valutato — mantieni
                # le info originali per non rompere l'allineamento con X.npy
                missing += 1
                new_meta.append(m)
                new_y.append(0)  # fallback neutro
                continue
            new_meta.append({
                "id":                 rec["id"],
                "category":           rec["category"],
                "hallucination_type": rec["hallucination_type"],
            })
            new_y.append(int(rec["label"]))

        _write_jsonl(meta_path, new_meta)
        np.save(y_path, np.array(new_y, dtype=np.int8))
        msg = f"  [activations] {split_dir.name}: {len(new_y)} record aggiornati"
        if missing:
            msg += f" (⚠ {missing} id non trovati nel dataset)"
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(records: list[dict], prev_metrics: dict | None = None) -> dict:
    """Calcola un metrics.json analogo a quello prodotto da pipeline.py."""
    total = len(records)
    n_halluc = sum(1 for r in records if r["label"] == 1)
    n_correct = total - n_halluc

    htype_counts: Counter = Counter()
    cat_stats: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "n_correct": 0, "n_halluc": 0, "htypes": Counter()}
    )
    for r in records:
        cat = r["category"]
        st = cat_stats[cat]
        st["n"] += 1
        if r["label"] == 1:
            st["n_halluc"] += 1
            ht = r.get("hallucination_type")
            if ht:
                htype_counts[ht] += 1
                st["htypes"][ht] += 1
        else:
            st["n_correct"] += 1

    per_category = {}
    for cat, st in sorted(cat_stats.items()):
        per_category[cat] = {
            "n":                  st["n"],
            "n_correct":          st["n_correct"],
            "n_halluc":           st["n_halluc"],
            "accuracy":           round(st["n_correct"] / st["n"], 4) if st["n"] else 0.0,
            "hallucination_rate": round(st["n_halluc"]  / st["n"], 4) if st["n"] else 0.0,
            "hallucination_types": dict(st["htypes"]),
        }

    metrics = {
        "total":              total,
        "n_correct":          n_correct,
        "n_halluc":           n_halluc,
        "accuracy":           round(n_correct / total, 4) if total else 0.0,
        "hallucination_rate": round(n_halluc  / total, 4) if total else 0.0,
        "hallucination_types": dict(htype_counts),
        "per_category":       per_category,
    }
    # Preserva campi del precedente metrics.json se presenti
    if prev_metrics:
        for k in ("model", "splits", "timings_sec", "inference_samples_per_sec", "sources"):
            if k in prev_metrics:
                metrics[k] = prev_metrics[k]
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _diff_report(old_records: list[dict], new_records: list[dict]) -> None:
    old_by_id = {r["id"]: r for r in old_records}
    flipped_to_correct = 0
    flipped_to_halluc  = 0
    same                = 0
    by_cat_flips: dict[str, dict] = defaultdict(lambda: {"to_correct": 0, "to_halluc": 0})

    for r in new_records:
        old = old_by_id.get(r["id"])
        if old is None:
            continue
        if old["label"] == r["label"]:
            same += 1
        elif old["label"] == 1 and r["label"] == 0:
            flipped_to_correct += 1
            by_cat_flips[r["category"]]["to_correct"] += 1
        elif old["label"] == 0 and r["label"] == 1:
            flipped_to_halluc += 1
            by_cat_flips[r["category"]]["to_halluc"] += 1

    print(f"\n  Diff label vecchi vs nuovi:")
    print(f"    invariati                  : {same}")
    print(f"    1 → 0  (halluc → corretto) : {flipped_to_correct}")
    print(f"    0 → 1  (corretto → halluc) : {flipped_to_halluc}")

    if by_cat_flips:
        print(f"\n  Flip per categoria:")
        for cat in sorted(by_cat_flips):
            f = by_cat_flips[cat]
            print(f"    {cat:<30} +{f['to_correct']:>3} corretti, -{f['to_halluc']:>3} alluc")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def reevaluate_experiment(exp_dir: Path, dry_run: bool = False) -> None:
    print(f"\n══ Re-evaluation: {exp_dir} ══")

    dataset_path, records = _load_dataset(exp_dir)
    print(f"  Caricati {len(records)} record da {dataset_path.name}")

    new_records = [_reevaluate_record(r) for r in records]

    old_halluc = sum(1 for r in records if r["label"] == 1)
    new_halluc = sum(1 for r in new_records if r["label"] == 1)
    n = len(new_records)
    print(f"  Hallucination rate: {old_halluc/n*100:.1f}% → {new_halluc/n*100:.1f}%  "
          f"({old_halluc} → {new_halluc} su {n})")

    _diff_report(records, new_records)

    if dry_run:
        print("\n  [dry-run] Nessun file scritto.")
        return

    # Scrittura
    _write_jsonl(dataset_path, new_records)
    print(f"\n  [dataset] sovrascritto {dataset_path.name}")

    _rewrite_splits(exp_dir, new_records)
    _rewrite_activations(exp_dir, new_records)

    metrics_path = exp_dir / "metrics.json"
    prev_metrics = None
    if metrics_path.exists():
        try:
            prev_metrics = json.loads(metrics_path.read_text())
        except Exception:
            prev_metrics = None
    metrics = _compute_metrics(new_records, prev_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"  [metrics] sovrascritto metrics.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True, type=str,
                   help="Directory dell'esperimento (contiene labeled_dataset.json[l])")
    p.add_argument("--dry_run", action="store_true",
                   help="Calcola il diff senza scrivere nulla su disco")
    args = p.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        print(f"ERRORE: {exp_dir} non esiste", file=sys.stderr)
        sys.exit(1)

    reevaluate_experiment(exp_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
