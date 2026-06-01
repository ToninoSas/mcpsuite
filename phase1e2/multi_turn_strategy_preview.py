"""
multi_turn_strategy_preview.py — Confronta strategie di aggregazione dei
label sui sample multi-turn SENZA rifare inferenza né training.

Lo script legge `labeled_dataset.jsonl` di un esperimento multi-turn, estrae
`per_turn_labels` da `eval_details`, e applica diverse regole di aggregazione
mostrando l'impatto su:
  - numero di sample mantenuti
  - hallucination rate complessivo
  - breakdown per categoria
  - distribuzione di n_turns

Strategie supportate:
  * any           regola attuale: label=1 se ALMENO un turno fallisce
  * majority      label=1 se >= 50% dei turni falliscono           (B1)
  * threshold     label=1 se >= THRESHOLD dei turni falliscono     (B2)
  * homogeneous   tieni solo sample con tutti turni uguali         (D1)
  * max_turns     tieni solo sample con n_turns <= K               (D2)

Uso programmatico (in cella Kaggle):
    from multi_turn_strategy_preview import (
        load_multi_turn_records, apply_strategy, summarize, print_summary,
        print_turns_distribution,
    )
    records = load_multi_turn_records("/path/to/labeled_dataset.jsonl")
    baseline = summarize(apply_strategy(records, "any"))
    s_b1 = summarize(apply_strategy(records, "majority"))
    print_summary("B1 — Majority (≥50%)", s_b1, baseline=baseline)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_multi_turn_records(path: str | Path) -> list[dict]:
    """
    Carica un labeled_dataset.json[l] e restituisce SOLO i sample multi-turn
    (quelli con `eval_details.per_turn_labels`).

    Supporta JSONL puro e JSON concatenato pretty-printed.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # tentativo JSONL
    raw_records: list[dict] = []
    try:
        raw_records = [json.loads(l) for l in text.splitlines() if l.strip()]
        if not (raw_records and all(isinstance(r, dict) for r in raw_records)):
            raw_records = []
    except json.JSONDecodeError:
        raw_records = []

    # fallback: concatenated JSON
    if not raw_records:
        decoder = json.JSONDecoder()
        idx, n = 0, len(text)
        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                raw_records.append(obj)
            idx = end

    out: list[dict] = []
    for rec in raw_records:
        details = rec.get("eval_details", {}) or {}
        ptl = details.get("per_turn_labels")
        if ptl is None or not isinstance(ptl, list):
            continue  # non multi-turn
        if not ptl:
            continue
        n_turns = len(ptl)
        n_failed = sum(1 for t in ptl if t == 1)
        out.append({
            "id":                 rec.get("id", ""),
            "category":           rec.get("category", ""),
            "label_any":          rec.get("label", 1 if n_failed else 0),
            "hallucination_type": rec.get("hallucination_type"),
            "per_turn_labels":    ptl,
            "n_turns":            n_turns,
            "n_failed":           n_failed,
            "frac_failed":        n_failed / n_turns,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Strategie
# ─────────────────────────────────────────────────────────────────────────────

def apply_strategy(
    records:  list[dict],
    strategy: str,
    **kwargs: Any,
) -> list[dict]:
    """
    Applica una strategia di aggregazione e restituisce la lista dei record
    risultanti, ognuno con il campo `new_label`.

    Le strategie 'homogeneous' e 'max_turns' filtrano (scartano sample); le
    altre re-etichettano senza filtrare.
    """
    out: list[dict] = []
    for r in records:
        n_failed = r["n_failed"]
        n_turns  = r["n_turns"]
        frac     = r["frac_failed"]
        r2 = dict(r)

        if strategy == "any":
            r2["new_label"] = 1 if n_failed > 0 else 0
            out.append(r2)

        elif strategy == "majority":
            r2["new_label"] = 1 if frac >= 0.5 else 0
            out.append(r2)

        elif strategy == "threshold":
            thr = float(kwargs.get("threshold", 0.3))
            r2["new_label"] = 1 if frac >= thr else 0
            out.append(r2)

        elif strategy == "homogeneous":
            # tieni solo sample tutti-corretti o tutti-sbagliati
            if n_failed == 0:
                r2["new_label"] = 0
                out.append(r2)
            elif n_failed == n_turns:
                r2["new_label"] = 1
                out.append(r2)
            # altrimenti scarta

        elif strategy == "max_turns":
            K = int(kwargs.get("K", 3))
            if n_turns <= K:
                r2["new_label"] = 1 if n_failed > 0 else 0  # any-turn entro K
                out.append(r2)
            # altrimenti scarta

        else:
            raise ValueError(f"strategia sconosciuta: {strategy!r}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize(records: list[dict], label_field: str = "new_label") -> dict:
    total = len(records)
    if total == 0:
        return {"total": 0, "n_halluc": 0, "halluc_rate": 0.0, "per_category": {}}

    n_halluc = sum(1 for r in records if r[label_field] == 1)

    by_cat: dict[str, dict] = {}
    for r in records:
        c = r["category"]
        if c not in by_cat:
            by_cat[c] = {"n": 0, "n_halluc": 0, "n_turns_sum": 0}
        by_cat[c]["n"] += 1
        by_cat[c]["n_halluc"] += r[label_field]
        by_cat[c]["n_turns_sum"] += r["n_turns"]

    for c, st in by_cat.items():
        st["halluc_rate"]  = st["n_halluc"] / st["n"]
        st["n_turns_avg"] = st["n_turns_sum"] / st["n"]
        del st["n_turns_sum"]

    return {
        "total":        total,
        "n_halluc":     n_halluc,
        "halluc_rate":  n_halluc / total,
        "per_category": by_cat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    name:     str,
    summary:  dict,
    baseline: dict | None = None,
) -> None:
    print(f"\n══ {name} {'═' * max(0, 70 - len(name) - 4)}")
    total = summary["total"]
    if total == 0:
        print("  Nessun sample.")
        return

    halluc_pct = summary["halluc_rate"] * 100
    delta_line = ""
    if baseline is not None:
        d_total = total - baseline["total"]
        d_halluc_pp = halluc_pct - baseline["halluc_rate"] * 100
        delta_line = (
            f"   Δ vs baseline: {d_total:+d} sample, "
            f"{d_halluc_pp:+.1f} pp halluc rate"
        )

    print(f"  Totale: {total}  |  Halluc: {summary['n_halluc']} "
          f"({halluc_pct:.1f}%){delta_line}")

    print(f"\n  {'categoria':<28} {'N':>5} {'halluc%':>9} {'avg n_turns':>13}")
    print(f"  {'─'*28} {'─'*5} {'─'*9} {'─'*13}")
    for cat in sorted(summary["per_category"]):
        st = summary["per_category"][cat]
        print(f"  {cat:<28} {st['n']:>5} {st['halluc_rate']*100:>8.1f}% "
              f"{st['n_turns_avg']:>12.2f}")


def print_turns_distribution(records: list[dict]) -> None:
    """Istogramma di n_turns nei record originali — utile per scegliere K in D2."""
    from collections import Counter
    counts = Counter(r["n_turns"] for r in records)
    print(f"\n  Distribuzione n_turns ({len(records)} sample multi-turn):")
    print(f"  {'n_turns':>8} {'count':>6} {'cum%':>7}  histogram")
    print(f"  {'─'*8} {'─'*6} {'─'*7}  {'─'*40}")
    cum = 0
    total = len(records)
    for k in sorted(counts):
        c = counts[k]
        cum += c
        bar = "█" * min(40, c)
        print(f"  {k:>8} {c:>6} {cum/total*100:>6.1f}%  {bar}")


def compare_strategies(rows: list[tuple[str, dict]]) -> None:
    """Tabella riassuntiva di tutte le strategie testate."""
    print(f"\n══ Confronto strategie {'═' * 48}")
    print(f"  {'Strategia':<35} {'Total':>6} {'Halluc':>8} {'Halluc%':>9}")
    print(f"  {'─'*35} {'─'*6} {'─'*8} {'─'*9}")
    for name, summary in rows:
        if summary["total"] == 0:
            print(f"  {name:<35} {'-':>6} {'-':>8} {'-':>9}")
            continue
        print(f"  {name:<35} {summary['total']:>6} "
              f"{summary['n_halluc']:>8} {summary['halluc_rate']*100:>8.1f}%")
