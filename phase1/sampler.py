"""
sampler.py — Campionamento proporzionale tra categorie BFCL

Strategia:
  - Raggruppiamo le categorie in 4 macro-gruppi logici:
      single_simple    → simple
      single_complex   → multiple, parallel, parallel_multiple
      multi_turn       → multi_turn_base, miss_func, miss_param, long_context, composite
  - L'utente specifica il budget totale di sample; il sampler distribuisce
    proporzionalmente rispettando la dimensione reale di ogni file.
  - All'interno di ogni categoria si usa campionamento casuale senza reinserimento.
  - Seme riproducibile per esperimenti comparabili.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

from loader import BFCLSample


# ── Pesi target per macro-gruppo ──────────────────────────────────────────────
# Modificabili dall'utente; verranno normalizzati a 1.

DEFAULT_WEIGHTS: dict[str, float] = {
    "simple":                    0.25,
    "multiple":                  0.20,
    "parallel":                  0.10,
    "parallel_multiple":         0.05,
    "multi_turn_base":           0.15,
    "multi_turn_miss_func":      0.08,
    "multi_turn_miss_param":     0.08,
    "multi_turn_long_context":   0.05,
    "multi_turn_composite":      0.04,
}


def proportional_sample(
    corpus: dict[str, list[BFCLSample]],
    total: int,
    weights: dict[str, float] | None = None,
    seed: int = 42,
    filter_fn: Callable[[BFCLSample], bool] | None = None,
) -> list[BFCLSample]:
    """
    Campiona `total` sample dal corpus in modo proporzionale.

    Args:
        corpus:    output di loader.load_all()
        total:     numero totale di sample desiderati
        weights:   peso relativo per categoria (default: DEFAULT_WEIGHTS)
        seed:      seme random per riproducibilità
        filter_fn: funzione opzionale per escludere sample (es. quelli senza GT)

    Returns:
        Lista di BFCLSample mescolati casualmente, pronti per l'inferenza.
    """
    rng = random.Random(seed)
    weights = weights or DEFAULT_WEIGHTS

    # Filtra il corpus se necessario
    available: dict[str, list[BFCLSample]] = {}
    for cat, samples in corpus.items():
        pool = [s for s in samples if filter_fn(s)] if filter_fn else samples
        if pool:
            available[cat] = pool

    # Normalizza i pesi sulle sole categorie presenti
    present_cats = [c for c in available]
    raw_w = {c: weights.get(c, 1.0) for c in present_cats}
    total_w = sum(raw_w.values())
    norm_w = {c: v / total_w for c, v in raw_w.items()}

    # Calcola quanti sample per categoria
    quotas: dict[str, int] = {}
    remainder: dict[str, float] = {}
    assigned = 0
    for cat in present_cats:
        exact = norm_w[cat] * total
        quotas[cat] = int(exact)
        remainder[cat] = exact - int(exact)
        assigned += quotas[cat]

    # Distribuisce i rimanenti alle categorie con fraction più alta
    leftover = total - assigned
    for cat in sorted(remainder, key=lambda c: remainder[c], reverse=True):
        if leftover == 0:
            break
        # Non superare la disponibilità reale
        cap = len(available[cat])
        if quotas[cat] < cap:
            quotas[cat] += 1
            leftover -= 1

    # Campiona
    selected: list[BFCLSample] = []
    stats: dict[str, int] = {}

    for cat in present_cats:
        n = min(quotas[cat], len(available[cat]))
        picked = rng.sample(available[cat], n)
        selected.extend(picked)
        stats[cat] = n

    rng.shuffle(selected)

    # Report
    print(f"\n[sampler] Budget totale: {total}  |  Effettivo: {len(selected)}")
    print(f"{'Categoria':<35} {'Peso%':>6}  {'Quota':>6}  {'Effettivo':>9}")
    print("─" * 62)
    for cat in present_cats:
        print(f"  {cat:<33} {norm_w[cat]*100:>5.1f}%  {quotas[cat]:>6}  {stats[cat]:>9}")
    print()

    return selected


def split_train_val_test(
    samples: list[BFCLSample],
    train: float = 0.70,
    val: float = 0.15,
    seed: int = 42,
) -> tuple[list[BFCLSample], list[BFCLSample], list[BFCLSample]]:
    """
    Divide il dataset in train/val/test mantenendo la distribuzione
    di categorie proporzionale (stratified split).
    """
    rng = random.Random(seed)

    # Raggruppa per categoria
    by_cat: dict[str, list[BFCLSample]] = defaultdict(list)
    for s in samples:
        by_cat[s.category].append(s)

    train_set, val_set, test_set = [], [], []

    for cat, cat_samples in by_cat.items():
        n = len(cat_samples)
        shuffled = cat_samples[:]
        rng.shuffle(shuffled)

        n_train = int(n * train)
        n_val = int(n * val)

        train_set.extend(shuffled[:n_train])
        val_set.extend(shuffled[n_train:n_train + n_val])
        test_set.extend(shuffled[n_train + n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    print(f"[sampler] Split → train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    return train_set, val_set, test_set
