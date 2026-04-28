"""
pipeline.py — Orchestratore Fase 1

Flusso completo:
  1. Carica il corpus BFCL da disco
  2. Campionamento proporzionale
  3. Inferenza con Qwen 4-bit
  4. Valutazione deterministica (AST)
  5. Salvataggio del dataset labellato in JSONL

Uso:
  python pipeline.py \
    --data_dir  ./data \
    --output    ./outputs/labeled_dataset.jsonl \
    --total     2000 \
    --backend   transformers \
    --model     Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path

from loader    import load_all, BFCLSample, MULTI_TURN_CATEGORIES
from sampler   import proportional_sample, split_train_val_test, DEFAULT_WEIGHTS
from runner    import RunnerConfig, build_runner, run_inference_on_samples, run_inference_parallel
from evaluator import evaluate, evaluate_multi_turn


# ─────────────────────────────────────────────────────────────────────────────
# Serializzazione
# ─────────────────────────────────────────────────────────────────────────────

def sample_to_dict(s: BFCLSample, eval_result) -> dict:
    """Converte un sample labellato in dizionario JSON-serializzabile."""
    return {
        "id":                    s.id,
        "category":              s.category,
        "question":              s.question,
        "functions":             s.functions,
        "ground_truth":          s.ground_truth,
        "execution_result_type": s.execution_result_type,
        "model_raw_output":      s.model_raw_output or "",
        "label":                 eval_result.label,
        "hallucination_type":    eval_result.hallucination_type,
        "eval_details":          eval_result.details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_dir: str,
    output_path: str,
    total_samples: int,
    runner_config: RunnerConfig,
    seed: int = 42,
    skip_inference: bool = False,
    num_gpus: int = 1,
    weights: dict[str, float] | None = None,
    capture_activations: bool = False,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.time()
    timings: dict[str, float] = {}

    # ── 1. Caricamento ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("FASE 1.1 — Caricamento corpus BFCL")
    print("═" * 60)
    t0 = time.time()
    corpus = load_all(data_dir)
    timings["load"] = time.time() - t0

    if not corpus:
        print("[pipeline] ERRORE: nessun file trovato in", data_dir)
        sys.exit(1)

    total_available = sum(len(v) for v in corpus.values())
    print(f"\n[pipeline] Totale sample disponibili: {total_available}")

    # ── 2. Campionamento ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("FASE 1.2 — Campionamento proporzionale")
    print("═" * 60)

    # Esclude sample senza ground truth
    t0 = time.time()
    samples = proportional_sample(
        corpus,
        total=total_samples,
        weights=weights,
        seed=seed,
        filter_fn=lambda s: len(s.ground_truth) > 0,
    )
    timings["sample"] = time.time() - t0

    # ── 3. Inferenza ──────────────────────────────────────────────────────────
    t0 = time.time()
    if not skip_inference:
        print("\n" + "═" * 60)
        print("FASE 1.3 — Inferenza Qwen 4-bit")
        print("═" * 60)
        if capture_activations:
            print("[pipeline] capture_activations=True — hidden state catturato durante l'inferenza")
        if num_gpus > 1:
            print(f"[pipeline] Modalità multi-GPU: {num_gpus} GPU in data-parallel")
            samples = run_inference_parallel(
                samples, runner_config, num_gpus=num_gpus,
                capture_activations=capture_activations,
            )
        else:
            runner = build_runner(runner_config)
            samples = run_inference_on_samples(
                samples, runner, capture_activations=capture_activations
            )
    else:
        print("[pipeline] skip_inference=True — salto l'inferenza")
    timings["inference"] = time.time() - t0

    # ── 4. Valutazione ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("FASE 1.4 — Valutazione deterministica")
    print("═" * 60)

    t0 = time.time()
    labeled: list[dict] = []
    label_counts = Counter()
    htype_counts = Counter()
    cat_stats: dict[str, dict] = {}

    for sample in samples:
        is_multi = sample.category in MULTI_TURN_CATEGORIES
        raw = sample.model_raw_output

        if not raw:
            from evaluator import EvalResult
            result = EvalResult(
                label=1,
                hallucination_type="INFERENCE_ERROR",
                details={"error": "empty output"},
            )
        elif is_multi:
            # raw è list[str] (uno per turno); ground_truth è list[list[str]]
            turn_outputs = raw if isinstance(raw, list) else [raw]
            result = evaluate_multi_turn(
                turn_outputs=turn_outputs,
                per_turn_gt=sample.ground_truth,
                category=sample.category,
            )
        else:
            result = evaluate(
                model_raw_output=raw,
                ground_truth=sample.ground_truth,
                category=sample.category,
            )

        label_counts[result.label] += 1
        if result.hallucination_type:
            htype_counts[result.hallucination_type] += 1

        # Statistiche per categoria
        cat = sample.category
        if cat not in cat_stats:
            cat_stats[cat] = {"n": 0, "n_correct": 0, "n_halluc": 0, "htypes": Counter()}
        cat_stats[cat]["n"] += 1
        if result.label == 0:
            cat_stats[cat]["n_correct"] += 1
        else:
            cat_stats[cat]["n_halluc"] += 1
            if result.hallucination_type:
                cat_stats[cat]["htypes"][result.hallucination_type] += 1

        labeled.append(sample_to_dict(sample, result))

    timings["eval"] = time.time() - t0

    # ── 5. Salvataggio ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("FASE 1.5 — Salvataggio dataset")
    print("═" * 60)

    t0 = time.time()
    # Dataset completo
    with open(out, "w", encoding="utf-8") as f:
        for record in labeled:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Split train/val/test
    # Ricostruiamo i BFCLSample con i label per lo split
    labeled_samples = []
    for rec in labeled:
        s = BFCLSample(
            id=rec["id"],
            category=rec["category"],
            question=rec["question"],
            functions=rec["functions"],
            ground_truth=rec["ground_truth"],
        )
        s.model_raw_output = rec["model_raw_output"]
        s.label = rec["label"]
        labeled_samples.append(s)

    train_set, val_set, test_set = split_train_val_test(labeled_samples, seed=seed)

    splits_dir = out.parent / "splits"
    splits_dir.mkdir(exist_ok=True)

    for split_name, split_data in [
        ("train", train_set),
        ("val",   val_set),
        ("test",  test_set),
    ]:
        split_path = splits_dir / f"{split_name}.jsonl"
        ids = {s.id for s in split_data}
        with open(split_path, "w", encoding="utf-8") as f:
            for rec in labeled:
                if rec["id"] in ids:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    timings["save"] = time.time() - t0

    # ── 6. Salvataggio activations (se richiesto) ────────────────────────────
    t0 = time.time()
    if capture_activations and not skip_inference:
        import numpy as np

        print("\n" + "═" * 60)
        print("FASE 1.6 — Salvataggio activations")
        print("═" * 60)

        # Lookup: id → sample originale (con hidden_vec) e id → record labellato
        sample_by_id  = {s.id: s for s in samples}
        labeled_by_id = {r["id"]: r for r in labeled}

        acts_base = out.parent / "activations"
        for split_name, split_data in [
            ("train", train_set),
            ("val",   val_set),
            ("test",  test_set),
        ]:
            split_dir = acts_base / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            X, y, meta = [], [], []
            missing = 0

            for s in split_data:
                orig = sample_by_id.get(s.id)
                vec  = getattr(orig, "hidden_vec", None) if orig else None
                if vec is None:
                    missing += 1
                    continue
                rec = labeled_by_id[s.id]
                X.append(vec)
                y.append(rec["label"])
                meta.append({
                    "id":                 rec["id"],
                    "category":           rec["category"],
                    "hallucination_type": rec["hallucination_type"],
                })

            if missing:
                print(f"[pipeline] ⚠  {missing} sample senza hidden_vec in '{split_name}'")

            if X:
                np.save(split_dir / "X.npy", np.stack(X).astype(np.float16))
                np.save(split_dir / "y.npy", np.array(y, dtype=np.int8))
                with open(split_dir / "meta.jsonl", "w", encoding="utf-8") as f:
                    for m in meta:
                        f.write(json.dumps(m, ensure_ascii=False) + "\n")
                print(f"[pipeline]   {split_name}: {len(X)} vettori → {split_dir}/")
            else:
                print(f"[pipeline] ⚠  nessuna activation salvata per '{split_name}'")
    timings["activations"] = time.time() - t0

    # ── Report finale ─────────────────────────────────────────────────────────
    timings["total"] = time.time() - t_pipeline_start

    total_labeled = len(labeled)
    n_correct     = label_counts[0]
    n_halluc      = label_counts[1]
    accuracy      = n_correct / total_labeled if total_labeled else 0.0
    halluc_rate   = n_halluc  / total_labeled if total_labeled else 0.0

    W = 62
    print(f"\n{'═' * W}")
    print(f"  RISULTATI — {total_labeled} sample")
    print(f"{'═' * W}")
    print(f"  Accuracy  (corretti/totale)  : {accuracy*100:>6.1f}%  ({n_correct}/{total_labeled})")
    print(f"  Hallucination rate           : {halluc_rate*100:>6.1f}%  ({n_halluc}/{total_labeled})")

    if n_halluc:
        print(f"\n  Distribuzione tipi allucinazione:")
        for htype, count in sorted(htype_counts.items(), key=lambda x: -x[1]):
            print(f"    {htype:<25} {count:>5}  ({count/n_halluc*100:.1f}%)")

    print(f"\n  Breakdown per categoria:")
    print(f"  {'categoria':<30} {'N':>5}  {'acc':>7}  {'hall%':>6}  top halluc. type")
    print(f"  {'─'*30} {'─'*5}  {'─'*7}  {'─'*6}  {'─'*20}")
    for cat, st in sorted(cat_stats.items()):
        acc_cat   = st["n_correct"] / st["n"] if st["n"] else 0
        hall_cat  = st["n_halluc"]  / st["n"] if st["n"] else 0
        top_htype = max(st["htypes"], key=st["htypes"].get) if st["htypes"] else "—"
        print(f"  {cat:<30} {st['n']:>5}  {acc_cat*100:>6.1f}%  {hall_cat*100:>5.1f}%  {top_htype}")

    print(f"\n  Tempi di esecuzione:")
    phase_labels = [
        ("load",        "caricamento"),
        ("sample",      "campionamento"),
        ("inference",   "inferenza"),
        ("eval",        "valutazione"),
        ("save",        "salvataggio"),
        ("activations", "activations"),
    ]
    for key, label in phase_labels:
        t_sec = timings.get(key, 0.0)
        if key == "inference" and not skip_inference and t_sec > 0:
            tput = f"  ({len(samples)/t_sec:.2f} sample/s)"
        else:
            tput = ""
        print(f"    {label:<16}: {t_sec:>8.1f}s{tput}")
    t_tot = timings["total"]
    print(f"    {'─'*36}")
    print(f"    {'totale':<16}: {t_tot:>8.1f}s  ({t_tot/60:.1f} min)")

    print(f"\n  Output:")
    print(f"    dataset  : {out}")
    print(f"    splits   : {splits_dir}/")
    print(f"{'═' * W}\n")

    # Avviso sbilanciamento
    if halluc_rate < 0.20 or halluc_rate > 0.80:
        print(
            f"[pipeline] ⚠  Dataset sbilanciato ({halluc_rate*100:.0f}% allucinazioni). "
            "Considera class_weight o oversampling nel training."
        )

    # ── Salvataggio metrics.json ──────────────────────────────────────────────
    metrics = {
        "total":             total_labeled,
        "n_correct":         n_correct,
        "n_halluc":          n_halluc,
        "accuracy":          round(accuracy,    4),
        "hallucination_rate": round(halluc_rate, 4),
        "hallucination_types": dict(htype_counts),
        "per_category": {
            cat: {
                "n":                st["n"],
                "n_correct":        st["n_correct"],
                "n_halluc":         st["n_halluc"],
                "accuracy":         round(st["n_correct"] / st["n"], 4) if st["n"] else 0,
                "hallucination_rate": round(st["n_halluc"] / st["n"], 4) if st["n"] else 0,
                "hallucination_types": dict(st["htypes"]),
            }
            for cat, st in cat_stats.items()
        },
        "splits": {
            "train": len(train_set),
            "val":   len(val_set),
            "test":  len(test_set),
        },
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
        "inference_samples_per_sec": (
            round(len(samples) / timings["inference"], 3)
            if not skip_inference and timings.get("inference", 0) > 0
            else None
        ),
    }
    metrics_path = out.parent / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[pipeline] Metriche salvate in: {metrics_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BFCL Hallucination Probe — Fase 1")
    p.add_argument("--data_dir",        default="./data",           help="Cartella BFCL")
    p.add_argument("--output",          default="./outputs/labeled_dataset.jsonl")
    p.add_argument("--total",           type=int, default=2000,     help="Sample totali")
    p.add_argument("--model",           default="Qwen/Qwen3.5-9B")
    p.add_argument("--backend",         default="transformers",     choices=["transformers", "llama_cpp"])
    p.add_argument("--max_new_tokens",  type=int, default=512)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--num_gpus",        type=int, default=1,
                   help="GPU da usare in data-parallel (default 1; usa 2 per dual-T4 Kaggle)")
    p.add_argument("--no_multi_turn",   action="store_true",
                   help="Azzera i pesi delle categorie multi-turn (test/debug veloce)")
    p.add_argument("--weights",         type=str, default=None,
                   help="Pesi per categoria in formato JSON, es: '{\"simple\":0.5,\"multiple\":0.5}'")
    p.add_argument("--skip_inference",  action="store_true",
                   help="Salta l'inferenza (utile per ri-valutare output già generati)")
    p.add_argument("--capture_activations", action="store_true",
                   help="Cattura hidden state durante l'inferenza e salva X.npy/y.npy/meta.jsonl in outputs/activations/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Costruisce il dizionario dei pesi
    weights: dict[str, float] | None = None
    if args.weights:
        weights = json.loads(args.weights)
    elif args.no_multi_turn:
        weights = {**DEFAULT_WEIGHTS, **{cat: 0.0 for cat in MULTI_TURN_CATEGORIES}}

    runner_cfg = RunnerConfig(
        model_name_or_path=args.model,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
    )

    run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        total_samples=args.total,
        runner_config=runner_cfg,
        seed=args.seed,
        skip_inference=args.skip_inference,
        num_gpus=args.num_gpus,
        weights=weights,
        capture_activations=args.capture_activations,
    )
