"""
loader.py — BFCL dataset loader

Struttura attesa su disco:
  data/
    BFCL_v3_simple.json          ← domande (una per riga, JSONL)
    BFCL_v3_multiple.json
    BFCL_v3_parallel.json
    BFCL_v3_multi_turn_base.json
    BFCL_v3_multi_turn_miss_func.json
    BFCL_v3_multi_turn_miss_param.json
    BFCL_v3_multi_turn_long_context.json
    BFCL_v3_multi_turn_composite.json
    possible_answer/
      BFCL_v3_simple.json        ← ground truth (stessa struttura per id)
      ...

Scaricabile con:
  huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset --local-dir data/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Mappatura classe → file func_doc ─────────────────────────────────────────
# Le multi-turn non hanno il campo "function"; gli schemi vengono da
# data/multi_turn_func_doc/{filename}.json (uno schema per riga, JSONL).
# Il nome del file NON corrisponde sempre al nome della classe (es. TwitterAPI
# è in posting_api.json, TravelAPI è in travel_booking.json).

_CLASS_TO_FUNC_DOC: dict[str, str] = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI":           "math_api",
    "MessageAPI":        "message_api",
    "TwitterAPI":        "posting_api",
    "TicketAPI":         "ticket_api",
    "TradingBot":        "trading_bot",
    "TravelAPI":         "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Cache in-process per evitare di rileggere i file a ogni sample
_FUNC_DOC_CACHE: dict[str, list[dict]] = {}


def _load_func_doc_for_classes(
    data_dir: Path,
    class_names: list[str],
) -> list[dict]:
    """
    Restituisce la lista concatenata degli schemi di funzione per le classi
    indicate, leggendo i file JSONL in data/multi_turn_func_doc/.
    """
    func_doc_dir = data_dir / "multi_turn_func_doc"
    schemas: list[dict] = []

    for cls in class_names:
        if cls in _FUNC_DOC_CACHE:
            schemas.extend(_FUNC_DOC_CACHE[cls])
            continue

        fname = _CLASS_TO_FUNC_DOC.get(cls)
        if fname is None:
            print(f"[loader] ⚠  classe sconosciuta in func_doc: {cls}")
            _FUNC_DOC_CACHE[cls] = []
            continue

        fpath = func_doc_dir / f"{fname}.json"
        if not fpath.exists():
            print(f"[loader] ⚠  func_doc non trovato: {fpath}")
            _FUNC_DOC_CACHE[cls] = []
            continue

        class_schemas: list[dict] = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    class_schemas.append(json.loads(line))

        _FUNC_DOC_CACHE[cls] = class_schemas
        schemas.extend(class_schemas)

    return schemas


# ── Categorie supportate ─────────────────────────────────────────────────────

SINGLE_TURN_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple"
]

MULTI_TURN_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "multi_turn_composite",
]

ALL_CATEGORIES = SINGLE_TURN_CATEGORIES + MULTI_TURN_CATEGORIES


# ── Strutture dati ────────────────────────────────────────────────────────────

@dataclass
class BFCLSample:
    id: str
    category: str
    question: list[list[dict]]          # [[{"role": "user", "content": "..."}]]
    functions: list[dict]               # OpenAI function schema
    ground_truth: list[Any]             # lista di call accettabili
    # Campi presenti solo negli exec samples
    execution_result_type: list[str] = field(default_factory=list)
    # Riempito dopo l'inferenza.
    # Single-turn: str; multi-turn: list[str] (uno per turno)
    model_raw_output: str | list[str] | None = None
    label: int | None = None            # 0=corretto, 1=allucinazione
    hallucination_type: str | None = None
    # Hidden state catturato durante l'inferenza (Phase 2).
    # Shape (4096,) float16 numpy array; None se non catturato.
    hidden_vec: Any = None


# ── Funzioni di caricamento ───────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Legge un file JSONL (o JSON array su righe separate)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_gt_index(gt_path: Path) -> dict[str, list[Any]]:
    """
    Costruisce un indice  id → ground_truth  dal file possible_answer.

    Due formati possibili nel dataset BFCL:
      • AST  : {"id": "simple_0", "ground_truth": [{"func": {"arg": [v]}}]}
      • Exec : {"id": "exec_simple_0", "ground_truth": ["func(arg=v)"],
                 "execution_result_type": ["exact_match"]}
    """
    if not gt_path.exists():
        return {}

    index: dict[str, list[Any]] = {}
    exec_types: dict[str, list[str]] = {}

    for rec in _load_jsonl(gt_path):
        rid = rec.get("id", "")
        index[rid] = rec.get("ground_truth", [])
        exec_types[rid] = rec.get("execution_result_type", [])

    return index, exec_types


def load_category(
    data_dir: str | Path,
    category: str,
) -> list[BFCLSample]:
    """
    Carica tutti i sample di una categoria, correlando domande e ground truth.

    Args:
        data_dir: cartella radice (contiene i .json e la sottocartella possible_answer/)
        category: es. "simple", "multi_turn_base"

    Returns:
        Lista di BFCLSample con ground_truth già popolato.
    """
    data_dir = Path(data_dir)
    q_path = data_dir / f"BFCL_v3_{category}.json"
    gt_path = data_dir / "possible_answer" / f"BFCL_v3_{category}.json"

    if not q_path.exists():
        raise FileNotFoundError(f"File domande non trovato: {q_path}")

    questions = _load_jsonl(q_path)
    gt_index, exec_types = _build_gt_index(gt_path)

    samples: list[BFCLSample] = []
    missing_gt = 0

    for rec in questions:
        rid = rec.get("id", "")

        # Recupera ground truth — può essere nel file questions (exec)
        # oppure nel file possible_answer (AST)
        if "ground_truth" in rec:
            gt = rec["ground_truth"]
            exec_result_type = rec.get("execution_result_type", [])
        elif rid in gt_index:
            gt = gt_index[rid]
            exec_result_type = exec_types.get(rid, [])
        else:
            gt = []
            exec_result_type = []
            missing_gt += 1

        # I record multi-turn non hanno il campo "function": gli schemi
        # vengono caricati da multi_turn_func_doc/ tramite involved_classes.
        if category in MULTI_TURN_CATEGORIES:
            functions = _load_func_doc_for_classes(
                data_dir, rec.get("involved_classes", [])
            )
        else:
            functions = rec.get("function", [])

        samples.append(BFCLSample(
            id=rid,
            category=category,
            question=rec.get("question", []),
            functions=functions,
            ground_truth=gt,
            execution_result_type=exec_result_type,
        ))

    if missing_gt:
        print(f"[loader] ⚠  {missing_gt}/{len(samples)} sample senza GT in '{category}'")

    return samples


def load_all(
    data_dir: str | Path,
    categories: list[str] | None = None,
) -> dict[str, list[BFCLSample]]:
    """
    Carica tutte le categorie disponibili.

    Returns:
        dict  category_name → [BFCLSample, ...]
    """
    data_dir = Path(data_dir)
    cats = categories or ALL_CATEGORIES
    result: dict[str, list[BFCLSample]] = {}

    for cat in cats:
        q_path = data_dir / f"BFCL_v3_{cat}.json"
        if not q_path.exists():
            print(f"[loader] skip '{cat}' (file non trovato)")
            continue
        try:
            samples = load_category(data_dir, cat)
            result[cat] = samples
            print(f"[loader] ✓  {cat:40s}  {len(samples):5d} sample")
        except Exception as exc:
            print(f"[loader] ✗  {cat}: {exc}")

    return result
