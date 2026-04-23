"""
evaluator.py — Valutazione deterministica delle tool call (no LLM-as-judge)

Strategia:
  1. Parsing dell'output del modello (due formati: Python call-string oppure JSON)
  2. Normalizzazione  (snake_case, type coercion, keyword matching)
  3. Confronto AST contro il ground truth del BFCL

Tassonomia delle allucinazioni (label=1):
  • INVALID_FORMAT      output non parseable come function call
  • WRONG_FUNCTION      nome funzione sbagliato
  • WRONG_ARG_NAMES     parametri con nomi diversi dal GT
  • WRONG_ARG_VALUES    parametri giusti ma valori fuori dal set ammesso
  • MISSING_ARGS        argomenti required mancanti
  • EXTRA_ARGS          argomenti extra non previsti
  • WRONG_CALL_COUNT    numero di chiamate diverso (parallel / multi-turn)
  • NO_CALL_MADE        il modello non ha prodotto nessuna chiamata
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Struttura del risultato
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    label: int                          # 0=corretto, 1=allucinazione
    hallucination_type: str | None      # None se label==0
    details: dict[str, Any]            # info diagnostiche


# ─────────────────────────────────────────────────────────────────────────────
# Parser dell'output del modello
# ─────────────────────────────────────────────────────────────────────────────

def _extract_calls_from_output(raw: str) -> list[dict]:
    """
    Tenta di estrarre una o più tool call dall'output grezzo del modello.

    Supporta:
      • Python call-string:  "func(a=1, b='x')"
      • JSON OpenAI format:  [{"name": "func", "arguments": {...}}]
      • JSON single:         {"name": "func", "arguments": {...}}
      • Tool-call block:     ```json\n[...]\n```
      • Qwen native format:  <tool_call>{"name": ..., "arguments": ...}</tool_call>
    """
    calls: list[dict] = []

    # ── 1. Qwen native <tool_call> tag ────────────────────────────────────────
    tc_pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL,
    )
    for m in tc_pattern.finditer(raw):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name") or obj.get("function", {}).get("name", "")
            args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                calls.append({"name": name, "arguments": args})
        except Exception:
            pass
    if calls:
        return calls

    # ── 2. JSON fenced block ──────────────────────────────────────────────────
    fence = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", raw)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return _normalize_json_calls(obj)
        except Exception:
            pass

    # ── 3. JSON inline ────────────────────────────────────────────────────────
    # Cerca il primo '[' o '{' che apre una struttura valida
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = raw.find(start_char)
        if idx == -1:
            continue
        # Trova la chiusura bilanciata
        depth, end_idx = 0, -1
        for i, ch in enumerate(raw[idx:], idx):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx == -1:
            continue
        try:
            obj = json.loads(raw[idx:end_idx + 1])
            parsed = _normalize_json_calls(obj)
            if parsed:
                return parsed
        except Exception:
            pass

    # ── 4. Python call-string — balanced-paren extractor ─────────────────────
    # Trova ogni occorrenza di  word(  e poi cerca la parentesi di chiusura
    # bilanciando le annidamenti, così gestisce anche argomenti con funzioni.
    func_start = re.compile(r"\b([A-Za-z_]\w*(?:\.\w+)*)\s*\(")
    for m in func_start.finditer(raw):
        name_candidate = m.group(1)
        open_pos = m.end() - 1   # posizione del '('
        # Bilancia le parentesi
        depth = 0
        close_pos = -1
        for i in range(open_pos, len(raw)):
            if raw[i] == "(":
                depth += 1
            elif raw[i] == ")":
                depth -= 1
                if depth == 0:
                    close_pos = i
                    break
        if close_pos == -1:
            continue
        call_str = raw[m.start():close_pos + 1]
        parsed = _parse_python_call(call_str)
        if parsed:
            # Evita duplicati (stesso nome e args già presenti)
            already = any(
                c["name"] == parsed["name"] and c["arguments"] == parsed["arguments"]
                for c in calls
            )
            if not already:
                calls.append(parsed)

    return calls


def _normalize_json_calls(obj: Any) -> list[dict]:
    """Normalizza JSON in formato lista di {"name": ..., "arguments": {...}}."""
    results = []
    items = obj if isinstance(obj, list) else [obj]
    for item in items:
        if isinstance(item, dict):
            # OpenAI tool_call format
            if "function" in item and isinstance(item["function"], dict):
                item = item["function"]
            name = item.get("name") or item.get("tool_name", "")
            args = item.get("arguments") or item.get("parameters") or item.get("args") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if name:
                results.append({"name": str(name), "arguments": args})
    return results


def _parse_python_call(expr: str) -> dict | None:
    """
    Parsa  func(a=1, b='x')  → {"name": "func", "arguments": {"a": 1, "b": "x"}}
    """
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        # Prova a correggere virgolette miste
        try:
            tree = ast.parse(expr.replace("'", '"'), mode="eval")
        except SyntaxError:
            return None

    if not isinstance(tree.body, ast.Call):
        return None

    call = tree.body
    # Funzione name (gestisce anche modulo.func)
    if isinstance(call.func, ast.Attribute):
        name = f"{ast.unparse(call.func.value)}.{call.func.attr}"
    elif isinstance(call.func, ast.Name):
        name = call.func.id
    else:
        return None

    # Keyword arguments
    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue  # **kwargs unpacking — ignora
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            kwargs[kw.arg] = ast.unparse(kw.value)

    # Positional arguments (alcuni modelli li usano)
    for i, arg in enumerate(call.args):
        try:
            kwargs[f"__pos_{i}"] = ast.literal_eval(arg)
        except Exception:
            kwargs[f"__pos_{i}"] = ast.unparse(arg)

    return {"name": name, "arguments": kwargs}


# ─────────────────────────────────────────────────────────────────────────────
# Normalizzazione dei valori
# ─────────────────────────────────────────────────────────────────────────────

def _coerce(value: Any) -> Any:
    """
    Type coercion per confronto flessibile:
      "20"  →  20
      "3.14" → 3.14
      "true" → True
    """
    if isinstance(value, str):
        low = value.lower().strip()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _values_match(predicted: Any, acceptable: list[Any]) -> bool:
    """
    Il GT BFCL contiene liste di valori accettabili per ogni parametro.
    Controlla se `predicted` corrisponde ad almeno uno.
    """
    pred_coerced = _coerce(predicted)
    for acc in acceptable:
        acc_coerced = _coerce(acc)
        # Match diretto
        if pred_coerced == acc_coerced:
            return True
        # Match su stringa normalizzata (lowercase, underscore)
        if (isinstance(pred_coerced, str) and isinstance(acc_coerced, str) and
                pred_coerced.lower().replace("-", "_") == acc_coerced.lower().replace("-", "_")):
            return True
        # Se il valore accettabile è stringa vuota → parametro opzionale accettato
        if acc == "" or acc is None:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Confronto singola call con GT
# ─────────────────────────────────────────────────────────────────────────────

def _compare_single_call(
    predicted: dict,       # {"name": str, "arguments": dict}
    gt_entry: Any,         # dalla lista ground_truth
) -> dict[str, Any]:
    """
    Confronta una singola call predetta con un'entry del GT.

    GT format AST:
      {"func_name": {"param_a": [val1, val2], "param_b": [val]}}

    GT format exec (stringa Python):
      "func_name(param_a=val, param_b=val)"

    Returns:
      dict con chiavi: name_match, args_match, missing_args, extra_args,
                       wrong_values, is_correct
    """
    result = {
        "name_match": False,
        "args_match": False,
        "missing_args": [],
        "extra_args": [],
        "wrong_values": {},
        "is_correct": False,
    }

    pred_name = predicted.get("name", "").strip()
    pred_args = predicted.get("arguments", {})

    # ── GT stringa (exec format) ──────────────────────────────────────────────
    if isinstance(gt_entry, str):
        gt_parsed = _parse_python_call(gt_entry)
        if gt_parsed is None:
            return result
        gt_name = gt_parsed["name"]
        # Converti args in formato {param: [value]} per uniformità
        gt_args_acceptable = {k: [v] for k, v in gt_parsed["arguments"].items()}

    # ── GT dict (AST format) ─────────────────────────────────────────────────
    elif isinstance(gt_entry, dict):
        if len(gt_entry) != 1:
            return result
        gt_name = list(gt_entry.keys())[0]
        gt_args_acceptable: dict[str, list] = gt_entry[gt_name]

    else:
        return result

    # 1. Controlla il nome della funzione
    if pred_name.lower() != gt_name.lower():
        result["name_match"] = False
        return result
    result["name_match"] = True

    # 2. Controlla gli argomenti
    pred_keys = set(pred_args.keys()) - {k for k in pred_args if k.startswith("__pos_")}
    gt_keys   = set(gt_args_acceptable.keys())

    # Argomenti mancanti (required: non sono nella lista con "" come valore accettabile)
    for k in gt_keys:
        acceptable_vals = gt_args_acceptable[k]
        # Se "" o None è tra i valori accettabili → parametro opzionale
        is_optional = any(v == "" or v is None for v in acceptable_vals)
        if k not in pred_keys and not is_optional:
            result["missing_args"].append(k)

    # Argomenti extra (non presenti nel GT)
    for k in pred_keys:
        if k not in gt_keys:
            result["extra_args"].append(k)

    # Valori sbagliati
    for k in pred_keys & gt_keys:
        if not _values_match(pred_args[k], gt_args_acceptable[k]):
            result["wrong_values"][k] = {
                "predicted": pred_args[k],
                "acceptable": gt_args_acceptable[k],
            }

    result["args_match"] = (
        not result["missing_args"] and
        not result["extra_args"] and
        not result["wrong_values"]
    )

    result["is_correct"] = result["name_match"] and result["args_match"]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point principale
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_raw_output: str,
    ground_truth: list[Any],
    category: str = "simple",
) -> EvalResult:
    """
    Valuta l'output del modello contro il ground truth BFCL.

    Args:
        model_raw_output: output grezzo del modello (stringa)
        ground_truth:     lista di entry GT dal file possible_answer
        category:         categoria BFCL del sample

    Returns:
        EvalResult con label (0/1) e diagnostica dettagliata.
    """
    details: dict[str, Any] = {"category": category, "raw_output_len": len(model_raw_output)}

    # ── 1. Parsing dell'output ────────────────────────────────────────────────
    predicted_calls = _extract_calls_from_output(model_raw_output)
    details["predicted_calls"] = predicted_calls
    details["n_predicted"] = len(predicted_calls)

    if not predicted_calls:
        # Modello non ha prodotto nessuna call
        details["parse_error"] = True
        return EvalResult(
            label=1,
            hallucination_type="NO_CALL_MADE",
            details=details,
        )

    # ── 2. Normalizza il GT in lista di "acceptable sets" ────────────────────
    # Il GT può essere:
    #   [{"func": {"arg": [v]}}, ...]          → AST, ogni entry è una call accettabile
    #   ["func(arg=v)", ...]                   → exec, stessa logica
    # Per le categorie parallel/multiple il GT contiene più call da matchare
    # Per multi-turn ogni turno ha il proprio GT

    # Calcola quante call ci si aspetta
    expected_n = len(ground_truth)
    details["n_expected"] = expected_n

    # ── 3. Verifica numero di call (per parallel/multiple) ───────────────────
    is_parallel = "parallel" in category.lower()
    is_multi    = "multiple" in category.lower()

    if (is_parallel or is_multi) and len(predicted_calls) != expected_n:
        details["count_mismatch"] = {
            "expected": expected_n,
            "predicted": len(predicted_calls),
        }
        return EvalResult(
            label=1,
            hallucination_type="WRONG_CALL_COUNT",
            details=details,
        )

    # ── 4. Confronto call per call ────────────────────────────────────────────
    # Per categorie non-parallel: basta che la PRIMA call predetta
    # corrisponda ad almeno una entry GT.
    # Per parallel/multiple: ogni call predetta deve matchare una GT entry
    # (matching bipartito greedy).

    comparison_results = []

    if is_parallel or is_multi:
        gt_remaining = list(enumerate(ground_truth))
        for pred_call in predicted_calls:
            best = None
            best_idx = -1
            for idx, gt_entry in gt_remaining:
                cmp = _compare_single_call(pred_call, gt_entry)
                if cmp["is_correct"]:
                    best = cmp
                    best_idx = idx
                    break
            if best_idx >= 0:
                comparison_results.append({"matched": True, **best})
                gt_remaining = [(i, e) for i, e in gt_remaining if i != best_idx]
            else:
                # Fallback: salva il miglior partial match per diagnostica
                partials = [_compare_single_call(pred_call, e) for _, e in gt_remaining]
                comparison_results.append({
                    "matched": False,
                    "partial": max(partials, key=lambda x: sum([
                        x["name_match"], x["args_match"]
                    ])) if partials else {},
                })
    else:
        # Single call: usa la prima call predetta, confronta contro tutti i GT
        pred_call = predicted_calls[0]
        matched = False
        best_partial = None
        for gt_entry in ground_truth:
            cmp = _compare_single_call(pred_call, gt_entry)
            if cmp["is_correct"]:
                matched = True
                comparison_results.append({"matched": True, **cmp})
                break
            if best_partial is None or (cmp["name_match"] and not best_partial.get("name_match")):
                best_partial = cmp
        if not matched:
            comparison_results.append({"matched": False, "partial": best_partial})

    details["comparisons"] = comparison_results

    # ── 5. Determina label e tipo di allucinazione ────────────────────────────
    all_matched = all(r.get("matched", False) for r in comparison_results)

    if all_matched:
        return EvalResult(label=0, hallucination_type=None, details=details)

    # Classifica il tipo di errore dal miglior match parziale
    first_fail = next(
        (r for r in comparison_results if not r.get("matched")),
        {}
    )
    partial = first_fail.get("partial", {}) or first_fail

    if not partial.get("name_match", False):
        htype = "WRONG_FUNCTION"
    elif partial.get("missing_args"):
        htype = "MISSING_ARGS"
    elif partial.get("extra_args"):
        htype = "EXTRA_ARGS"
    elif partial.get("wrong_values"):
        htype = "WRONG_ARG_VALUES"
    else:
        htype = "WRONG_ARG_NAMES"

    return EvalResult(label=1, hallucination_type=htype, details=details)
