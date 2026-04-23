"""
test_evaluator.py — Test unitari per il valutatore deterministico

Esegui con:  python -m pytest test_evaluator.py -v
"""

import pytest
from evaluator import evaluate, _extract_calls_from_output, _parse_python_call


# ─────────────────────────────────────────────────────────────────────────────
# Test parser
# ─────────────────────────────────────────────────────────────────────────────

class TestParser:

    def test_python_call_kwargs(self):
        result = _parse_python_call("calc_binomial_probability(n=20, k=5, p=0.6)")
        assert result is not None
        assert result["name"] == "calc_binomial_probability"
        assert result["arguments"]["n"] == 20
        assert result["arguments"]["p"] == pytest.approx(0.6)

    def test_python_call_string_args(self):
        result = _parse_python_call("get_weather(city='Rome', unit='celsius')")
        assert result["arguments"]["city"] == "Rome"
        assert result["arguments"]["unit"] == "celsius"

    def test_qwen_tool_call_tag(self):
        raw = '<tool_call>{"name": "search", "arguments": {"query": "hello"}}</tool_call>'
        calls = _extract_calls_from_output(raw)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"]["query"] == "hello"

    def test_json_array_format(self):
        raw = '[{"name": "add", "arguments": {"a": 1, "b": 2}}]'
        calls = _extract_calls_from_output(raw)
        assert len(calls) == 1
        assert calls[0]["name"] == "add"

    def test_json_fenced_block(self):
        raw = "Here is the call:\n```json\n{\"name\": \"greet\", \"arguments\": {\"name\": \"Alice\"}}\n```"
        calls = _extract_calls_from_output(raw)
        assert len(calls) == 1
        assert calls[0]["name"] == "greet"

    def test_no_call_empty(self):
        assert _extract_calls_from_output("") == []
        assert _extract_calls_from_output("I cannot help with that.") == []

    def test_multiple_python_calls(self):
        raw = "func_a(x=1), func_b(y=2)"
        calls = _extract_calls_from_output(raw)
        assert len(calls) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Test evaluate() — label 0 (corretti)
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrect:

    def test_exact_match_ast(self):
        gt = [{"calc_binomial_probability": {"n": [20], "k": [5], "p": [0.6]}}]
        out = "calc_binomial_probability(n=20, k=5, p=0.6)"
        r = evaluate(out, gt)
        assert r.label == 0
        assert r.hallucination_type is None

    def test_type_coercion_string_int(self):
        gt = [{"set_speed": {"speed": [100]}}]
        out = 'set_speed(speed="100")'   # modello usa stringa invece di int
        r = evaluate(out, gt)
        assert r.label == 0

    def test_type_coercion_bool(self):
        gt = [{"toggle": {"enabled": [True]}}]
        out = 'toggle(enabled="true")'
        r = evaluate(out, gt)
        assert r.label == 0

    def test_optional_param_absent(self):
        # Il GT ammette anche il valore "" → parametro opzionale
        gt = [{"search": {"query": ["test"], "limit": [10, ""]}}]
        out = 'search(query="test")'    # omette `limit` (opzionale)
        r = evaluate(out, gt)
        assert r.label == 0

    def test_multiple_acceptable_values(self):
        gt = [{"convert": {"unit": ["celsius", "C", "c"]}}]
        out = 'convert(unit="C")'
        r = evaluate(out, gt)
        assert r.label == 0

    def test_qwen_tool_call_format(self):
        gt = [{"get_weather": {"city": ["Rome"], "unit": ["celsius"]}}]
        out = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "unit": "celsius"}}</tool_call>'
        r = evaluate(out, gt)
        assert r.label == 0

    def test_exec_format_ground_truth(self):
        gt = ["calc_binomial_probability(n=20, k=5, p=0.6)"]
        out = "calc_binomial_probability(n=20, k=5, p=0.6)"
        r = evaluate(out, gt)
        assert r.label == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test evaluate() — label 1 (allucinazioni)
# ─────────────────────────────────────────────────────────────────────────────

class TestHallucinations:

    def test_no_call_made(self):
        gt = [{"get_weather": {"city": ["Rome"]}}]
        r = evaluate("I don't know the weather.", gt)
        assert r.label == 1
        assert r.hallucination_type == "NO_CALL_MADE"

    def test_wrong_function_name(self):
        gt = [{"get_weather": {"city": ["Rome"]}}]
        out = "fetch_weather(city='Rome')"
        r = evaluate(out, gt)
        assert r.label == 1
        assert r.hallucination_type == "WRONG_FUNCTION"

    def test_missing_required_arg(self):
        gt = [{"send_email": {"to": ["alice@ex.com"], "subject": ["Hello"], "body": ["Hi"]}}]
        out = 'send_email(to="alice@ex.com", subject="Hello")'  # manca body
        r = evaluate(out, gt)
        assert r.label == 1
        assert r.hallucination_type == "MISSING_ARGS"

    def test_extra_arg(self):
        gt = [{"add": {"a": [1], "b": [2]}}]
        out = "add(a=1, b=2, c=3)"   # c non esiste
        r = evaluate(out, gt)
        assert r.label == 1
        assert r.hallucination_type == "EXTRA_ARGS"

    def test_wrong_value(self):
        gt = [{"set_speed": {"speed": [100]}}]
        out = "set_speed(speed=999)"
        r = evaluate(out, gt)
        assert r.label == 1
        assert r.hallucination_type == "WRONG_ARG_VALUES"

    def test_parallel_wrong_count(self):
        # GT ha 2 call, modello ne fa 1 → WRONG_CALL_COUNT
        gt = [
            {"book_flight": {"origin": ["NYC"], "dest": ["LAX"]}},
            {"book_hotel": {"city": ["LAX"], "nights": [3]}},
        ]
        out = "book_flight(origin='NYC', dest='LAX')"
        r = evaluate(out, gt, category="parallel")
        assert r.label == 1
        assert r.hallucination_type == "WRONG_CALL_COUNT"

    def test_parallel_all_correct(self):
        gt = [
            {"book_flight": {"origin": ["NYC"], "dest": ["LAX"]}},
            {"book_hotel": {"city": ["LAX"], "nights": [3]}},
        ]
        out = (
            "book_flight(origin='NYC', dest='LAX'), "
            "book_hotel(city='LAX', nights=3)"
        )
        r = evaluate(out, gt, category="parallel")
        assert r.label == 0


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_case_insensitive_function_name(self):
        gt = [{"GetWeather": {"city": ["Rome"]}}]
        out = 'getweather(city="Rome")'
        r = evaluate(out, gt)
        assert r.label == 0

    def test_underscore_vs_hyphen_in_value(self):
        gt = [{"set_mode": {"mode": ["dark_mode"]}}]
        out = 'set_mode(mode="dark-mode")'
        r = evaluate(out, gt)
        assert r.label == 0

    def test_empty_ground_truth_list(self):
        # Nessun GT → non possiamo valutare, ma non deve crashare
        r = evaluate("func(x=1)", [])
        # Con GT vuoto, non c'è match → allucinazione
        assert r.label == 1

    def test_json_arguments_as_string(self):
        # Alcuni modelli serializzano arguments come stringa JSON
        raw = '{"name": "search", "arguments": "{\\"q\\": \\"hello\\"}"}'
        calls = _extract_calls_from_output(raw)
        assert len(calls) == 1
        assert calls[0]["arguments"].get("q") == "hello"
