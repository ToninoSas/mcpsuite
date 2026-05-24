"""
test_sampler.py — Test unitari per exact_sample

Esegui con:  python -m pytest test_sampler.py -v
"""

import pytest
from loader import BFCLSample
from sampler import exact_sample


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def make_sample(id: str, category: str = "simple") -> BFCLSample:
    """Crea un BFCLSample minimale con ground_truth non vuoto."""
    return BFCLSample(
        id=id,
        category=category,
        question=[[{"role": "user", "content": f"Test question {id}"}]],
        functions=[],
        ground_truth=[f"{id}_answer"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test exact_sample
# ─────────────────────────────────────────────────────────────────────────────

class TestExactSample:

    def test_basic(self):
        """Campiona esattamente i conteggi richiesti."""
        corpus = {
            "simple":   [make_sample(f"s{i}") for i in range(100)],
            "multiple": [make_sample(f"m{i}") for i in range(100)],
        }
        out = exact_sample(corpus, {"simple": 10, "multiple": 5}, seed=42)
        assert len(out) == 15

    def test_caps_at_availability(self):
        """Cappa silenziosamente se richiesti più sample di quelli disponibili."""
        corpus = {"simple": [make_sample(f"s{i}") for i in range(5)]}
        out = exact_sample(corpus, {"simple": 100}, seed=42)
        assert len(out) == 5

    def test_skips_zero_counts(self):
        """Categorie con count=0 non vengono incluse."""
        corpus = {
            "simple":   [make_sample(f"s{i}") for i in range(10)],
            "multiple": [make_sample(f"m{i}") for i in range(10)],
        }
        out = exact_sample(corpus, {"simple": 5, "multiple": 0}, seed=42)
        assert len(out) == 5

    def test_rejects_negative(self):
        """Solleva ValueError su valori negativi."""
        corpus = {"simple": [make_sample("s1")]}
        with pytest.raises(ValueError, match="negativi"):
            exact_sample(corpus, {"simple": -1}, seed=42)

    def test_rejects_empty_counts(self):
        """Solleva ValueError su dict vuoto."""
        corpus = {"simple": [make_sample("s1")]}
        with pytest.raises(ValueError, match="vuoto"):
            exact_sample(corpus, {}, seed=42)

    def test_reproducibility(self):
        """Stesso seed → stesso output."""
        corpus = {"simple": [make_sample(f"s{i}") for i in range(100)]}
        out1 = exact_sample(corpus, {"simple": 20}, seed=123)
        out2 = exact_sample(corpus, {"simple": 20}, seed=123)
        assert [s.id for s in out1] == [s.id for s in out2]

    def test_unknown_category_skipped(self):
        """Categoria non nel corpus viene ignorata con warning, non errore."""
        corpus = {"simple": [make_sample("s1")]}
        out = exact_sample(corpus, {"simple": 1, "ghost": 10}, seed=42)
        assert len(out) == 1

    def test_different_seeds_give_different_order(self):
        """Semi diversi producono ordini diversi (test probabilistico su corpus grande)."""
        corpus = {"simple": [make_sample(f"s{i}") for i in range(100)]}
        out1 = exact_sample(corpus, {"simple": 50}, seed=1)
        out2 = exact_sample(corpus, {"simple": 50}, seed=2)
        # Con 50 sample da 100 è praticamente impossibile ottenere lo stesso ordine
        assert [s.id for s in out1] != [s.id for s in out2]

    def test_filter_fn_applied(self):
        """filter_fn esclude i sample che non la passano."""
        corpus = {
            "simple": [make_sample(f"s{i}") for i in range(20)],
        }
        # Tieni solo sample con id pari
        out = exact_sample(
            corpus,
            {"simple": 20},
            seed=42,
            filter_fn=lambda s: int(s.id[1:]) % 2 == 0,
        )
        # Solo 10 sample passano il filtro (s0, s2, ..., s18)
        assert len(out) == 10
        assert all(int(s.id[1:]) % 2 == 0 for s in out)

    def test_exact_counts_respected(self):
        """Verifica che le categorie abbiano esattamente il numero richiesto."""
        corpus = {
            "simple":            [make_sample(f"s{i}",  "simple")            for i in range(200)],
            "multiple":          [make_sample(f"m{i}",  "multiple")          for i in range(200)],
            "parallel":          [make_sample(f"p{i}",  "parallel")          for i in range(200)],
            "parallel_multiple": [make_sample(f"pm{i}", "parallel_multiple") for i in range(200)],
        }
        counts = {"simple": 40, "multiple": 20, "parallel": 20, "parallel_multiple": 20}
        out = exact_sample(corpus, counts, seed=42)
        assert len(out) == 100

        # Verifica breakdown per categoria
        from collections import Counter
        cat_counts = Counter(s.category for s in out)
        assert cat_counts["simple"] == 40
        assert cat_counts["multiple"] == 20
        assert cat_counts["parallel"] == 20
        assert cat_counts["parallel_multiple"] == 20
