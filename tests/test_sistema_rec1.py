"""Tests para MinHeap y SistemaRec1."""

import math
import os

import pytest

from estructuras_datos.heap import MinHeap
from sistema_rec.sistema_rec1 import SistemaRec1


# ====================================================================== #
# Tests MinHeap                                                           #
# ====================================================================== #
class TestMinHeap:
    """Tests unitarios para MinHeap (no requieren dataset)."""

    def test_insert_and_extract_min(self):
        h = MinHeap()
        h.min_heap_insert((3, "c"))
        h.min_heap_insert((1, "a"))
        h.min_heap_insert((2, "b"))
        assert h.heap_extract_min() == (1, "a")
        assert h.heap_extract_min() == (2, "b")
        assert h.heap_extract_min() == (3, "c")

    def test_extract_min_order(self):
        """Extraer todos los elementos debe dar orden ascendente."""
        h = MinHeap()
        values = [(5, "e"), (3, "c"), (8, "h"), (1, "a"), (4, "d")]
        for v in values:
            h.min_heap_insert(v)

        result = []
        while len(h) > 0:
            result.append(h.heap_extract_min())

        scores = [s for s, _ in result]
        assert scores == sorted(scores)

    def test_build_min_heap(self):
        data = [(5, "e"), (3, "c"), (8, "h"), (1, "a"), (4, "d")]
        h = MinHeap()
        h.build_min_heap(data)
        assert h.heap_minimum() == (1, "a")
        assert len(h) == 5

    def test_extract_min_empty_raises(self):
        h = MinHeap()
        with pytest.raises(IndexError):
            h.heap_extract_min()

    def test_minimum_empty_raises(self):
        h = MinHeap()
        with pytest.raises(IndexError):
            h.heap_minimum()

    def test_decrease_key(self):
        h = MinHeap()
        h.min_heap_insert((10, "x"))
        h.min_heap_insert((20, "y"))
        h.min_heap_insert((30, "z"))
        # Disminuir la clave del último insertado (posición puede variar)
        # Insertamos un valor conocido y lo disminuimos
        h.heap_decrease_key(0, (5, "x"))
        assert h.heap_minimum() == (5, "x")

    def test_decrease_key_invalid_raises(self):
        h = MinHeap()
        h.min_heap_insert((10, "x"))
        with pytest.raises(ValueError):
            h.heap_decrease_key(0, (20, "bigger"))

    def test_len(self):
        h = MinHeap()
        assert len(h) == 0
        h.min_heap_insert((1, "a"))
        assert len(h) == 1
        h.min_heap_insert((2, "b"))
        assert len(h) == 2
        h.heap_extract_min()
        assert len(h) == 1


# ====================================================================== #
# Tests SistemaRec1 — fórmula de score                                   #
# ====================================================================== #
class TestSistemaRec1Score:
    """Tests de la fórmula de score (no requieren dataset)."""

    def test_compute_score_formula(self):
        # 5 reviews con rating promedio 4.0
        # score = 4.0 * log(1 + 5) = 4.0 * log(6)
        score = SistemaRec1.compute_score(20.0, 5)
        expected = 4.0 * math.log(6)
        assert math.isclose(score, expected, rel_tol=1e-9)

    def test_score_increases_with_reviews(self):
        """A mayor cantidad de reviews (mismo rating), mayor score."""
        score_few = SistemaRec1.compute_score(4.0 * 10, 10)
        score_many = SistemaRec1.compute_score(4.0 * 100, 100)
        assert score_many > score_few

    def test_score_increases_with_rating(self):
        """A mayor rating promedio (misma cantidad de reviews), mayor score."""
        score_low = SistemaRec1.compute_score(2.0 * 50, 50)
        score_high = SistemaRec1.compute_score(5.0 * 50, 50)
        assert score_high > score_low

    def test_single_review(self):
        score = SistemaRec1.compute_score(5.0, 1)
        expected = 5.0 * math.log(2)
        assert math.isclose(score, expected, rel_tol=1e-9)


# ====================================================================== #
# Tests integración (requieren dataset descargado)                        #
# ====================================================================== #
class TestSistemaRec1Integration:
    """Tests de integración que leen el dataset real."""

    @pytest.fixture()
    def sistema(self, request):
        category = request.config.getoption("--category")
        top_k = request.config.getoption("--top-k")
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "dataset",
            "amazon_reviews",
        )
        return SistemaRec1(category=category, k=top_k, data_dir=data_dir)

    @pytest.fixture()
    def dataset_available(self, sistema):
        filepath = sistema._get_filepath()
        if not os.path.exists(filepath):
            pytest.skip(f"Dataset no disponible: {filepath}")

    def test_returns_k_results(self, sistema, dataset_available):
        results = sistema.top_k()
        assert len(results) == sistema.k

    def test_descending_order(self, sistema, dataset_available):
        results = sistema.top_k()
        scores = [score for score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_positive_scores(self, sistema, dataset_available):
        results = sistema.top_k()
        for score, _ in results:
            assert score > 0

    def test_file_not_found(self):
        sistema = SistemaRec1(category="Categoria_Inexistente", k=5)
        with pytest.raises(FileNotFoundError):
            sistema.top_k()
