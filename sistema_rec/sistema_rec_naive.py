"""
SistemaRecNaive — Top-K ranking naive de productos Amazon.
Fórmula de score:  mean_rating * log(1 + N_reviews)
Ordena TODOS los productos por score usando sorted() en O(n log n)
y luego toma los primeros K.  Esto es peor que SistemaRec1 que usa
un MinHeap de tamaño K en O(n log K).
"""

import json
import math
import os


class SistemaRecNaive:
    """Sistema de recomendación Top-K naive — sort completo O(n log n)."""

    def __init__(
        self,
        category: str = "Electronics",
        k: int = 10,
        data_dir: str = "dataset/amazon_reviews",
    ):
        self.category = category
        self.k = k
        self.data_dir = data_dir

    # ------------------------------------------------------------------ #
    # Ruta al archivo JSONL                                               #
    # ------------------------------------------------------------------ #
    def _get_filepath(self) -> str:
        """Retorna la ruta al archivo JSONL de la categoría."""
        return os.path.join(
            self.data_dir, "raw", "review_categories", f"{self.category}.jsonl"
        )

    # ------------------------------------------------------------------ #
    # Lectura y agregación                                                #
    # ------------------------------------------------------------------ #
    def _load_and_aggregate(self) -> dict:
        """Lee el JSONL línea por línea y agrega ratings por parent_asin.

        Retorna dict[parent_asin] -> [sum_ratings, count].
        """
        filepath = self._get_filepath()
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No se encontró el archivo: {filepath}\n"
                f"Ejecuta descargar_dataset_amazon_reviews.py "
                f"--categorias {self.category} --solo-reviews"
            )

        aggregated: dict[str, list] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                review = json.loads(line)
                parent_asin = review["parent_asin"]
                rating = review["rating"]
                if parent_asin in aggregated:
                    aggregated[parent_asin][0] += rating
                    aggregated[parent_asin][1] += 1
                else:
                    aggregated[parent_asin] = [rating, 1]

        return aggregated

    # ------------------------------------------------------------------ #
    # Fórmula de score                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_score(sum_ratings: float, count: int) -> float:
        """Calcula score = mean_rating * log(1 + N_reviews)."""
        mean_rating = sum_ratings / count
        return mean_rating * math.log(1 + count)

    # ------------------------------------------------------------------ #
    # Top-K naive: sort completo O(n log n)                               #
    # ------------------------------------------------------------------ #
    def top_k(self) -> list[tuple[float, str]]:
        """Retorna los K productos con mayor score en orden descendente.

        Algoritmo naive:
        1. Calcula scores para TODOS los n productos.
        2. Ordena los n productos por score descendente — O(n log n).
        3. Retorna los primeros K.
        """
        aggregated = self._load_and_aggregate()

        scored = [
            (self.compute_score(sum_r, count), parent_asin)
            for parent_asin, (sum_r, count) in aggregated.items()
        ]

        # O(n log n) — ordena TODOS los productos
        scored.sort(reverse=True)

        return scored[: self.k]

    # ------------------------------------------------------------------ #
    # Ejecutar e imprimir                                                 #
    # ------------------------------------------------------------------ #
    def run(self) -> list[tuple[float, str]]:
        """Ejecuta el ranking e imprime los resultados."""
        print(f"Categoría: {self.category}")
        print(f"Top-{self.k} productos (naive sort O(n log n)):\n")

        results = self.top_k()

        for rank, (score, parent_asin) in enumerate(results, 1):
            print(f"  {rank:>3}. {parent_asin}  score={score:.4f}")

        print(f"\nTotal: {len(results)} productos")
        return results
