"""
SistemaRec1 — Top-K ranking de productos Amazon usando MinHeap.

Fórmula de score:  mean_rating * log(1 + N_reviews)

Usa un MinHeap de tamaño K para obtener los K mejores productos
en una sola pasada sobre los datos agregados, con complejidad
O(n log K) donde n es la cantidad de productos únicos.
"""

import json
import math
import os

from estructuras_datos.heap import MinHeap


class SistemaRec1:
    """Sistema de recomendación Top-K basado en MinHeap."""

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
        """Lee el JSONL y agrega ratings por parent_asin.
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
    # Top-K con MinHeap                                                   #
    # ------------------------------------------------------------------ #
    def top_k(self) -> list[tuple[float, str]]:
        """Retorna los K productos con mayor score en orden descendente.

        Algoritmo:
        - Mantiene un MinHeap de tamaño K.
        - Para cada producto:
          - Si el heap tiene menos de K elementos: insertar.
          - Si el score supera al mínimo del heap: extraer mínimo e insertar.
        - Al final, extraer todos y revertir para orden descendente.
        """
        aggregated = self._load_and_aggregate()
        heap = MinHeap()

        for parent_asin, (sum_ratings, count) in aggregated.items():
            score = self.compute_score(sum_ratings, count)

            if len(heap) < self.k:
                heap.min_heap_insert((score, parent_asin))
            elif score > heap.heap_minimum()[0]:
                heap.heap_extract_min()
                heap.min_heap_insert((score, parent_asin))

        # Extraer todos en orden ascendente, luego invertir
        result = []
        while len(heap) > 0:
            result.append(heap.heap_extract_min())
        result.reverse()
        return result

    # ------------------------------------------------------------------ #
    # Ejecutar e imprimir                                                 #
    # ------------------------------------------------------------------ #
    def run(self) -> list[tuple[float, str]]:
        """Ejecuta el ranking e imprime los resultados."""
        print(f"Categoría: {self.category}")
        print(f"Top-{self.k} productos por score "
              f"(mean_rating * log(1 + N_reviews)):\n")

        results = self.top_k()

        for rank, (score, parent_asin) in enumerate(results, 1):
            print(f"  {rank:>3}. {parent_asin}  score={score:.4f}")

        print(f"\nTotal: {len(results)} productos")
        return results
