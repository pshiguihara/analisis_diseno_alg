"""
Benchmark completo: SistemaRec1 vs SistemaRecNaive variando K.

Ejecuta ambos sistemas para cada valor de K en range(5, 1000, 100)
sobre una categoría dada, y genera un scatterplot PNG comparando
el tiempo de ejecución en milisegundos.

Optimización: el JSONL se lee y agrega UNA sola vez. Luego se mide
únicamente la fase de ranking para cada valor de K, aislando la
diferencia algorítmica O(n log K) vs O(n log n) sin ruido de I/O.

Uso:
    python tests/benchmark_completo.py
    python tests/benchmark_completo.py --categoria Amazon_Fashion
    python tests/benchmark_completo.py --output mi_grafico.png
"""

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from estructuras_datos.heap import MinHeap
from sistema_rec.sistema_rec1 import SistemaRec1

K_VALUES = list(range(5, 1000, 100))

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "amazon_reviews",
)


# ====================================================================== #
# Funciones de ranking aisladas (sin I/O)                                 #
# ====================================================================== #

def rank_with_heap(aggregated: dict, k: int) -> list[tuple[float, str]]:
    """Top-K con MinHeap — O(n log K). Solo ranking, sin I/O."""
    heap = MinHeap()
    for parent_asin, (sum_ratings, count) in aggregated.items():
        score = SistemaRec1.compute_score(sum_ratings, count)
        if len(heap) < k:
            heap.min_heap_insert((score, parent_asin))
        elif score > heap.heap_minimum()[0]:
            heap.heap_extract_min()
            heap.min_heap_insert((score, parent_asin))
    result = []
    while len(heap) > 0:
        result.append(heap.heap_extract_min())
    result.reverse()
    return result


def rank_with_sort(aggregated: dict, k: int) -> list[tuple[float, str]]:
    """Top-K con sort completo — O(n log n). Solo ranking, sin I/O."""
    scored = [
        (SistemaRec1.compute_score(sum_r, count), parent_asin)
        for parent_asin, (sum_r, count) in aggregated.items()
    ]
    scored.sort(reverse=True)
    return scored[:k]


# ====================================================================== #
# Lectura y agregación (una sola vez)                                     #
# ====================================================================== #

def load_and_aggregate(filepath: str) -> dict:
    """Lee el JSONL y retorna dict[parent_asin] -> [sum_ratings, count]."""
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


# ====================================================================== #
# Benchmark runner                                                        #
# ====================================================================== #

def run_benchmark(category: str, output_png: str, output_csv: str) -> None:
    """Ejecuta el benchmark para todos los valores de K y genera el gráfico."""
    filepath = os.path.join(
        DATA_DIR, "raw", "review_categories", f"{category}.jsonl"
    )
    if not os.path.exists(filepath):
        print(f"Error: archivo no encontrado: {filepath}")
        sys.exit(1)

    # --- Fase de I/O: una sola lectura ---
    print(f"Benchmark completo: SistemaRec1 vs SistemaRecNaive")
    print(f"Categoría: {category}")
    print(f"Valores de K: {K_VALUES}")
    print("=" * 60)

    print(f"\nLeyendo y agregando {filepath} ...", flush=True)
    t0 = time.perf_counter()
    aggregated = load_and_aggregate(filepath)
    time_io = time.perf_counter() - t0
    print(f"  {len(aggregated):,} productos únicos en {time_io:.2f}s")
    print(f"\nMidiendo solo la fase de ranking (sin I/O):\n")

    # --- Fase de ranking: solo CPU ---
    times_rec1 = []
    times_naive = []

    for k in K_VALUES:
        print(f"  K={k:>4} ...", end=" ", flush=True)

        t0 = time.perf_counter()
        rank_with_heap(aggregated, k)
        t_rec1 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rank_with_sort(aggregated, k)
        t_naive = (time.perf_counter() - t0) * 1000

        times_rec1.append(t_rec1)
        times_naive.append(t_naive)
        print(f"Rec1: {t_rec1:>8.4f} ms | Naive: {t_naive:>8.4f} ms")

    # --- Guardar CSV ---
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["top_k", "time_rec1_ms", "time_naive_ms"])
        for k, t1, tn in zip(K_VALUES, times_rec1, times_naive):
            writer.writerow([k, round(t1, 4), round(tn, 4)])
    print(f"\nCSV guardado en: {output_csv}")

    # --- Generar scatterplot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(K_VALUES, times_rec1, label="SistemaRec1 (MinHeap)", marker="o")
    ax.scatter(K_VALUES, times_naive, label="SistemaRecNaive (sort)", marker="s")

    ax.set_xlabel("top-k")
    ax.set_ylabel("Tiempo de ejecución (ms)")
    ax.set_title(f"SistemaRec1 vs SistemaRecNaive — {category} (solo ranking)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)

    print(f"Scatterplot guardado en: {output_png}")


# ====================================================================== #
# CLI                                                                     #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark completo variando K con scatterplot"
    )
    parser.add_argument(
        "--categoria",
        default="Amazon_Fashion",
        help="Categoría a evaluar (default: Amazon_Fashion)",
    )
    parser.add_argument(
        "--output",
        default="tests/benchmark_completo.png",
        help="Ruta del PNG de salida (default: tests/benchmark_completo.png)",
    )
    parser.add_argument(
        "--output-csv",
        default="tests/benchmark_completo.csv",
        help="Ruta del CSV de salida (default: tests/benchmark_completo.csv)",
    )
    args = parser.parse_args()

    run_benchmark(args.categoria, args.output, args.output_csv)


if __name__ == "__main__":
    main()
