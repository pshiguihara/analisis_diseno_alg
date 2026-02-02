"""
Benchmark: SistemaRec1 (MinHeap O(n log K)) vs SistemaRecNaive (sort O(n log n)).

Compara ambos sistemas sobre todas las categorías de Amazon < 2 GB,
calcula métricas de sistemas de recomendación y genera un reporte CSV.

Uso:
    python tests/benchmark.py
    python tests/benchmark.py --top-k 20
    python tests/benchmark.py --output resultados.csv
"""

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sistema_rec.sistema_rec1 import SistemaRec1
from sistema_rec.sistema_rec_naive import SistemaRecNaive

# Categorías cuyo archivo JSONL de reviews pesa menos de 2 GB
CATEGORIES_UNDER_2GB = [
    "Subscription_Boxes",       # 8.95 MB
    "Magazine_Subscriptions",   # 33.3 MB
    "Gift_Cards",               # 50.2 MB
    "Digital_Music",            # 78.8 MB
    "Health_and_Personal_Care", # 227 MB
    "Handmade_Products",        # 289 MB
    "All_Beauty",               # 326.6 MB
    "Appliances",               # 929.5 MB
    "Amazon_Fashion",           # 1.05 GB
    "Musical_Instruments",      # 1.56 GB
    "Software",                 # 1.87 GB
]

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset",
    "amazon_reviews",
)

CSV_COLUMNS = [
    "category",
    "k",
    "time_rec1_s",
    "time_naive_s",
    "speedup",
    "precision_at_k",
    "ap_at_k",
    "ndcg_at_k",
    "jaccard_at_k",
    "spearman_rho",
]


# ====================================================================== #
# Métricas de sistemas de recomendación                                   #
# ====================================================================== #

def precision_at_k(ref: list, evl: list) -> float:
    """Precision@K: fracción de items en *evl* presentes en *ref*.

    Mide qué proporción de los K items recomendados por el sistema
    evaluado coincide con los del sistema de referencia.
    """
    set_ref = {item_id for _, item_id in ref}
    set_evl = {item_id for _, item_id in evl}
    if not set_evl:
        return 0.0
    return len(set_ref & set_evl) / len(set_evl)


def average_precision_at_k(ref: list, evl: list) -> float:
    """AP@K (Average Precision at K).

    Recorre el ranking evaluado posición por posición; cada vez que
    encuentra un item relevante (presente en ref), acumula la precisión
    en esa posición.  Penaliza rankings donde los items relevantes
    aparecen en posiciones bajas.
    """
    set_ref = {item_id for _, item_id in ref}
    if not set_ref:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for i, (_, item_id) in enumerate(evl, 1):
        if item_id in set_ref:
            hits += 1
            sum_precision += hits / i

    return sum_precision / min(len(ref), len(evl)) if evl else 0.0


def _dcg(gains: list[float]) -> float:
    """Discounted Cumulative Gain."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ref: list, evl: list) -> float:
    """NDCG@K (Normalized Discounted Cumulative Gain).

    Usa los scores del ranking de referencia como relevancias ideales.
    Mide qué tan bien el ranking evaluado preserva el orden óptimo,
    penalizando items relevantes que aparecen en posiciones tardías.
    """
    relevance = {item_id: score for score, item_id in ref}

    eval_gains = [relevance.get(item_id, 0.0) for _, item_id in evl]
    actual_dcg = _dcg(eval_gains)

    ideal_gains = sorted(relevance.values(), reverse=True)
    ideal_dcg = _dcg(ideal_gains)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def jaccard_at_k(ranking_a: list, ranking_b: list) -> float:
    """Jaccard@K: similaridad de conjuntos entre dos top-K.

    |A ∩ B| / |A ∪ B|.  Mide el solapamiento global de los items
    recomendados sin considerar el orden.
    """
    set_a = {item_id for _, item_id in ranking_a}
    set_b = {item_id for _, item_id in ranking_b}
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def spearman_rho(ranking_a: list, ranking_b: list) -> float:
    """Correlación de Spearman entre los rankings de items comunes.

    Mide qué tan similar es el *orden* de los items que ambos sistemas
    tienen en común.  ρ = 1 indica orden idéntico, ρ = 0 sin correlación.
    """
    rank_a = {item_id: i for i, (_, item_id) in enumerate(ranking_a)}
    rank_b = {item_id: i for i, (_, item_id) in enumerate(ranking_b)}

    common = set(rank_a) & set(rank_b)
    n = len(common)
    if n < 2:
        return float("nan")

    d_sq_sum = sum((rank_a[item] - rank_b[item]) ** 2 for item in common)
    return 1 - (6 * d_sq_sum) / (n * (n ** 2 - 1))


# ====================================================================== #
# Benchmark runner                                                        #
# ====================================================================== #

def benchmark_category(category: str, k: int) -> dict | None:
    """Ejecuta ambos sistemas sobre una categoría y retorna métricas."""
    rec1 = SistemaRec1(category=category, k=k, data_dir=DATA_DIR)
    naive = SistemaRecNaive(category=category, k=k, data_dir=DATA_DIR)

    # Verificar que el archivo exista
    filepath = rec1._get_filepath()
    if not os.path.exists(filepath):
        print(f"  [SKIP] Archivo no encontrado: {filepath}")
        return None

    # --- SistemaRec1 (MinHeap) ---
    t0 = time.perf_counter()
    results_rec1 = rec1.top_k()
    time_rec1 = time.perf_counter() - t0

    # --- SistemaRecNaive (sort) ---
    t0 = time.perf_counter()
    results_naive = naive.top_k()
    time_naive = time.perf_counter() - t0

    speedup = time_naive / time_rec1 if time_rec1 > 0 else float("inf")

    # --- Métricas (ref = naive como ground-truth con sort determinista) ---
    prec = precision_at_k(results_naive, results_rec1)
    ap = average_precision_at_k(results_naive, results_rec1)
    ndcg = ndcg_at_k(results_naive, results_rec1)
    jacc = jaccard_at_k(results_rec1, results_naive)
    rho = spearman_rho(results_naive, results_rec1)

    return {
        "category": category,
        "k": k,
        "time_rec1_s": round(time_rec1, 4),
        "time_naive_s": round(time_naive, 4),
        "speedup": round(speedup, 4),
        "precision_at_k": round(prec, 4),
        "ap_at_k": round(ap, 4),
        "ndcg_at_k": round(ndcg, 4),
        "jaccard_at_k": round(jacc, 4),
        "spearman_rho": round(rho, 4) if not math.isnan(rho) else "NaN",
    }


def run_benchmark(categories: list[str], k: int, output_path: str) -> None:
    """Ejecuta el benchmark completo y genera el CSV."""
    rows = []

    print(f"Benchmark: SistemaRec1 vs SistemaRecNaive  (K={k})")
    print(f"Categorías: {len(categories)}")
    print(f"Output: {output_path}")
    print("=" * 70)

    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] {category} ...", flush=True)
        row = benchmark_category(category, k)
        if row is None:
            continue
        rows.append(row)
        print(f"  Rec1: {row['time_rec1_s']}s | "
              f"Naive: {row['time_naive_s']}s | "
              f"Speedup: {row['speedup']}x")
        print(f"  Precision@K={row['precision_at_k']}  "
              f"AP@K={row['ap_at_k']}  "
              f"NDCG@K={row['ndcg_at_k']}  "
              f"Jaccard={row['jaccard_at_k']}  "
              f"Spearman={row['spearman_rho']}")

    # Escribir CSV
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{'=' * 70}")
        print(f"Reporte CSV guardado en: {output_path}")
    else:
        print("\nNo se procesó ninguna categoría. Verifica que el dataset "
              "esté descargado.")


# ====================================================================== #
# CLI                                                                     #
# ====================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: SistemaRec1 vs SistemaRecNaive"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cantidad de productos en el ranking (default: 10)",
    )
    parser.add_argument(
        "--output",
        default="tests/benchmark_report.csv",
        help="Ruta del CSV de salida (default: tests/benchmark_report.csv)",
    )
    args = parser.parse_args()

    run_benchmark(CATEGORIES_UNDER_2GB, args.top_k, args.output)


if __name__ == "__main__":
    main()
