"""
Script para descargar el dataset Amazon-Reviews-2023 de HuggingFace.
Fuente: McAuley-Lab/Amazon-Reviews-2023

Descarga los archivos JSONL directamente usando huggingface_hub,
evitando el error "Dataset scripts are no longer supported" de datasets>=4.0.
"""

import argparse
import os

from huggingface_hub import hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"

CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]


def download_category(category, output_dir, download_reviews, download_metadata):
    """Descarga reviews y/o metadata de una categoría específica."""
    cat_dir = os.path.join(output_dir, category)
    os.makedirs(cat_dir, exist_ok=True)

    if download_reviews:
        remote_path = f"raw/review_categories/{category}.jsonl"
        local_path = os.path.join(cat_dir, f"{category}.jsonl")
        print(f"  Descargando reviews: {remote_path}")
        downloaded = hf_hub_download(
            repo_id=REPO,
            filename=remote_path,
            repo_type="dataset",
            local_dir=output_dir,
        )
        print(f"  Reviews guardadas en {downloaded}")

    if download_metadata:
        remote_path = f"raw/meta_categories/meta_{category}.jsonl"
        local_path = os.path.join(cat_dir, f"meta_{category}.jsonl")
        print(f"  Descargando metadata: {remote_path}")
        downloaded = hf_hub_download(
            repo_id=REPO,
            filename=remote_path,
            repo_type="dataset",
            local_dir=output_dir,
        )
        print(f"  Metadata guardada en {downloaded}")


def main():
    parser = argparse.ArgumentParser(
        description="Descarga el dataset Amazon-Reviews-2023 de HuggingFace"
    )
    parser.add_argument(
        "--categorias",
        nargs="+",
        default=None,
        help="Categorías a descargar (por defecto: todas). "
        "Ejemplo: --categorias Electronics Books",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/amazon_reviews",
        help="Directorio de salida (default: dataset/amazon_reviews)",
    )
    parser.add_argument(
        "--solo-reviews",
        action="store_true",
        help="Descargar solo reviews (sin metadata)",
    )
    parser.add_argument(
        "--only-metadata",
        action="store_true",
        help="Descargar solo metadata (sin reviews)",
    )
    parser.add_argument(
        "--listar-categorias",
        action="store_true",
        help="Listar categorías disponibles y salir",
    )
    args = parser.parse_args()

    if args.listar_categorias:
        print("Categorías disponibles:")
        for cat in CATEGORIES:
            print(f"  - {cat}")
        return

    categories = args.categorias if args.categorias else CATEGORIES
    download_reviews = not args.only_metadata
    download_metadata = not args.solo_reviews

    invalid = [c for c in categories if c not in CATEGORIES]
    if invalid:
        parser.error(
            f"Categorías no válidas: {invalid}\n"
            f"Usa --listar-categorias para ver las opciones."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Descargando {len(categories)} categoría(s) en '{args.output_dir}'")
    print(f"  Reviews: {'sí' if download_reviews else 'no'}")
    print(f"  Metadata: {'sí' if download_metadata else 'no'}")
    print()

    for i, category in enumerate(categories, 1):
        print(f"[{i}/{len(categories)}] {category}")
        download_category(category, args.output_dir, download_reviews, download_metadata)
        print()

    print("Descarga completada.")


if __name__ == "__main__":
    main()
