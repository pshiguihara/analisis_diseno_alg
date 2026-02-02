"""Opciones CLI compartidas para pytest."""

import os
import sys

# Agregar la raíz del proyecto a sys.path para que pytest encuentre
# los paquetes estructuras_datos y sistema_rec.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_addoption(parser):
    parser.addoption(
        "--category",
        action="store",
        default="Amazon_Fashion",
        help="Categoría de Amazon Reviews para tests de integración",
    )
    parser.addoption(
        "--top-k",
        action="store",
        default=10,
        type=int,
        help="Cantidad de productos en el ranking (default: 10)",
    )
