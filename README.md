# Amazon Reviews 2023 - Descarga de Dataset

Script para descargar el dataset [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) desde HuggingFace.

## Requisitos

```bash
conda install datasets huggingface_hub
```

## Uso

### Descargar reviews de una categoría específica

```bash
python descargar_dataset_amazon_reviews.py --categorias Electronics --solo-reviews
```

### Descargar reviews de varias categorías

```bash
python descargar_dataset_amazon_reviews.py --categorias Electronics Books Video_Games --solo-reviews
```

### Descargar reviews de todas las categorías

```bash
python descargar_dataset_amazon_reviews.py --solo-reviews
```

### Listar categorías disponibles

```bash
python descargar_dataset_amazon_reviews.py --listar-categorias
```

## Categorías disponibles

| Categoría | Tamaño (reviews) |
|---|---|
| Subscription_Boxes | 8.95 MB |
| Magazine_Subscriptions | 33.3 MB |
| Gift_Cards | 50.2 MB |
| Digital_Music | 78.8 MB |
| Health_and_Personal_Care | 227 MB |
| Handmade_Products | 289 MB |
| All_Beauty | 326.6 MB |
| Appliances | 929.5 MB |
| Amazon_Fashion | 1.05 GB |
| Musical_Instruments | 1.56 GB |
| Software | 1.87 GB |
| Industrial_and_Scientific | 2.3 GB |
| Video_Games | 2.7 GB |
| Baby_Products | 2.9 GB |
| CDs_and_Vinyl | 3.3 GB |
| Arts_Crafts_and_Sewing | 3.9 GB |
| Grocery_and_Gourmet_Food | 5.97 GB |
| Office_Products | 5.8 GB |
| Toys_and_Games | 7.3 GB |
| Patio_Lawn_and_Garden | 7.7 GB |
| Movies_and_TV | 8.4 GB |
| Pet_Supplies | 8.4 GB |
| Automotive | 8.7 GB |
| Cell_Phones_and_Accessories | 9.3 GB |
| Sports_and_Outdoors | 9.3 GB |
| Beauty_and_Personal_Care | 11 GB |
| Health_and_Household | 11.4 GB |
| Tools_and_Home_Improvement | 12.8 GB |
| Kindle_Store | 15.8 GB |
| Books | 20.1 GB |
| Electronics | 22.6 GB |
| Clothing_Shoes_and_Jewelry | 27.8 GB |
| Unknown | 29.9 GB |
| Home_and_Kitchen | 31.4 GB |

## Estructura de salida

Los archivos se descargan en formato JSONL dentro del directorio `dataset/amazon_reviews/`:

```
dataset/amazon_reviews/
  raw/review_categories/
    Electronics.jsonl
    Books.jsonl
    ...
```

## Campos del dataset de reviews

| Campo | Descripción |
|---|---|
| `rating` | Calificación (1-5) |
| `title` | Título de la review |
| `text` | Texto de la review |
| `asin` | ID del producto |
| `parent_asin` | ASIN del producto padre |
| `user_id` | ID del usuario |
| `timestamp` | Fecha de la review (epoch ms) |
| `verified_purchase` | Compra verificada |
| `helpful_vote` | Votos de utilidad |

## SistemaRec1: Top-K Ranking con MinHeap

Sistema de recomendación que identifica los K mejores productos de una categoría usando un MinHeap de tamaño fijo.

**Fórmula de score:**

```
score = mean_rating * log(1 + N_reviews)
```

Combina calidad (rating promedio) con popularidad (cantidad de reviews).

### Uso en Python

```python
from sistema_rec import SistemaRec1

sistema = SistemaRec1(category="Amazon_Fashion", k=10)
resultados = sistema.run()
```

### Tests

```bash
# Tests unitarios (no requieren dataset)
pytest tests/test_sistema_rec1.py -v -k "TestMinHeap or TestSistemaRec1Score"

# Tests de integración (requieren dataset descargado)
pytest tests/test_sistema_rec1.py -v --category Amazon_Fashion --top-k 10
```
