"""
MinHeap: Almacena tuplas (score, item_id) y usa una comparación 
para mantener el elemento con menor score en la raíz.
"""


class MinHeap:
    """Min-heap que almacena tuplas (score, item_id)."""

    def __init__(self):
        self._data: list = []

    # ------------------------------------------------------------------ #
    # Índices                                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def parent(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def right(i: int) -> int:
        return 2 * i + 2

    # ------------------------------------------------------------------ #
    # Mantener propiedad min-heap (MIN-HEAPIFY)                     #
    # ------------------------------------------------------------------ #
    def min_heapify(self, i: int) -> None:
        """Corrige el sub-árbol con raíz en *i* para mantener la
        propiedad min-heap.  Implementación recursiva."""
        n = len(self._data)
        smallest = i
        l = self.left(i)
        r = self.right(i)

        if l < n and self._data[l] < self._data[smallest]:
            smallest = l
        if r < n and self._data[r] < self._data[smallest]:
            smallest = r

        if smallest != i:
            self._data[i], self._data[smallest] = (
                self._data[smallest],
                self._data[i],
            )
            self.min_heapify(smallest)

    # ------------------------------------------------------------------ #
    # Construir heap desde arreglo — O(n).                               #
    # ------------------------------------------------------------------ #
    def build_min_heap(self, array: list) -> None:
        """Construye el heap *in-place* a partir de *array* en O(n)."""
        self._data = list(array)
        for i in range(len(self._data) // 2 - 1, -1, -1):
            self.min_heapify(i)

    # ------------------------------------------------------------------ #
    # Operaciones de cola de prioridad                                     #
    # ------------------------------------------------------------------ #
    def heap_minimum(self):
        """Retorna el elemento mínimo sin extraerlo."""
        if not self._data:
            raise IndexError("heap_minimum en heap vacío")
        return self._data[0]

    def heap_extract_min(self):
        """Extrae y retorna el elemento mínimo."""
        if not self._data:
            raise IndexError("heap_extract_min en heap vacío")
        minimum = self._data[0]
        self._data[0] = self._data[-1]
        self._data.pop()
        if self._data:
            self.min_heapify(0)
        return minimum

    def heap_decrease_key(self, i: int, key) -> None:
        """Disminuye la clave del elemento en posición *i* a *key*.

        *key* debe ser menor o igual que la clave actual.
        """
        if key > self._data[i]:
            raise ValueError("La nueva clave es mayor que la clave actual")
        self._data[i] = key
        while i > 0 and self._data[self.parent(i)] > self._data[i]:
            p = self.parent(i)
            self._data[i], self._data[p] = self._data[p], self._data[i]
            i = p

    def min_heap_insert(self, key) -> None:
        """Inserta *key* en el heap usando un sentinel (inf, '')."""
        self._data.append((float("inf"), ""))
        self.heap_decrease_key(len(self._data) - 1, key)

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"MinHeap({self._data})"
