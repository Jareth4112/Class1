import Interfaces as itf
from scipy.stats import qmc
from typing import List
import threading  # ¡Faltaba este import!

class SobolSequence(itf.RandomNumberGenerator):
    def __init__(self, dimension: int):
        self._dimension = dimension
        self._generator = qmc.Sobol(d=dimension, scramble=False)
        self._lock = threading.Lock()
        self._has_skipped_first = False

    def getNext(self) -> List[float]:
        with self._lock:
            if not self._has_skipped_first:
                self._generator.fast_forward(1)
                self._has_skipped_first = True
            return self._generator.random()[0].tolist()  # Necesario retornar la muestra

    def getDimension(self) -> int:
        return self._dimension


class SobolSequence1D(itf.RandomNumberGenerator1D):
    def __init__(self):
        self.sobol_sequence = SobolSequence(dimension=1)

    def nextDouble(self) -> float:
        return self.sobol_sequence.getNext()[0]

    def __str__(self):
        return "Sobol Sequence"

    