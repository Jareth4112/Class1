import random as rng
from abc import ABC, abstractmethod
import threading
import Interfaces as ifc

class AtomicInteger:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._value

    def getAndIncrement(self):
        with self._lock:
            current = self._value
            self._value += 1
            return current


class VanDerCorputSequence(ifc.RandomNumberGenerator1D):
    def __init__(self, base: int, start_index: int = 0):
        if start_index < 0:
            raise ValueError("start_index must be >= 0")
        if base <= 1:
            raise ValueError("base must be > 1")

        self.index = AtomicInteger(start_index)
        self.base = base

    def nextDouble(self) -> float:
        return self.getVanDerCorputNumber(self.index.getAndIncrement(), self.base)

    @staticmethod
    def getVanDerCorputNumber(index: int, base: int) -> float:
        index += 1  # Java version starts from 1
        x = 0.0
        refinement_factor = 1.0 / base

        while index > 0:
            x += (index % base) * refinement_factor
            index //= base
            refinement_factor /= base

        return x

    def __str__(self):
        return f'VanDerCorputSequence [index={self.index.get()}, base={self.base}]'
    

class VanDerCorputSequenceExperiment:
    def __init__(self, base=2, count=10):
        self.random_number_generator = VanDerCorputSequence(base)
        self.count = count

    def run(self):
        for i in range(self.count):
            x = self.random_number_generator.nextDouble()
            print(f"i={i}\tx_i={x}")


