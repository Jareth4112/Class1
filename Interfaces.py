from abc import ABC, abstractmethod

class RandomNumberGenerator(ABC):
    @abstractmethod
    def getNext(self):
        """Returns the next sample vector in [0,1]^n."""
        pass

    @abstractmethod
    def getDimension(self):
        """Returns the dimension of the sample vector."""
        pass


class DoubleSupplier(ABC):
    @abstractmethod
    def getAsDouble(self):
        """Returns the next double value (float in Python)."""
        pass


class RandomNumberGenerator1D(RandomNumberGenerator, DoubleSupplier):
    @abstractmethod
    def nextDouble(self):
        """Thread-safe implementation returning the next float in [0,1)."""
        pass

    def nextDoubleFast(self):
        """Possibly faster, non-thread-safe implementation."""
        return self.nextDouble()

    def getNext(self):
        """Returns a list [nextDouble()] to simulate 1D vector."""
        return [self.nextDouble()]

    def getDimension(self):
        """Always returns 1."""
        return 1

    def getAsDouble(self):
        """Alias to allow functional-style use."""
        return self.nextDouble()


import threading

class AtomicLong:
    def __init__(self, value=0):
        self.value = value
        self._lock = threading.Lock()

    def getAndIncrement(self):
        with self._lock:
            val = self.value
            self.value += 1
            return val

    def get(self):
        with self._lock:
            return self.value

