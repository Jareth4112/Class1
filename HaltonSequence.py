import Interfaces as itf
import VanDerCorputSeq as vqs
import numpy as np

class HaltonSequence(itf.RandomNumberGenerator):
    """
    Constructs a Halton sequence with the given bases.
    The bases should be integers > 1.
    """

    def __init__(self, base, currentIndex=None):
        if any(b <= 1 for b in base):
            raise ValueError("Each base must be greater than 1")
        self.base = base
        self.currentIndex = currentIndex if currentIndex is not None else itf.AtomicLong()

    def getNext(self):
        index = self.currentIndex.getAndIncrement()
        return self.getHaltonNumber(index)

    def getDimension(self):
        return len(self.base)

    def getHaltonNumber(self, index):
        return [self.getVanDerCorputNumber(index, b) for b in self.base]
    
    def getHaltonNumber(self, index):
        x = np.zeros(len(self.base), dtype=np.float64)
        for dimension in range(len(self.base)):
            x[dimension] = vqs.VanDerCorputSequence.getVanDerCorputNumber(index, self.base[dimension])
        return x
    
    def getHaltonNumberForGivenBase(self, index, base):
        return vqs.getVanDerCorputNumber(index, base)
    
    def __repr__(self):
        return f"HaltonSequence(base={self.base}, currentIndex={self.currentIndex.get()})"