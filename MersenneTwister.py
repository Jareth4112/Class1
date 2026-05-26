import numpy as np
import Interfaces as itf

class MersenneTwister(itf.RandomNumberGenerator1D):
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.SeedSequence().entropy  # comportamiento similar a Random().nextLong()
        self.seed = seed
        bit_generator = np.random.MT19937(seed)
        self.rng = np.random.Generator(bit_generator)

    def nextDouble(self) -> float:
        return self.rng.random()

    def nextDoubleFast(self) -> float:
        return self.rng.random()

    def __str__(self):
        return f"MersenneTwister [seed={self.seed}]"
