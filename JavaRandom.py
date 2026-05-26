
import random
import Interfaces as itf

class JavaRandomGenerator(itf.RandomNumberGenerator1D):
    def __init__(self, seed=3636):
        self.random = random.Random(seed)

    def nextDouble(self):
        return self.random.random()

    def __str__(self):
        return "Java Random (LCG)"
