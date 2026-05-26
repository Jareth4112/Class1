import pandas as pd
import math
import matplotlib.pyplot as plt
import Interfaces as itf
import numpy as np
from scipy.stats import norm

class AcceptanceRejectionRandomNumberGenerator(itf.RandomNumberGenerator1D):
    def __init__(self, uniformRandomNumberGenerator, targetDensity, referenceDensity, referenceDistributionICDF, acceptanceLevel):
        super().__init__()
        self.targetDensity = targetDensity
        self.uniformRandomNumberGenerator = uniformRandomNumberGenerator
        self.referenceDensity = referenceDensity
        self.referenceDistributionICDF = referenceDistributionICDF
        self.acceptanceLevel = acceptanceLevel
        dimension = uniformRandomNumberGenerator.getDimension()
        if not (2 <= dimension <= float('inf')):  # Integer.MAX_VALUE no tiene sentido en Python
            raise ValueError("The acceptance rejection method requires a uniform distributed random number generator with at least dimension 2.")
        
    def nextDouble(self):
        rejected = True
        y = np.nan
        while rejected:
            uniform = self.uniformRandomNumberGenerator.getNext()  # [u, v]
            u = uniform[0]
            v = uniform[1]
            y = self.referenceDistributionICDF(v)
            rejected = self.targetDensity(y) < u * self.acceptanceLevel * self.referenceDensity(y) #rejected if f(y) is less C*g(y)
        return y