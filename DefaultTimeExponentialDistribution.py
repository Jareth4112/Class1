##############################################-->CLASE DTED<--##############################################
import Interfaces as itf
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import MersenneTwister as mt
import SobolSequence as ss
import VanDerCorputSeq as vq
import seaborn as sns

class DefaultTimeExponentialDistribution:
    def __init__(self, uniformSequence, lambdaE):
        self._uniformSequence = uniformSequence
        self._lambdaE = lambdaE

    def getNext(self):
        uniform = self._uniformSequence.getAsDouble()
        time = -1.0/self._lambdaE * math.log(1 - uniform)
        return time
    

class DefaultTimeExponentialDistributionExperiment:
    def __init__(self, numberOfSamples=10000, lambdaE=0.2):
        self.numberOfSamples = numberOfSamples
        self.lambdaE = lambdaE
        
    def plot(self, uniformSequence, lambdaE, numberOfSamples):
        self.lambdaE = lambdaE
        self.numberOfSamples = numberOfSamples

        defaultTimesSequence = DefaultTimeExponentialDistribution(uniformSequence, lambdaE)
        
        print(f'Exponential Distribution {uniformSequence}, λ = {lambdaE}')
        print("_"*79)
        
        defaultTimes = []
        maturity = 5
        survivalCounter = 0
        sumOfTimes = 0.0
        for i in range(numberOfSamples):
            time = defaultTimesSequence.getNext()
            defaultTimes.append(time)
            if i < 10:
                print(f'i={i}:\ttime={time}')
            sumOfTimes += time

            if time > maturity:
                survivalCounter += 1


        averageTime = sumOfTimes / numberOfSamples
        survivalProb = survivalCounter / numberOfSamples

        print(f'E[\u03c4] = {averageTime}')
        print(f'E[\u03c4 > T] = {survivalProb} (T={maturity})')
        print(f'exp[-\u03bb T] = {math.exp(-lambdaE*maturity)}')
        print("_"*79)

        plt.figure(figsize=(6, 4))
        stdv = 8.0
        maxr = 3*stdv
        sns.histplot(defaultTimes, bins = 500, stat='density')
        plt.xlim(-10, maxr)
        plt.title(f'Exponential Distribution {uniformSequence}, λ = {lambdaE}')
        plt.show
        

    def main(self):
        self.plot(mt.MersenneTwister(3141), self.lambdaE, self.numberOfSamples)
        self.plot(vq.VanDerCorputSequence(2), self.lambdaE, self.numberOfSamples)
