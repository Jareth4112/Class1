import matplotlib.pyplot as plt
import MersenneTwister as mt
import AnalitycsFormulas as af
from scipy.stats import norm
import math
import numpy as np

class ImportanceSamplingExperiment():
    def __init__(self, initialStockValue = 100, riskFreeRate = 0.05, volatility = 0.30,
				 optionMaturity = 5.0, optionStrike = 150, seed = 1616, numberOfSamples = 10000):
        #Model parameters
        self.initialStockValue = initialStockValue
        self.riskFreeRate = riskFreeRate
        self.volatility = volatility

        #Product parameters
        self.optionMaturity = optionMaturity
        self.optionStrike = optionStrike
        
        #Monte-Carlo parameters
        self.seed = seed
        self.numberOfSamples = numberOfSamples

    def plot(self):
        shifts = []
        errors = []

        for i in range(100):
            shift = i/100.00*3.0
            monteCarloErrorForShift = self.getMCErrorForValueWithImportanceSamplingShift(shift)

            shifts.append(shift)
            errors.append(monteCarloErrorForShift)

        plt.figure(figsize=(8, 6))
        plt.scatter(shifts, errors, colorizer='red')
        plt.title(f'Monte-Carlo Approximation Error (seed={self.seed}')
        plt.xlabel('Shift size (importance sampling)')
        plt.ylabel('Error')
        plt.show()

    def getMCErrorForValueWithImportanceSamplingShift(self, shift):
        randomNumberGenerator = mt.MersenneTwister(self.seed)

        #Value of the European option
        valueAnalytic = af.AnalyticFormulas.blackScholesOptionValue(self.initialStockValue, self.riskFreeRate, self.volatility, self.optionMaturity, self.optionStrike)

        sum = 0.0
        sumError = 0.0


        for _ in range(self.numberOfSamples):
            #Sample
            uniform = randomNumberGenerator.nextDouble()
            
            #Inverse of u using AR
            standardNormal = norm.ppf(uniform)
    
            x = standardNormal
            
            #Shift de - a 3.0
            y = x + shift

            #Underlying stock price 
            underlying = self.initialStockValue * math.exp(self.riskFreeRate * self.optionMaturity - 0.5 * self.volatility * self.volatility * self.optionMaturity + self.volatility * math.sqrt(self.optionMaturity) * y)
            
            #Payoff max(St - K)*exp(-rT)
            payoffDiscounted = max(underlying - self.optionStrike, 0) * math.exp(-self.riskFreeRate * self.optionMaturity)

            #**f**(original density in y)/**g**(shifted density in y)
            weight = math.exp(-y*y/2 + (y-shift)*(y-shift)/2)

            #price * weigth 
            value = payoffDiscounted * weight
        
            sum += value
            
            #Error between analityc value - value shifted
            sumError += math.pow(value-valueAnalytic, 2)
            
        #Monte carlo with the value shifted
        valueMonteCarlo = sum / self.numberOfSamples;
        #Squared error
        error = math.sqrt(sumError / self.numberOfSamples)

        print(f'{shift:10.2f} {shift} {valueMonteCarlo} {valueAnalytic} {error}')
        return error