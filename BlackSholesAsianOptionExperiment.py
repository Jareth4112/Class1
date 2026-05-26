import MersenneTwister as mt
import VanDerCorputSeq as vq
import RandomNumberGeneratorFrom1D as rg1d
import HaltonSequence as hs
import math
import Interfaces as itf
from scipy.stats import norm

class BlackSholesAsianOptionExperiment():     
    def __init__(self, initialStockValue = 100, riskFreeRate = 0.05, volatility = 0.30, 
                 timesForAveraging = [ 1.0, 2.0, 3.0, 4.0, 5.0 ], optionMaturity = 5.0, 
                 optionStrike = 150, seed = 3141, numberOfSamples = 100000, 
                 primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53, 59]):
        
        #Model parameters
        self.initialStockValue = initialStockValue
        self.riskFreeRate = riskFreeRate
        self.volatility = volatility

        #Product parameters
        self.timesForAveraging = timesForAveraging
        self.optionMaturity = optionMaturity
        self.optionStrike = optionStrike

        #Monte-Carlo parameters
        self.seed = seed 
        self.numberOfSamples = numberOfSamples
        self.primes = primes

    def run(self):
        randomNumberGenerator = mt.MersenneTwister(self.seed)
        value = self.getValueOfAsianOption(randomNumberGenerator)		
        print(f"Value of Asian Option (Mersenne Twister 1D)..: {value}")
        
        randomNumberGeneratorVdC = vq.VanDerCorputSequence(2)
        valueVdC =self. getValueOfAsianOption(randomNumberGeneratorVdC)		
        print(f"Value of Asian Option (with v.d.C. seq, 1D)..: {valueVdC} \t(wrong)")
        
        dimension = len(self.timesForAveraging)
        
        randomNumberGeneratorPseudo = rg1d.RandomNumberGeneratorFrom1D(mt.MersenneTwister(self.seed), dimension)
        valuePseudo = self.getValueOfAsianOption(randomNumberGeneratorPseudo)		
        print(f"Value of Asian Option (Mersenne Twister {randomNumberGeneratorPseudo.getDimension() } D)..: { valuePseudo}")
        
        randomNumberGeneratorQuasi = hs.HaltonSequence(self.primes[0:dimension])	
        valueQuasi = self.getValueOfAsianOptionSeq(randomNumberGeneratorQuasi)		
        print(f"Value of Asian Option (Halton {randomNumberGeneratorQuasi.getDimension()} D)............: {valueQuasi}")

    def getValueOfAsianOption(self, randomNumberGenerator):
        sum = 0.0
        for _ in range(self.numberOfSamples):
            numberOfTimeSteps = len(self.timesForAveraging)
            sumOfStockValues = 0.0
            time = 0.0
            valueOfStockAtTime = self.initialStockValue
            for timeStepIndex in range(numberOfTimeSteps):
                uniform = randomNumberGenerator.nextDouble()
                standardNormal = norm.ppf(uniform)

                timeNext = self.timesForAveraging[timeStepIndex]
                timeStep = timeNext - time

                #Time step
                valueOfStockAtTime = valueOfStockAtTime * math.exp(self.riskFreeRate * timeStep - 0.5 * self.volatility * self.volatility * timeStep + self.volatility * math.sqrt(timeStep) * standardNormal)
                time = timeNext

                sumOfStockValues += valueOfStockAtTime
            
            averageOfStockValues = sumOfStockValues / numberOfTimeSteps
        
            payoffDiscounted = max(averageOfStockValues - self.optionStrike,  0) * math.exp(-self.riskFreeRate * self.optionMaturity)

            sum += payoffDiscounted

        value = sum / self.numberOfSamples
        return value

    def getValueOfAsianOptionSeq(self, randomNumberGenerator):
        sum = 0.0
        for _ in range(self.numberOfSamples):
            uniforms = randomNumberGenerator.getNext()
            numberOfTimeSteps = len(self.timesForAveraging)
            sumOfStockValues = 0.0
            time = 0.0
            valueOfStockAtTime = self.initialStockValue	# S(T_0)
            for timeStepIndex in range(numberOfTimeSteps):
                uniform = uniforms[timeStepIndex]
                standardNormal = norm.ppf(uniform)
                
                timeNext = self.timesForAveraging[timeStepIndex]
                timeStep = timeNext - time

				#time step
                valueOfStockAtTime = valueOfStockAtTime * math.exp(self.riskFreeRate * timeStep - 0.5 * self.volatility * self.volatility * timeStep + self.volatility * math.sqrt(timeStep) * standardNormal)	
                time = timeNext
				
                sumOfStockValues += valueOfStockAtTime
            
            averageOfStockValues = sumOfStockValues / numberOfTimeSteps
            payoffDiscounted = max(averageOfStockValues - self.optionStrike,  0) * math.exp(-self.riskFreeRate * self.optionMaturity)
            
            sum += payoffDiscounted
        
        value = sum / self.numberOfSamples
        return value
			
