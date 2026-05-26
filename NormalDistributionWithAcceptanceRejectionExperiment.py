import time 
from MersenneTwister import MersenneTwister
import math
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy.stats import norm

class NormalDistributionWithAcceptanceRejectionExperiment(MersenneTwister):
    def __init__(self, numberOfSamples = 10000000) :
        self.numberOfSamples = numberOfSamples

    def signum(self, x):
        return (x > 0) - (x < 0)

    def testARWithMersenneTwister3D(self):
        '''
        It's name is related to 3 uniforms that I need to simulate a variable with normal distribution X. 
        - u = to sample my acceptance rejection criteria
        - v = to do inversion of the distribution function to sample *Y*, representing the magnitude |X|
        - s = to sample the sign of X.  
        '''
        timeStart = int(time.time() * 1000)
        mersenne = MersenneTwister(3636)
        valuesNormal = []
        
        for _ in range(self.numberOfSamples):
            x = 0
            isRejected = True

            while isRejected:
                u = mersenne.nextDouble()
                v = mersenne.nextDouble() #

                x = math.log(1-v) #inversion of the function.

                isRejected = u >= math.exp(-0.5*(x-1)*(x-1)) #acceptance rejection criteria. 

            s = 1.0 if mersenne.nextDouble() >= 0.5 else -1.0 #sample of the sign of x
            normal = s * x
            valuesNormal.append(normal)

        timeEnd = int(time.time() * 1000) 
        timeSec = (timeEnd-timeStart) / 1000.0;
        print(f'Time AR from MersenneTwister 3D.....: " {timeSec} " sec.')

        plt.figure(figsize=(6, 4))
        sns.kdeplot(valuesNormal, bw_adjust=1, fill=True)
        plt.xlim(-4.0, 4.0) 
        plt.title(f'Normal via AR from MersenneTwister 3D')
        plt.show()

    def testARWithMersenneTwister2D(self):
        """
        Acceptance-Rejection sampling of the standard normal distribution
        using the double exponential (Laplace) distribution as proposal (2D case).

        Sampling steps:

            S = sign(2v - 1)
            Z = -ln(1 - |2v - 1|)
            Y = S * Z
            Accept Y if u <= exp(-0.5 * (Z - 1)^2)

        Where:
            u, v ~ Uniform(0, 1)
            Accepted Y ~ Normal(0, 1)
        """
        timeStart = int(time.time() * 1000)
        mersenne = MersenneTwister(3636)
        valuesNormal = []

        for _ in range(self.numberOfSamples):
            normal = 0
            isRejected = True

            while isRejected:
                u = mersenne.nextDouble()
                v = mersenne.nextDouble()

                x = math.log(1-np.abs(2*v-1))

                isRejected = u >= math.exp(-0.5*(x-1)*(x-1))
                s = self.signum(2*v-1)
                normal = s * x
            valuesNormal.append(normal)
        timeEnd = int(time.time() * 1000) 
        timeSec = (timeEnd-timeStart) / 1000.0;
        print(f'Time AR from MersenneTwister 2D.....: " {timeSec} " sec.')

        plt.figure(figsize=(6, 4))
        sns.kdeplot(valuesNormal, bw_adjust=1, fill=True)
        plt.xlim(-4.0, 4.0) 
        plt.title(f'Normal via AR from MersenneTwister 2D')
        plt.show()

    def testICDFWithMersenneTwister(self):
        timeStart = int(time.time() * 1000)
        mersenne = MersenneTwister(3636)
        valuesNormal = []

        for _ in range(self.numberOfSamples):
            uniform = mersenne.nextDouble()
            normal = norm.ppf(uniform)
            valuesNormal.append(normal)
        timeEnd = int(time.time() * 1000) 
        timeSec = (timeEnd-timeStart) / 1000.0;
        print(f'Time ICDF from MersenneTwister 1D...:" {timeSec} " sec.')
        
        plt.figure(figsize=(6, 4))
        sns.kdeplot(valuesNormal, bw_adjust=1, fill=True)
        plt.xlim(-4.0, 4.0) 
        plt.title(f'Normal via ICDF from MersenneTwister')
        plt.show()
    



