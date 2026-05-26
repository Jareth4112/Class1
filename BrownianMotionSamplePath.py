from TimeDiscretization import TimeDiscretizationFromArray 
from scipy.stats import norm
import math as m
import numpy as np
import bisect
import matplotlib.pyplot as plt
import plot2D as p2D
import plotProcess2D as pp2D
import RandomVariableFromDoubleArray as RVF1eA
import MersenneTwister as ms

class BrownianMotionSamplePaths():
	def __init__(self):
		self.numberOfTimeSteps = 100
		self.timeStep = 0.01
		self.numberOfPaths = 1000
		self.seed = 3141
		
	def plotBM(self):
		randomNumberGenerator = ms.MersenneTwister(self.seed)
		timeDiscretization = np.zeros(self.numberOfTimeSteps+1)

		for i in range(len(timeDiscretization)):
			timeDiscretization[i] = i*self.timeStep
		brownianMotionSamplePaths = []
		
		for _ in range(self.numberOfPaths):
			brownianMotionSimplePath = np.zeros(len(timeDiscretization))
			brownianMotionSimplePath[0] = 0.0

			for timeIndex in range(len(timeDiscretization)-1):
				uniform = randomNumberGenerator.nextDouble()

				#Standart normal:
				normal = norm.ppf(uniform)

				timeStep = timeDiscretization[timeIndex+1] - timeDiscretization[timeIndex]
				brownianIncrement = m.sqrt(timeStep)*normal

				brownianMotionSimplePath[timeIndex-1] = brownianMotionSimplePath[timeIndex] + brownianIncrement

			brownianMotionSamplePaths.append(brownianMotionSimplePath)

		nomberOfPathstoPlot = 100
		# --- Array of functions that map t to W(t,𝜔) (the array is over all 𝜔), i.e., array of paths.  ---
		doubleUnaryOperators = [
			(lambda t, samplePath=samplePath: samplePath[getTimeIndexLessOrEqual(timeDiscretization, t)])
			for samplePath in brownianMotionSamplePaths[:nomberOfPathstoPlot]
		]
		
		plot = p2D.Plot2D(timeDiscretization[0], timeDiscretization[len(timeDiscretization)-1], len(timeDiscretization), doubleUnaryOperators)
		plot.setTitle('Brownian Motion (observed at discrete times for selected 𝜔)')
		plot.setXAxisLabel('time T')
		plot.setYAxisLabel('W(t,𝜔)')
		plot.show()

		timeToRandomVariable = (lambda t: RVF1eA.RandomVariableFromDoubleArray(t, 
																  np.array([samplePath[getTimeIndexLessOrEqual(timeDiscretization, t)] for samplePath in brownianMotionSamplePaths])))
		
		plot = p2D.Plot2D(TimeDiscretizationFromArray(times = list(timeDiscretization), 
					dt=timeStep/10), timeToRandomVariable, 100)
		plot.setTitle('Brownian Motion (observed at discrete times for selected 𝜔)')
		plot.setXAxisLabel('time T')
		plot.setYAxisLabel('W(t,𝜔)')
		plot.show()

		def getTimeIndexLessOrEqual(timeDiscretization, time):
			"""
			Returns the index of the largest time in 'timeDiscretization' that is <= 'time'.
			If 'time' is smaller than the smallest element, returns 0.
			"""
			# bisect_right returns insertion index to the right of any existing entries of 'time'
			timeIndex = bisect.bisect_right(timeDiscretization, time) - 1
			if timeIndex < 0:
				timeIndex = 0
			return timeIndex

