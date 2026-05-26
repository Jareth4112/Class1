import Interfaces as itf
class RandomNumberGeneratorFrom1D(itf.RandomNumberGenerator):
	def __init__(self, randomNumberGenerator, dimension):
		super().__init__()
		self.randomNumberGenerator = randomNumberGenerator
		self.dimension = dimension

	def getNext(self):
		value = [self.randomNumberGenerator.getAsDouble() for _ in range(self.dimension)]
		return value
	
	def getDimension(self):
		return self.dimension
	
	def nextDouble(self):
		# Simula lo que tendría en Java un RandomNumberGenerator1D
		return self.randomNumberGenerator.getAsDouble()

	def __str__(self):
		return f"RandomNumberGeneratorFrom1D [randomNumberGenerator={self.randomNumberGenerator}, dimension={self.dimension}]"


