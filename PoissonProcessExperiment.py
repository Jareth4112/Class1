import math
import MersenneTwister as ms  # tu clase ya implementada
import matplotlib.pyplot as plt
import numpy as np

class PoissonProcessExperiment:

    def plotPoissonPaths(self):
        numberOfPaths = 1000
        maturity = 10.0
        lambda_ = 1.0

        randomNumberGenerator = ms.MersenneTwister(3141)

        # Parte 1: generar lista de listas de tiempos de salto
        jumpTimesPaths = []
        for pathIndex in range(numberOfPaths):
            jumpTimes = []
            nextJumpTime = 0.0
            while nextJumpTime < maturity:
                uniform = randomNumberGenerator.nextDouble()
                timeStep = -math.log(uniform) / lambda_
                nextJumpTime += timeStep
                if nextJumpTime <= maturity:
                    jumpTimes.append(nextJumpTime)
            jumpTimesPaths.append(jumpTimes)

        # --- Parte 2: función M(t) = N(t) - lambda * t ---
        def Process(time):
            """
            Retorna un array numpy con los valores M(t) para todas las trayectorias
            en un tiempo fijo 'time'.
            """
            values = np.zeros(numberOfPaths)
            for pathIndex in range(numberOfPaths):
                count = sum(1 for jt in jumpTimesPaths[pathIndex] if jt <= time)
                values[pathIndex] = count - lambda_ * time
            return values

        # --- Parte 3: Ploteo del proceso ---
        numberOfTimeSteps = 1000
        deltaT = 0.01
        timeGrid = np.linspace(0.0, numberOfTimeSteps * deltaT, numberOfTimeSteps+1)

        plt.figure(figsize=(10,6))
        for pathIndex in range(20):  # ploteamos hasta 200 caminos
            values = [Process(time)[pathIndex] for time in timeGrid]
            plt.plot(timeGrid, values, lw=1)

        plt.title("Paths of compensated Poisson process")
        plt.xlabel("time (t)")
        plt.ylabel("M(t) = N(t) - lambda t")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    experiment = PoissonProcessExperiment()
    experiment.plotPoissonPaths()