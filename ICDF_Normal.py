##############################################-->CLASE ICDF<--##############################################
import VanDerCorputSeq as vdc
import Interfaces as itf
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import MersenneTwister as mt
import SobolSequence as ss

class NormalICDFExperiments:

    def __init__(self, seed=3636):
        # Solo estado, sin plots ni prints.
        self.seed = seed
        self.random = np.random.default_rng(seed)

    # ---------------------- Tests numéricos ----------------------
    def testICDFImplementation(self, icdf, cdf, uniform):
        print("u: uniform, x: normal")
        print(f"        u = {uniform}")
        x = icdf(uniform)
        print(f"     x(u) = {x}")
        c = cdf(x)
        print(f"  u(x(u)) = {c}")
        expPlusX = math.exp(x)
        print(f"   exp(x) = {expPlusX}")
        expMinusX = math.exp(-x)
        print(f"  exp(-x) = {expMinusX}")
        print("-" * 80)

    def testICDFImplementations(self):
        icdf = norm.ppf
        cdf = norm.cdf

        test_points = [
            0.0,
            np.finfo(float).tiny,
            2 ** -53,
            1.0 - 2 ** -53,
            1.0
        ]

        for u in test_points:
            print(f"\nTesting Apache Commons Math and Finmath ICDF implementation: u = {u}")
            print("_" * 80)
            self.testICDFImplementation(icdf, cdf, u)  #points tested

    # ---------------------- Rutina genérica de plots ----------------------
    def plotDensityUniformAndNormalViaICDF(self, randomNumberGenerator, icdf):
        print("Entrando a plotDensityUniformAndNormalViaICDF...")
        # Debug prints opcionales:
        # print(f"Generando datos con {randomNumberGenerator}...")
        values_uniform = []
        values_normal = []

        for _ in range(100000):
            u = randomNumberGenerator.nextDouble()
            n = icdf(u)
            values_uniform.append(u)
            values_normal.append(n)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(values_uniform, bins=100, density=True, color='skyblue')
        plt.title(f"Uniforme\n{randomNumberGenerator}")

        plt.subplot(1, 2, 2)
        plt.hist(values_normal, bins=100, density=True, color='lightgreen')
        plt.title(f"Normal via ICDF\n{randomNumberGenerator}")

        plt.tight_layout()
        plt.show()

    # ---------------------- Adaptadores concretos (como métodos estáticos en Java) ----------------------
    def plotDensityUniformAndNormalViaICDFJavaRandom(self):
        class JavaRandomAdapter(itf.RandomNumberGenerator1D):
            def __init__(self, seed):
                import numpy as _np  # import local para evitar dependencia circular
                self.rng = _np.random.default_rng(seed)
            def nextDouble(self):
                return self.rng.random()
            def __str__(self):
                return "Java Random (LCG)"
        self.plotDensityUniformAndNormalViaICDF(JavaRandomAdapter(self.seed), norm.ppf)

    def plotDensityUniformAndNormalViaICDFMersenneTwister(self):
        self.plotDensityUniformAndNormalViaICDF(mt.MersenneTwister(self.seed), norm.ppf)

    def plotDensityUniformAndNormalViaICDFVanDerCorput(self):
        self.plotDensityUniformAndNormalViaICDF(vdc.VanDerCorputSequence(2), norm.ppf)

    def plotDensityUniformAndNormalViaICDFSobol(self):
        sobol = ss.SobolSequence1D()
        self.plotDensityUniformAndNormalViaICDF(sobol, norm.ppf)

    # ---------------------- "main" estilo Java ----------------------
    def run_all(self):
        """Replica el main() de Java."""
        self.plotDensityUniformAndNormalViaICDFJavaRandom()
        self.plotDensityUniformAndNormalViaICDFMersenneTwister()
        self.plotDensityUniformAndNormalViaICDFVanDerCorput()
        self.plotDensityUniformAndNormalViaICDFSobol()
        self.testICDFImplementations()

