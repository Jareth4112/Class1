import random
import numpy as np
import math 
import time
from concurrent.futures import ThreadPoolExecutor #Execute computations asynchronously using threads or processes.
from scipy.stats.qmc import Halton
import multiprocessing
from scipy.stats import qmc
from numpy.random import MT19937, Generator
from typing import Callable
import VanDerCorputSeq as vdcs

class Integrator1D:
    def integrate(self, integrand, lower_bound, upper_bound):
        """
        Interface method to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")
    
class MonteCarloIntegrator1D(Integrator1D):
    def __init__(self, number_of_evaluation_points, seed):
        self.number_of_evaluation_points = number_of_evaluation_points
        self.seed = seed

    def integrate(self, integrand, lower_bound, upper_bound):
        np.random.seed(self.seed)
        domain_size = upper_bound - lower_bound 

        sum = 0.0

        for i in range(self.number_of_evaluation_points):
            random_number = np.random.uniform(0.0, 1.0)
            argument = lower_bound + random_number*domain_size
            value = integrand(argument)

            sum += value

        return sum / self.number_of_evaluation_points*domain_size


class MonteCarloIntegrator1DWithStreams(Integrator1D):
    def __init__(self, number_of_evaluation_points, seed=3141):
        self.number_of_evaluation_points = number_of_evaluation_points
        self.seed = seed

    def integrate(self, integrand, lower_bound, upper_bound):
        rng = random.Random(self.seed)
        domain_size = upper_bound - lower_bound

        random_numbers = [rng.random() for _ in range(self.number_of_evaluation_points)]
        sum_values = sum(integrand(lower_bound + x * domain_size) for x in random_numbers)

        return sum_values / self.number_of_evaluation_points * domain_size
    

class SimpsonsIntegrator1D(Integrator1D):
    def __init__(self, number_of_evaluation_points):
        if number_of_evaluation_points % 2 != 1: 
            raise ValueError("number_of_evaluation_points must be odd")
        self.number_of_evaluation_points = number_of_evaluation_points

    def integrate(self, integrand, lower_bound, upper_bound):
        n = self.number_of_evaluation_points
        h = (upper_bound - lower_bound) / (n - 1) #Step size

        integral = integrand(lower_bound) + integrand(upper_bound)

        for i in range(1, n - 1):
            x = lower_bound + i * h
            weight = 4 if i % 2 != 0 else 2
            integral += weight * integrand(x)

        return integral * h / 3


class SimpsonsIntegrator1DWithStreams(Integrator1D):
    def __init__(self, number_of_evaluation_points):
        self.number_of_evaluation_points = number_of_evaluation_points
        if number_of_evaluation_points % 2 != 1:
            raise ValueError("number_of_evaluation_points should be odd")

    def integrate(self, integrand, lower_bound, upper_bound):
        domain_size = upper_bound - lower_bound
        number_of_double_intervals = (self.number_of_evaluation_points - 1) // 2
        interval_size = domain_size / number_of_double_intervals / 2.0  # h

        # Suma principal con pesos 2 y 4 en los puntos internos
        sum_internal = sum(
            2 * integrand(lower_bound + 2 * i * interval_size) +
            4 * integrand(lower_bound + (2 * i + 1) * interval_size)
            for i in range(1, number_of_double_intervals)
        )

        # Agregamos los extremos y el primer punto impar
        sum_total = (
            sum_internal
            + 4 * integrand(lower_bound + interval_size)
            + integrand(lower_bound)
            + integrand(upper_bound)
        )

        return sum_total * interval_size / 3
    

class QuasiMonteCarloIntegrator1D(Integrator1D):
    def __init__(self, number_of_evaluation_points):
        super().__init__()
        self.number_of_evaluation_points = number_of_evaluation_points


    def integrate(self, integrand, lower_bound, upper_bound):
        domain_size = upper_bound - lower_bound
        sum_values = 0.0

        for i in range(self.number_of_evaluation_points):
            # x_i = (2i + 0) / (2n)
            uniform_sample = (2.0 * i + 0.0) / (2.0 * self.number_of_evaluation_points)
            argument = lower_bound + uniform_sample * domain_size
            value = integrand(argument)
            sum_values += value

        return sum_values / self.number_of_evaluation_points * domain_size
    

class QuasiMonteCarloIntegrator1DWithStreams(Integrator1D):
    def __init__(self, number_of_evaluation_points):
        super().__init__()
        self.number_of_evaluation_points = number_of_evaluation_points

    def integrate(self, integrand, lower_bound, upper_bound):
        domain_size = upper_bound - lower_bound
        #Here we sum by 0.0 cause the code is in the primitive code from java, and the result is an int cause i is an int 
        random_numbers =  [(2.0 * i + 0.0) / (2 * self.number_of_evaluation_points) for i in range(self.number_of_evaluation_points)]
        values = [integrand(lower_bound + x * domain_size) for x in random_numbers]

        return sum(values)/self.number_of_evaluation_points*domain_size
    

class RiemannMidPointIntegrator1D(Integrator1D):
    def __init__(self, number_of_evaluation_points):
        super().__init__()
        self.number_of_evaluation_points = number_of_evaluation_points

    def integrate(self, integrand, lower_bound, upper_bound):
        domain_size = upper_bound - lower_bound
        sum_values = 0.0

        for i in range(self.number_of_evaluation_points):
            uniform_sample = (2.0*i+1.0)/(2.0*self.number_of_evaluation_points)
            argument = lower_bound + uniform_sample * domain_size
            value = integrand(argument)
            sum_values += value

        return sum_values/self.number_of_evaluation_points*domain_size 
    
class RiemannMidPointIntegrator1DWithStreams(Integrator1D):
    def __init__(self, number_of_evaluation_points):
        super().__init__()
        self.number_of_evaluation_points = number_of_evaluation_points
    
    def integrate(self, integrand, lower_bound, upper_bound):
        domain_size = upper_bound - lower_bound
        random_numbers = [(2.0 * i + 1.0) / (2 * self.number_of_evaluation_points) for i in range(self.number_of_evaluation_points)]
        values = [integrand(lower_bound + x * domain_size) for x in random_numbers]
        return sum(values)/self.number_of_evaluation_points*domain_size
    

class integrator_1D_experiment():
    def test_integrator(self, integrator, name = ""):
        integrand = lambda x: math.cos(x)
        integral_analytic = lambda x: math.sin(x)
    
        lower_bound = 0.0
        upper_bound = 5.0

        integral_value_analytic = integrand(upper_bound) - integral_analytic(lower_bound)
        integral_value_integrator = integrator.integrate(integrand, lower_bound, upper_bound)
        error = integral_value_integrator - integral_value_analytic

        print(f"{name:<42}  {integral_value_integrator:20.16f}  ± {abs(error):5.3e}")
    
    def main(self):
        print("""Testing several implementations of a 1D integrator.
            (Note that some implementations may use Kahan summation or similar techniques.)""")
        
        number_of_evaluation_points = 10001
        print(f"Number of evaluation points....: {number_of_evaluation_points} (≈ {number_of_evaluation_points:.2e})\n")

        print("Theoretical (relative) errors are:")
        print(f"\tMonte-Carlo integration.......(1/n)^0.5..: {math.pow(1.0 / number_of_evaluation_points, 0.5):.2e}")
        print(f"\tSimpson's rule integration....(1/n)^4....: {math.pow(1.0 / number_of_evaluation_points, 4.0):.2e}\n")

        integrators = [
            (SimpsonsIntegrator1D(number_of_evaluation_points), "SimpsonsIntegrator1D"),
            (SimpsonsIntegrator1DWithStreams(number_of_evaluation_points), "SimpsonsIntegrator1DWithStreams"),
            (MonteCarloIntegrator1D(number_of_evaluation_points, 3141), "MonteCarloIntegrator1D"),
            (MonteCarloIntegrator1DWithStreams(number_of_evaluation_points, 3141), "MonteCarloIntegrator1DWithStreams"),
        ]

        for integrator, name in integrators:
            self.test_integrator(integrator, name)


def get_van_der_corput_number(index: int, base: int) -> float:
    index += 1  # Como en Java: index = index + 1

    x = 0.0
    refinement_factor = 1.0 / base

    while index > 0:
        x += (index % base) * refinement_factor
        index //= base  # División entera
        refinement_factor /= base

    return x


class montecarlo_integration_experiment():
    def main():
        function = lambda x: x*x*x
        integral_analytic = 0.25
        number_of_sample_points = 100000

        mersenne = np.random.default_rng(seed=3141)

        print("Integration errors:")
        print("n\tmersenne\tequidistant\tv.-d.-corput")

        sum_mersenne_twister = 0.0
        sum_equidistributed = 0.0
        sum_van_der_corput = 0.0

        for i in range(number_of_sample_points):
            sum_mersenne_twister += function(mersenne.random())
            sum_equidistributed += function(i/number_of_sample_points)
            sum_van_der_corput += function(get_van_der_corput_number(i, 2))

            current_number_of_samples = i+2

            integral_mersenne_twister  = sum_mersenne_twister  / number_of_sample_points
            error_mersenne_twister  = integral_mersenne_twister  - integral_analytic

            integral_equidistributed = sum_equidistributed / number_of_sample_points
            error_equidistributed = integral_equidistributed - integral_analytic

            integral_van_der_corput = sum_van_der_corput / number_of_sample_points
            error_van_der_corput = integral_van_der_corput - integral_analytic

            #Print every 100 result
            if current_number_of_samples % 100 == 0:
                print(f"{current_number_of_samples}\t"
                            f"{error_mersenne_twister:.3E}\t"
                            f"{error_equidistributed:.3E}\t"
                            f"{error_van_der_corput:.3E}")

        #Calculate the final result
        integral_mersenne_twister  = sum_mersenne_twister  / number_of_sample_points
        error_mersenne_twister  = integral_mersenne_twister  - integral_analytic

        integral_equisdistributed = sum_equidistributed / number_of_sample_points
        error_equidistributed = integral_equidistributed - integral_analytic

        integral_van_der_corput = sum_van_der_corput / number_of_sample_points
        error_van_der_corput = integral_van_der_corput - integral_analytic

        print("\nFinal results:")
        print(f"Pseudo RNG....: {integral_mersenne_twister:.6f}\t error: {error_mersenne_twister :.6f}")
        print(f"Equidistri....: {integral_equidistributed:.6f}\t error: {error_equidistributed:.6f}")
        print(f"v.d.Corput....: {integral_van_der_corput:.6f}\t error: {error_van_der_corput :.6f}")


def get_monte_carlo_approx_pi(number_of_simulations):
    number_of_points_inside_UnCircle = 0
    
    for i in range(number_of_simulations):
        x = 2 * (np.random.random()-0.5)
        y = 2 * (np.random.random()-0.5)

        if (x*x + y*y) < 1.0:
            number_of_points_inside_UnCircle += 1

    area_of_unitCirc = 4*number_of_points_inside_UnCircle/number_of_simulations

    pi = area_of_unitCirc

    return pi


# Clase que sobrescribe integrate y usa una fábrica de generadores
class MonteCarloIntegrator1DFromRandomGenerator1D(Integrator1D):
    def __init__(self, number_of_evaluation_points, rng_factory: Callable[[], Callable[[], float]]):
        self.number_of_evaluation_points = number_of_evaluation_points
        self.rng_factory = rng_factory  # Función que devuelve una función generadora aleatoria

    def integrate(self, integrand, lower_bound, upper_bound):
        rng = self.rng_factory()  # Obtener generador uniforme en [0, 1)
        domain_size = upper_bound - lower_bound

        total = 0.0
        for _ in range(self.number_of_evaluation_points):
            random_number = rng()
            argument = lower_bound + random_number * domain_size
            value = integrand(argument)
            total += value

        return (total / self.number_of_evaluation_points) * domain_size



class Integrator1DExperiment(Integrator1D):
    def test_integrator(self, integrator, name = ""):
        integrand = lambda x: math.cos(x)
        integral_analytic = lambda x: math.sin(x)

        lower_bound = 0.0
        upper_bound = 5.0

        integral_value_analytic = integral_analytic(upper_bound) - integral_analytic(lower_bound)
        integral_value_integrator = integrator.integrate(integrand, lower_bound, upper_bound)
        error = integral_value_integrator - integral_value_analytic

        print(f"{name:<42}  {integral_value_integrator:20.16f}  ± {abs(error):5.3e}")
    
    def __init__(self):
        print("""Testing several implementations of a 1D integrator.
              (Note that some implementations may use Kahan summation or similar techniques.)""")
        
        number_of_evaluation_points = 10001
        print(f"Number of evaluation points....: {number_of_evaluation_points} (≈ {number_of_evaluation_points:.2e})\n")

        print("Theoretical (relative) errors are:")
        print(f"\tMonte-Carlo integration.......(1/n)^0.5..: {math.pow(1.0 / number_of_evaluation_points, 0.5):.2e}")
        print(f"\tSimpson's rule integration....(1/n)^4....: {math.pow(1.0 / number_of_evaluation_points, 4.0):.2e}\n")

        integrators = [
            (SimpsonsIntegrator1D(number_of_evaluation_points), 'SimpsonsIntegrator1D'),
            (SimpsonsIntegrator1DWithStreams(number_of_evaluation_points), 'SimpsonsIntegrator1DWithStreams'),
            (MonteCarloIntegrator1D(number_of_evaluation_points, 3141), 'MonteCarloIntegrator1D'),
            (MonteCarloIntegrator1DWithStreams(number_of_evaluation_points, 3141), 'MonteCarloIntegrator1DWithStreams'),
            (QuasiMonteCarloIntegrator1D(number_of_evaluation_points), 'QuasiMonteCarloIntegrator1D'),
            (QuasiMonteCarloIntegrator1DWithStreams(number_of_evaluation_points), 'QuasiMonteCarloIntegrator1DWithStreams'),
            (RiemannMidPointIntegrator1D(number_of_evaluation_points), 'RiemannMidPointIntegrator1D'),
            (RiemannMidPointIntegrator1DWithStreams(number_of_evaluation_points), 'RiemannMidPointIntegrator1DWithStreams'),
            (MonteCarloIntegrator1DFromRandomGenerator1D(number_of_evaluation_points, lambda: vdcs.VanDerCorputSequence(base=2).nextDouble), 'MonteCarloIntegrator1DFromRandomGenerator1D')
        ]

        for integrator, name in integrators:
            self.test_integrator(integrator, name)
