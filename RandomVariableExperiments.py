import numpy as np

class RandomVariable:
    def __init__(self, values, precision='double'):
        self.values = np.array(values, dtype=np.float64 if precision == 'double' else np.float32)

    def get_realizations(self):
        return self.values

    def choose(self, value_if_positive, value_if_negative):
        return RandomVariable(np.where(self.values >= 0, value_if_positive, value_if_negative))

    def apply(self, func):
        return RandomVariable(np.vectorize(func)(self.values))

    def mult(self, other):
        return RandomVariable(self.values * other.values)

    def expectation(self):
        return np.mean(self.values)

def print_moments(random_variable):
    print(random_variable.__class__.__name__)
    print(random_variable.get_realizations())

    a = random_variable.choose(1.0, -1.0)
    b = random_variable.apply(np.sign)
    c = random_variable.apply(lambda x: 1.0 if x >= 0 else -1.0)

    print(a.get_realizations())
    print(b.get_realizations())
    print(c.get_realizations())

    class Signum:
        def __call__(self, x):
            if x > 0: return 1.0
            if x < 0: return -1.0
            return 0.0

    d = random_variable.apply(Signum())
    print(d.get_realizations())

    # E(X)
    expectation = random_variable.expectation()
    print("\tE(X)   =", expectation)

    # E(X^2)
    value_squared = random_variable.mult(random_variable)
    expectation_of_squared = value_squared.expectation()
    print("\tE(X^2) =", expectation_of_squared)
    print()

if __name__ == "__main__":
    rv_double = RandomVariable([-1.0/3.0, -1.0/3.0, 0.0/3.0, 2.0/3.0], precision='double')
    print_moments(rv_double)

    rv_single = RandomVariable([-1.0/3.0, -1.0/3.0, 2.0/3.0], precision='float')
    print_moments(rv_single)
