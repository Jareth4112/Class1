import numpy as np

class RandomVariableFromDoubleArray:
    def __init__(self, time, values):
        """
        time: float, filtration time
        values: float (deterministic) or np.ndarray (stochastic)
        """
        self.time = time
        if isinstance(values, (float, int)):
            self.values = float(values)
            self.deterministic = True
        else:
            self.values = np.array(values, dtype=float)
            self.deterministic = False

    def get_filtration_time(self):
        return self.time

    def is_deterministic(self):
        return self.deterministic

    def size(self):
        return 1 if self.deterministic else len(self.values)

    def get(self, idx):
        if self.deterministic:
            return self.values
        else:
            return self.values[idx]

    def get_realizations(self):
        if self.deterministic:
            return np.array([self.values])
        else:
            return self.values.copy()

    def get_average(self):
        if self.deterministic:
            return self.values
        else:
            return np.mean(self.values)

    def get_variance(self):
        if self.deterministic or self.size() == 1:
            return 0.0
        else:
            return np.var(self.values)

    def get_standard_deviation(self):
        return np.sqrt(self.get_variance())

    def get_min(self):
        if self.deterministic:
            return self.values
        else:
            return np.min(self.values)

    def get_max(self):
        if self.deterministic:
            return self.values
        else:
            return np.max(self.values)

    def apply(self, func):
        if self.deterministic:
            return RandomVariableFromDoubleArray(self.time, func(self.values))
        else:
            return RandomVariableFromDoubleArray(self.time, func(self.values))

    def add(self, other):
        if isinstance(other, RandomVariableFromDoubleArray):
            if self.deterministic and other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values + other.values)
            elif self.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values + other.values)
            elif other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values + other.values)
            else:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values + other.values)
        else:
            # other is a scalar
            return RandomVariableFromDoubleArray(self.time, self.values + other)

    def sub(self, other):
        if isinstance(other, RandomVariableFromDoubleArray):
            if self.deterministic and other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values - other.values)
            elif self.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values - other.values)
            elif other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values - other.values)
            else:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values - other.values)
        else:
            return RandomVariableFromDoubleArray(self.time, self.values - other)

    def mult(self, other):
        if isinstance(other, RandomVariableFromDoubleArray):
            if self.deterministic and other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values * other.values)
            elif self.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values * other.values)
            elif other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values * other.values)
            else:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values * other.values)
        else:
            return RandomVariableFromDoubleArray(self.time, self.values * other)

    def div(self, other):
        if isinstance(other, RandomVariableFromDoubleArray):
            if self.deterministic and other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values / other.values)
            elif self.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values / other.values)
            elif other.deterministic:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values / other.values)
            else:
                return RandomVariableFromDoubleArray(max(self.time, other.time), self.values / other.values)
        else:
            return RandomVariableFromDoubleArray(self.time, self.values / other)

    def pow(self, exponent):
        return RandomVariableFromDoubleArray(self.time, np.power(self.values, exponent))

    def exp(self):
        return RandomVariableFromDoubleArray(self.time, np.exp(self.values))

    def log(self):
        return RandomVariableFromDoubleArray(self.time, np.log(self.values))

    def sqrt(self):
        return RandomVariableFromDoubleArray(self.time, np.sqrt(self.values))

    def abs(self):
        return RandomVariableFromDoubleArray(self.time, np.abs(self.values))

    def __repr__(self):
        return f"RandomVariableFromDoubleArray(time={self.time}, values={self.values})"