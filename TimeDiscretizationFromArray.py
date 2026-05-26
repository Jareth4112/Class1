import numpy as np

class TimeDiscretizationFromArray:
    def __init__(self, initial_time: float, number_of_steps: int, dt: float):
        self.times = np.array([initial_time + i * dt for i in range(number_of_steps + 1)])
        self.dt = dt

    def getNumberOfTimes(self):
        return len(self.times)

    def getTime(self, index: int):
        return self.times[index]

    def getTimeIndex(self, time: float):
        return int(round((time - self.times[0]) / self.dt))

    def __iter__(self):
        return iter(self.times)