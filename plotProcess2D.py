import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Named:
    name: str
    func: callable

class PlotProcess2D:
    def __init__(self, timeDiscretization, process, maxNumberOfPaths=100):
        self.timeDiscretization = np.array(timeDiscretization)
        if isinstance(process, Named):
            self.process = process.func
        else:
            self.process = process
        self.maxNumberOfPaths = maxNumberOfPaths
        self.title = ""
        self.xAxisLabel = "x"
        self.yAxisLabel = "y"
        self.isLegendVisible = False
        self.colors = None

    def setTitle(self, title):
        self.title = title
        return self

    def setXAxisLabel(self, label):
        self.xAxisLabel = label
        return self

    def setYAxisLabel(self, label):
        self.yAxisLabel = label
        return self

    def setColors(self, colors):
        self.colors = colors
        return self

    def setIsLegendVisible(self, flag):
        self.isLegendVisible = flag
        return self

    def show(self):
        times = self.timeDiscretization
        # Obtener el proceso en cada instante t
        all_series = []
        for t in times:
            rv = self.process(t)
            all_series.append(rv)

        all_series = np.array(all_series).T  # shape: (num_paths, num_times)
        num_paths = min(self.maxNumberOfPaths, all_series.shape[0])

        plt.figure(figsize=(10, 5))
        for i in range(num_paths):
            plt.plot(times, all_series[i], linewidth=1)

        plt.title(self.title)
        plt.xlabel(self.xAxisLabel)
        plt.ylabel(self.yAxisLabel)
        plt.grid(True)
        plt.show()
