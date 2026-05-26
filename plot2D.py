import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

class Named:
    def __init__(self, name: str, function: Callable[[float], float]):
        self.name = name
        self.function = function

    def get(self):
        return self.function

    def getName(self):
        return self.name


class Plotable2D:
    """Representa una función o serie 2D que puede graficarse."""
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, name: str = "", style: Optional[dict] = None):
        self.x_values = x_values
        self.y_values = y_values
        self.name = name
        self.style = style or {}

    def getSeries(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_values, self.y_values

    def getName(self) -> str:
        return self.name


class Plot2D:
    """Versión Python simplificada de net.finmath.plots.Plot2D."""
    
    def __init__(self, 
                 xmin: Optional[float] = None,
                 xmax: Optional[float] = None,
                 numberOfPointsX: Optional[int] = None,
                 functions: Optional[List[Named]] = None,
                 plotables: Optional[List[Plotable2D]] = None):
        
        if plotables is not None:
            self.plotables = plotables
        else:
            self.plotables = []
            if functions is not None and xmin is not None and xmax is not None and numberOfPointsX is not None:
                x = np.linspace(xmin, xmax, numberOfPointsX)
                for namedFunc in functions:
                    f = namedFunc.get()
                    y = np.array([f(xi) for xi in x])
                    self.plotables.append(Plotable2D(x, y, name=namedFunc.getName()))

        # Configuración de estilo
        self.title = ""
        self.xAxisLabel = "x"
        self.yAxisLabel = "y"
        self.isLegendVisible = False
        self.xRange = None
        self.yRange = None
        self.figure = None
        self.ax = None

    # =============== Métodos principales ===================
    def show(self):
        self._init_plot()
        plt.show()

    def saveAsPNG(self, filename: str, width: int = 800, height: int = 400):
        self._init_plot()
        self.figure.set_size_inches(width / 100, height / 100)
        self.figure.savefig(filename, dpi=100)

    def _init_plot(self):
        if self.figure is None or self.ax is None:
            self.figure, self.ax = plt.subplots(figsize=(8, 4))

        self.ax.clear()
        for i, plotable in enumerate(self.plotables):
            x, y = plotable.getSeries()
            style = plotable.style
            label = plotable.getName() if self.isLegendVisible else None
            color = style.get("color", None)
            linestyle = style.get("linestyle", "-")
            linewidth = style.get("linewidth", 1.5)
            self.ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xAxisLabel)
        self.ax.set_ylabel(self.yAxisLabel)
        if self.xRange:
            self.ax.set_xlim(*self.xRange)
        if self.yRange:
            self.ax.set_ylim(*self.yRange)
        if self.isLegendVisible:
            self.ax.legend()

    # =============== Setters fluidos ===================
    def setTitle(self, title: str):
        self.title = title
        return self

    def setXAxisLabel(self, label: str):
        self.xAxisLabel = label
        return self

    def setYAxisLabel(self, label: str):
        self.yAxisLabel = label
        return self

    def setIsLegendVisible(self, visible: bool):
        self.isLegendVisible = visible
        return self

    def setXRange(self, xmin: float, xmax: float):
        self.xRange = (xmin, xmax)
        return self

    def setYRange(self, ymin: float, ymax: float):
        self.yRange = (ymin, ymax)
        return self

    def update(self, plotables: List[Plotable2D]):
        self.plotables = plotables
        return self
