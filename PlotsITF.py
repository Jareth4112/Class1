import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class GraphStyle:
    marker_size: Optional[Tuple[int, int]] = (4, 4)
    color: Optional[str] = None
    fill_color: Optional[str] = None
    edge_color: Optional[str] = None


@dataclass
class NumberAxis:
    label: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    ticks: Optional[List[float]] = None


class PlotablePoints2D:
    """
    Representa una serie de puntos 2D (scatter o línea) en un gráfico.
    """
    def __init__(self, name: str, series: List[Point2D],
                 domain_axis: Optional[NumberAxis] = None,
                 range_axis: Optional[NumberAxis] = None,
                 style: Optional[GraphStyle] = None):
        self.name = name
        self.series = series
        self.domain_axis = domain_axis
        self.range_axis = range_axis
        self.style = style or GraphStyle()

    def plot(self, ax):
        x = [p.x for p in self.series]
        y = [p.y for p in self.series]
        ax.scatter(x, y,
                   s=self.style.marker_size[0] if self.style.marker_size else 10,
                   label=self.name,
                   c=self.style.color or "C0",
                   alpha=0.8)


class Plot2D:
    """
    Gráfico bidimensional que puede contener múltiples 'plotables' (series).
    """
    def __init__(self, plotables: List[PlotablePoints2D]):
        self.plotables = plotables
        self.title = ""
        self.x_label = ""
        self.y_label = ""

    def setTitle(self, title):
        self.title = title
        return self

    def setXAxisLabel(self, label):
        self.x_label = label
        return self

    def setYAxisLabel(self, label):
        self.y_label = label
        return self

    def show(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        for plotable in self.plotables:
            plotable.plot(ax)

        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.legend()
        plt.show()

    def update(self, new_plotables):
        """
        Equivalente a plot.update(plotables) en Java.
        """
        self.plotables = new_plotables
        return self
