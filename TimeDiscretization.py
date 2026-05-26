"""
A lightweight Python port of key classes from net.finmath.time
focused on time discretizations and simple schedule generation.

Module: pytimes
- TimeDiscretization (abstract base)
- TimeDiscretizationFromArray
- Tenor (simple dataclass)
- TenorFromArray
- Period (represents a fixing/payment period)
- Schedule (interface) and ScheduleFromPeriods
- RegularSchedule (build from TimeDiscretization)
- ScheduleGenerator (simple generator using frequency)
- SchedulePrototype (container for metadata)
- FloatingPointDate (utility to convert float -> date)

This is intentionally lightweight and pure-python. It is
meant to be easy to extend for your specific needs.

Usage examples at bottom of file.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Iterable, Optional, Tuple, Iterator
import math

__all__ = [
    'TimeDiscretization', 'TimeDiscretizationFromArray', 'Tenor', 'TenorFromArray',
    'Period', 'Schedule', 'ScheduleFromPeriods', 'RegularSchedule', 'ScheduleGenerator',
    'SchedulePrototype', 'FloatingPointDate'
]


class TimeDiscretization:
    """Base class / interface for time discretizations.

    Methods to implement/override:
      - get_number_of_times()
      - get_time(index)
      - get_time_index(time)
      - __iter__() to iterate over times
    """

    def get_number_of_times(self) -> int:
        raise NotImplementedError

    def get_time(self, index: int) -> float:
        raise NotImplementedError

    def get_time_index(self, time: float) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[float]:
        for i in range(self.get_number_of_times()):
            yield self.get_time(i)


class TimeDiscretizationFromArray(TimeDiscretization):
    """Represents a set of discrete points in time.

    Parameters
    ----------
    initial_time : float
        typically 0.0
    number_of_steps : int
        number of steps (n). The object will contain n+1 times: 0..n
    dt : float
        step size. times are initial_time + i*dt
    times : Optional[List[float]]
        alternative explicit array of times. If provided, initial_time/number_of_steps/dt are ignored.
    tick_size : Optional[float]
        rounding quantum for times (like finmath default of 1/(365*24)).
    """

    def __init__(
        self,
        initial_time: float = 0.0,
        number_of_steps: int = 0,
        dt: float = 1.0,
        times: Optional[List[float]] = None,
        tick_size: Optional[float] = None,
    ) -> None:
        if times is not None:
            # ensure sorted unique
            self._times = sorted(float(t) for t in times)
        else:
            n = int(number_of_steps)
            self._times = [initial_time + i * dt for i in range(n + 1)]
        self._tick_size = tick_size if tick_size is not None else 1.0 / (365.0 * 24.0)

    def _round_time(self, t: float) -> float:
        if self._tick_size <= 0:
            return t
        return round(t / self._tick_size) * self._tick_size

    def get_number_of_times(self) -> int:
        return len(self._times)

    def get_time(self, index: int) -> float:
        return self._times[index]

    def get_time_index(self, time: float) -> int:
        # rounds to nearest tick and finds nearest index
        t = self._round_time(time)
        # simple linear search (fast for small arrays). Could be bisect for large arrays.
        # We return the index of the nearest time.
        # If exact match not found, return nearest index.
        best = 0
        best_diff = abs(self._times[0] - t)
        for i, val in enumerate(self._times):
            d = abs(val - t)
            if d < best_diff:
                best = i
                best_diff = d
        return best

    def __iter__(self) -> Iterator[float]:
        return iter(self._times)

    def to_numpy(self):
        try:
            import numpy as np
            return np.array(self._times)
        except Exception:
            raise RuntimeError("numpy not available")


@dataclass
class Tenor:
    """Simple tenor representation (year-fraction or label)."""

    label: str
    tenor_in_years: float


class TenorFromArray(TimeDiscretizationFromArray):
    """Time discretization constructed from actual dates and a daycount.

    For simplicity we accept a reference date and a list of actual dates or year fractions.
    This class stores times as year fractions from a reference.
    """

    def __init__(self, reference_date: date, dates: List[date], daycount_convention: str = 'ACT/365'):
        # naive implementation: ACT/365
        times = [0.0]
        for d in dates:
            delta = (d - reference_date).days
            times.append(delta / 365.0)
        super().__init__(times=times)
        self.reference_date = reference_date
        self.dates = dates
        self.daycount_convention = daycount_convention


@dataclass
class Period:
    fixing_start: float
    fixing_end: float
    payment: float
    info: Optional[dict] = None

    def __repr__(self):
        return f"Period(fixing_start={self.fixing_start}, fixing_end={self.fixing_end}, payment={self.payment})"


class Schedule(Iterable[Period]):
    """Interface for a schedule of interest rate periods with fixing and payment."""

    def __iter__(self) -> Iterator[Period]:
        raise NotImplementedError

    def get_periods(self) -> List[Period]:
        raise NotImplementedError


class ScheduleFromPeriods(Schedule):
    def __init__(self, periods: List[Period]):
        self._periods = list(periods)

    def __iter__(self) -> Iterator[Period]:
        return iter(self._periods)

    def get_periods(self) -> List[Period]:
        return list(self._periods)

    def __repr__(self):
        return f"ScheduleFromPeriods(num_periods={len(self._periods)})"


class RegularSchedule(Schedule):
    """Simple schedule generated from a time discretization.

    Each entry corresponds to a period [t_i, t_{i+1}] with payment at t_{i+1}.
    """

    def __init__(self, td: TimeDiscretization):
        times = list(td)
        periods = []
        for i in range(len(times) - 1):
            periods.append(Period(fixing_start=times[i], fixing_end=times[i + 1], payment=times[i + 1]))
        self._periods = periods

    def __iter__(self) -> Iterator[Period]:
        return iter(self._periods)

    def get_periods(self) -> List[Period]:
        return list(self._periods)


class ScheduleGenerator:
    """Generate schedules using simple metadata (frequency in years, maturity in years).

    This is a tiny helper. For production use, implement business day calendar and roll conventions.
    """

    def __init__(self, reference: float = 0.0):
        self.reference = reference

    def generate(self, frequency_in_years: float, maturity_in_years: float) -> ScheduleFromPeriods:
        if frequency_in_years <= 0:
            raise ValueError("frequency must be > 0")
        n = int(round(maturity_in_years / frequency_in_years))
        times = [self.reference + i * frequency_in_years for i in range(n + 1)]
        td = TimeDiscretizationFromArray(times=times)
        return ScheduleFromPeriods(RegularSchedule(td).get_periods())


@dataclass
class SchedulePrototype:
    """Container for schedule meta data. Deprecated in this simple port but kept for API parity."""
    frequency_in_years: float
    maturity_in_years: float
    roll_convention: Optional[str] = None


class FloatingPointDate:
    """Utility to map a floating point time (year fraction) to a real date.

    This is a naive implementation using reference date + round(years*365).
    """

    def __init__(self, reference_date: date = date.today()):
        self.reference_date = reference_date

    def to_date(self, time: float) -> date:
        days = int(round(time * 365.0))
        return self.reference_date + timedelta(days=days)

    def to_float(self, d: date) -> float:
        delta = (d - self.reference_date).days
        return delta / 365.0


# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Time discretization example
    td = TimeDiscretizationFromArray(initial_time=0.0, number_of_steps=10, dt=0.5)
    print("Times:", list(td))
    print("Number of times:", td.get_number_of_times())
    print("Time index for 1.0 ->", td.get_time_index(1.0))

    # Regular schedule
    reg = RegularSchedule(td)
    for p in reg:
        print(p)

    # Generate schedule via generator
    gen = ScheduleGenerator(reference=0.0)
    sch = gen.generate(frequency_in_years=0.5, maturity_in_years=5.0)
    print(sch)


