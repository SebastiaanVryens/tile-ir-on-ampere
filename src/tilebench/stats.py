"""Summary statistics for timing samples."""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Summary:
    median: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def summarize(samples: Sequence[float]) -> Summary:
    q1, median, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    return Summary(median=median, q1=q1, q3=q3)
