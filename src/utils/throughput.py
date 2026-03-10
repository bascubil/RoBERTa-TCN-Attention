"""Simple throughput timer for steps/sec tracking."""

from __future__ import annotations

import time


class ThroughputTimer:
    def __init__(self) -> None:
        self._start = None
        self.elapsed = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self, steps: int) -> float:
        if self._start is None:
            return 0.0
        self.elapsed = time.perf_counter() - self._start
        if self.elapsed <= 0:
            return 0.0
        return float(steps) / self.elapsed

