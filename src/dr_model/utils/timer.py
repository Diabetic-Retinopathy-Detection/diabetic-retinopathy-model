"""Lightweight timer for tracking epoch and total training time."""

from __future__ import annotations

import time


class Timer:
    """A stopwatch that supports lap and total queries.

    Usage::

        timer = Timer()
        # ... do work ...
        lap_time = timer.lap()   # seconds since creation or last lap
        # ... more work ...
        total_time = timer.total()  # seconds since creation
    """

    def __init__(self) -> None:
        self._lap_start = time.perf_counter()
        self._accumulated = 0.0

    def lap(self) -> float:
        """Return elapsed seconds since the last ``lap()`` or creation, and reset."""
        now = time.perf_counter()
        elapsed = now - self._lap_start
        self._accumulated += elapsed
        self._lap_start = now
        return elapsed

    def total(self) -> float:
        """Return total elapsed seconds since creation (does not reset)."""
        return self._accumulated + (time.perf_counter() - self._lap_start)
