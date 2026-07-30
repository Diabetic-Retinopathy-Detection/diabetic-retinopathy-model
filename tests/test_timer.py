"""Tests for the Timer utility."""

from __future__ import annotations

import time

from dr_model.utils.timer import Timer


class TestTimer:
    def test_lap_returns_positive(self) -> None:
        t = Timer()
        time.sleep(0.01)
        assert t.lap() > 0.0

    def test_total_geq_last_lap(self) -> None:
        t = Timer()
        time.sleep(0.01)
        lap1 = t.lap()
        time.sleep(0.01)
        lap2 = t.lap()
        total = t.total()
        assert total >= lap1 + lap2

    def test_lap_resets_internal_counter(self) -> None:
        t = Timer()
        time.sleep(0.01)
        lap1 = t.lap()
        time.sleep(0.01)
        lap2 = t.lap()
        assert lap2 < lap1 * 3
        assert lap2 > 0.0

    def test_total_without_lap(self) -> None:
        t = Timer()
        time.sleep(0.01)
        total = t.total()
        assert total > 0.0

    def test_multiple_laps(self) -> None:
        t = Timer()
        laps = []
        for _ in range(5):
            time.sleep(0.005)
            laps.append(t.lap())
        assert all(lap > 0.0 for lap in laps)
