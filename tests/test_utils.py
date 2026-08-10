"""Tests for device resolution and determinism utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from dr_model.utils.determinism import (
    setup_cublas_workspace,
    setup_determinism,
    worker_init_fn,
)
from dr_model.utils.device import resolve_device


class TestResolveDevice:
    def test_auto_returns_device(self) -> None:
        device = resolve_device("auto")
        assert isinstance(device, torch.device)

    def test_explicit_cpu(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_explicit_mps(self) -> None:
        assert resolve_device("mps") == torch.device("mps")


class TestSetupDeterminism:
    def test_torch_seed_is_set(self) -> None:
        setup_determinism(seed=99)
        assert torch.initial_seed() == 99

    def test_random_seed_is_set(self) -> None:
        setup_determinism(seed=42)
        a = random.randint(0, 2**31)  # noqa: S311
        random.seed(42)
        b = random.randint(0, 2**31)  # noqa: S311
        assert a == b

    def test_numpy_seed_is_set(self) -> None:
        setup_determinism(seed=7)
        a = np.random.randint(0, 2**31)
        np.random.seed(7)
        b = np.random.randint(0, 2**31)
        assert a == b

    def test_deterministic_algorithms_off_by_default(self) -> None:
        # Default call should NOT enable deterministic algorithms
        setup_determinism(seed=1)
        assert not torch.are_deterministic_algorithms_enabled()

    def test_deterministic_algorithms_optional(self) -> None:
        setup_determinism(seed=1, deterministic_algorithms=True)
        assert torch.are_deterministic_algorithms_enabled()
        # Reset so it doesn't affect other tests
        torch.use_deterministic_algorithms(False)

    def test_returns_seed(self) -> None:
        result = setup_determinism(seed=42)
        assert result == 42

    def test_deterministic_sets_cublas_workspace(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        setup_determinism(seed=1, deterministic_algorithms=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        torch.use_deterministic_algorithms(False)

    def test_default_does_not_set_cublas_workspace(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        setup_determinism(seed=1)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


class TestWorkerInitFn:
    def test_reseeds_with_seed_plus_id(self) -> None:
        setup_determinism(seed=100)
        worker_init_fn(worker_id=5)
        seed_state = random.getstate()
        random.seed(105)
        assert random.getstate() == seed_state

    def test_noop_when_not_called(self) -> None:
        """worker_init_fn without prior setup_determinism should not crash."""
        # _SEED is None from import
        worker_init_fn(worker_id=0)


class TestSetupCublasWorkspace:
    def test_sets_env_var(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        setup_cublas_workspace()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_does_not_overwrite(self) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":123:4"
        setup_cublas_workspace()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":123:4"
