"""Tests for device resolution and determinism utilities."""

from __future__ import annotations

import os

import torch

from dr_model.utils.determinism import setup_cublas_workspace, setup_determinism
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
    def test_sets_cublas_workspace(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        setup_determinism(seed=123)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_seed_is_set(self) -> None:
        setup_determinism(seed=99)
        # torch.manual_seed was called — just verify no error
        assert torch.initial_seed() == 99


class TestSetupCublasWorkspace:
    def test_sets_env_var(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        setup_cublas_workspace()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

    def test_does_not_overwrite(self) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":123:4"
        setup_cublas_workspace()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":123:4"
