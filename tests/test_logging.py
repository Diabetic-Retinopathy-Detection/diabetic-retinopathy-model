"""Tests for experiment logging — TensorBoard and MLflow utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from dr_model.config import Settings

if TYPE_CHECKING:
    from pathlib import Path


class TestLogEpochTime:
    """Training time per epoch is logged to TensorBoard and MLflow."""

    def test_epoch_time_passed_to_writer(self, tmp_path: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        from dr_model.training.pretrain_loop import _log_epoch

        config = Settings()
        writer = SummaryWriter(log_dir=str(tmp_path / "test_logs"))
        writer.add_scalar = MagicMock()  # type: ignore[method-assign]

        _log_epoch(
            epoch=0,
            config=config,
            avg_cl=1.0,
            avg_ss=2.0,
            lr=1e-4,
            moco_m=0.99,
            t_epoch=12.34,
            writer=writer,
        )

        writer.add_scalar.assert_any_call("time/epoch", 12.34, 0)
        writer.close()

    def test_epoch_time_logged_via_mlflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dr_model.training.pretrain_loop import _log_epoch

        logged_metrics: dict[str, float] = {}

        def fake_log_metrics(metrics: dict[str, float], step: int) -> None:
            logged_metrics.update(metrics)

        monkeypatch.setattr("mlflow.log_metrics", fake_log_metrics)

        config = Settings()
        _log_epoch(
            epoch=1,
            config=config,
            avg_cl=0.5,
            avg_ss=0.3,
            lr=5e-5,
            moco_m=0.95,
            t_epoch=8.75,
            writer=None,
        )

        assert "time/epoch" in logged_metrics
        assert logged_metrics["time/epoch"] == 8.75

    def test_epoch_time_zero_when_disabled(self) -> None:
        """Both MLflow and TensorBoard disabled — no crash."""
        from dr_model.training.pretrain_loop import _log_epoch

        config = Settings(mlflow=False)

        _log_epoch(
            epoch=0,
            config=config,
            avg_cl=0.0,
            avg_ss=0.0,
            lr=0.0,
            moco_m=0.99,
            t_epoch=0.0,
            writer=None,
        )


class TestEpochTimeIntegration:
    """Smoke tests for time tracking in the pretrain loop."""

    def test_epoch_time_call_args(self, tmp_path: Path) -> None:
        from dr_model.training.pretrain_loop import _log_epoch

        config = Settings()
        writer = MagicMock()

        _log_epoch(
            epoch=0,
            config=config,
            avg_cl=0.0,
            avg_ss=0.0,
            lr=0.0,
            moco_m=0.99,
            t_epoch=1.5,
            writer=writer,
        )

        found = any(
            args[0] == "time/epoch" and args[1] == 1.5
            for call_args in writer.add_scalar.call_args_list
            for args in [call_args[0]]
        )
        assert found


class TestSystemMetadataTags:
    """Environment metadata is logged as MLflow tags on run start."""

    def test_basic_tags_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dr_model.logging.mlflow_utils import init_mlflow_run

        tags: dict[str, str] = {}
        params: dict[str, str] = {}

        monkeypatch.setattr("mlflow.set_tracking_uri", lambda _: None)
        monkeypatch.setattr("mlflow.set_experiment", lambda _: None)
        monkeypatch.setattr("mlflow.start_run", lambda **_: None)
        monkeypatch.setattr("mlflow.log_param", lambda k, v: params.update({k: str(v)}))  # type: ignore[arg-type]
        monkeypatch.setattr("mlflow.set_tag", lambda k, v: tags.update({k: str(v)}))
        monkeypatch.setattr("mlflow.active_run", lambda: None)

        config = Settings()
        init_mlflow_run(config, experiment_name="test", run_name_prefix="test")

        assert tags.get("torch_version", "").startswith(("2.",))
        assert "platform" in tags
        assert "python_version" in tags
        assert "cuda_version" in tags

    def test_gpu_tag_skipped_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dr_model.logging.mlflow_utils import init_mlflow_run

        tags: dict[str, str] = {}

        monkeypatch.setattr("mlflow.set_tracking_uri", lambda _: None)
        monkeypatch.setattr("mlflow.set_experiment", lambda _: None)
        monkeypatch.setattr("mlflow.start_run", lambda **_: None)
        monkeypatch.setattr("mlflow.log_param", lambda k, v: None)
        monkeypatch.setattr("mlflow.set_tag", lambda k, v: tags.update({k: str(v)}))
        monkeypatch.setattr("mlflow.active_run", lambda: None)

        import torch

        config = Settings()
        init_mlflow_run(config, experiment_name="test", run_name_prefix="test", device=torch.device("cpu"))

        assert "gpu_name" not in tags

    def test_no_crash_when_device_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dr_model.logging.mlflow_utils import init_mlflow_run

        monkeypatch.setattr("mlflow.set_tracking_uri", lambda _: None)
        monkeypatch.setattr("mlflow.set_experiment", lambda _: None)
        monkeypatch.setattr("mlflow.start_run", lambda **_: None)
        monkeypatch.setattr("mlflow.log_param", lambda k, v: None)
        monkeypatch.setattr("mlflow.set_tag", lambda k, v: None)
        monkeypatch.setattr("mlflow.active_run", lambda: None)

        config = Settings()
        init_mlflow_run(config, experiment_name="test", run_name_prefix="test")
