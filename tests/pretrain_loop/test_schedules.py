from __future__ import annotations

import pytest
import torch

from dr_model.training.pretrain_loop import _adjust_lambda_s, _adjust_lr, _adjust_momentum

from .conftest import _make_config


class TestLearningRateSchedule:
    def test_warmup_linear(self) -> None:
        config = _make_config(learning_rate=0.1, warmup_epochs=10, max_epochs=100)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        for step in range(1, 11):
            lr = _adjust_lr(dummy_optimizer, config, float(step))
            expected = 0.1 * step / 10
            assert lr == pytest.approx(expected, abs=1e-7)

    def test_cosine_decay_at_end(self) -> None:
        config = _make_config(learning_rate=0.1, warmup_epochs=10, max_epochs=100)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        lr = _adjust_lr(dummy_optimizer, config, 100.0)
        assert lr == pytest.approx(0.0, abs=1e-7)

    def test_cosine_peak(self) -> None:
        config = _make_config(learning_rate=0.1, warmup_epochs=0, max_epochs=100)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        lr = _adjust_lr(dummy_optimizer, config, 0.0)
        assert lr == pytest.approx(0.1, abs=1e-7)

    def test_lr_set_on_param_groups(self) -> None:
        config = _make_config(learning_rate=0.05, warmup_epochs=5, max_epochs=20)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.0)
        lr = _adjust_lr(dummy_optimizer, config, 2.5)
        assert dummy_optimizer.param_groups[0]["lr"] == lr


class TestMomentumSchedule:
    def test_starts_at_base(self) -> None:
        config = _make_config(momentum_base=0.99)
        m = _adjust_momentum(config, 0.0)
        assert m == pytest.approx(0.99, abs=1e-6)

    def test_ends_at_one(self) -> None:
        config = _make_config(momentum_base=0.99)
        m = _adjust_momentum(config, 1.0)
        assert m == pytest.approx(1.0, abs=1e-6)

    def test_monotonically_increases(self) -> None:
        config = _make_config(momentum_base=0.99)
        values = [_adjust_momentum(config, t) for t in [i / 10 for i in range(11)]]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-7

    def test_no_oscillation_over_many_epochs(self) -> None:
        config = _make_config(momentum_base=0.99, max_epochs=300)
        t_values = [e / 300 for e in range(301)]
        values = [_adjust_momentum(config, t) for t in t_values]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-7


class TestLambdaSSchedule:
    def test_starts_at_lambda_s(self) -> None:
        config = _make_config(lambda_s=10.0)
        v = _adjust_lambda_s(config, 0.0)
        assert v == pytest.approx(10.0, abs=1e-6)

    def test_decays_to_zero(self) -> None:
        config = _make_config(lambda_s=10.0)
        v = _adjust_lambda_s(config, 1.0)
        assert v == pytest.approx(0.0, abs=1e-6)

    def test_no_oscillation_over_many_epochs(self) -> None:
        config = _make_config(lambda_s=10.0, max_epochs=300)
        t_values = [e / 300 for e in range(301)]
        values = [_adjust_lambda_s(config, t) for t in t_values]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-7
