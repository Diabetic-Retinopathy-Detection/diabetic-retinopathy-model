from __future__ import annotations

import torch

from dr_model.config import Settings
from dr_model.model.backbone import PatchSampler, ViTBackbone
from dr_model.model.pretrain import Pretrainer, _build_mlp, concat_all_gather

BATCH = 2
IMG_SIZE = 224


def _config(**kwargs: object) -> Settings:
    defaults: dict[str, object] = {
        "patch_size": 16,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 4.0,
        "num_classes": 5,
        "image_size": (IMG_SIZE, IMG_SIZE),
        "dim": 32,
        "mlp_dim": 128,
        "temperature": 1.0,
        "saliency_threshold": 0.25,
        "pool_mode": "max",
    }
    defaults.update(kwargs)
    return Settings(**defaults)  # type: ignore[arg-type]


class TestBuildMLP:
    def test_three_layer(self) -> None:
        mlp = _build_mlp(3, 64, 128, 32)
        x = torch.randn(4, 64)
        out = mlp(x)
        assert out.shape == (4, 32)

    def test_two_layer(self) -> None:
        mlp = _build_mlp(2, 32, 128, 32)
        x = torch.randn(4, 32)
        out = mlp(x)
        assert out.shape == (4, 32)

    def test_no_last_bn(self) -> None:
        mlp = _build_mlp(3, 64, 128, 32, last_bn=False)
        x = torch.randn(4, 64)
        out = mlp(x)
        assert out.shape == (4, 32)


class TestConcatAllGather:
    def test_single_gpu(self) -> None:
        t = torch.randn(4, 8)
        out = concat_all_gather(t)
        assert out.shape == (4, 8)
        assert torch.equal(out, t)


class TestPatchSampler:
    def test_selects_top_k(self) -> None:
        sampler = PatchSampler(patch_size=16, mask_ratio=0.25)
        pmap = torch.randn(2, 1, 14, 14)
        idx = sampler(pmap)
        assert idx.shape == (2, 148)
        assert (idx[:, 0] == 0).all()


class TestPretrainerInit:
    def test_creates_both_encoders(self) -> None:
        config = _config()
        model = Pretrainer(config)
        assert hasattr(model, "base_encoder")
        assert hasattr(model, "momentum_encoder")

    def test_momentum_no_grad(self) -> None:
        config = _config()
        model = Pretrainer(config)
        for p in model.momentum_encoder.parameters():
            assert not p.requires_grad

    def test_momentum_initialized_from_base(self) -> None:
        config = _config()
        model = Pretrainer(config)
        for pb, pm in zip(model.base_encoder.parameters(), model.momentum_encoder.parameters(), strict=True):
            assert torch.equal(pb.data, pm.data)

    def test_projectors_replaced(self) -> None:
        config = _config()
        model = Pretrainer(config)
        head = model.base_encoder.head
        assert isinstance(head, torch.nn.Sequential)
        assert isinstance(model.predictor, torch.nn.Sequential)

    def test_saliency_segmentor(self) -> None:
        config = _config()
        model = Pretrainer(config)
        assert isinstance(model.saliency_segmentor, torch.nn.Sequential)


class TestPretrainerForward:
    def test_returns_finite_losses(self) -> None:
        config = _config()
        model = Pretrainer(config)
        x1 = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        x2 = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        m1 = torch.rand(BATCH, 1, IMG_SIZE, IMG_SIZE)
        m2 = torch.rand(BATCH, 1, IMG_SIZE, IMG_SIZE)

        cl_loss, sp_loss = model(x1, x2, m1, m2, momentum_m=0.996)

        assert cl_loss.shape == ()
        assert sp_loss.shape == ()
        assert torch.isfinite(cl_loss)
        assert torch.isfinite(sp_loss)

    def test_momentum_updates(self) -> None:
        config = _config()
        model = Pretrainer(config)

        with torch.no_grad():
            for p in model.base_encoder.parameters():
                p.add_(0.1)

        before = [p.data.clone() for p in model.momentum_encoder.parameters()]
        model._update_momentum_encoder(0.996)
        after = list(model.momentum_encoder.parameters())

        for b, a in zip(before, after, strict=True):
            assert not torch.equal(b, a)


class TestViTBackboneFeatures:
    def test_forward_features_shape(self) -> None:
        config = _config()
        model = ViTBackbone(config)
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        features = model.forward_features(x)
        assert features.shape == (BATCH, 197, 64)

    def test_forward_features_with_pmap(self) -> None:
        config = _config()
        model = ViTBackbone(config)
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        pmap = torch.randn(BATCH, 1, 14, 14)
        features = model.forward_features(x, pmap=pmap)
        assert features.shape == (BATCH, 148, 64)
