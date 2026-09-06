"""Unit tests for ViTBackbone."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dr_model.config import Settings
from dr_model.model.backbone import ViTBackbone, interpolate_pos_embed


class TestViTBackboneOutputShape:
    """Verify forward pass produces correct logit shapes."""

    def test_square_input(self) -> None:
        settings = Settings()
        model = ViTBackbone(settings)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 5)

    def test_rectangular_input(self) -> None:
        settings = Settings(image_size=(224, 192))
        model = ViTBackbone(settings)
        x = torch.randn(1, 3, 224, 192)
        out = model(x)
        assert out.shape == (1, 5)

    def test_batch_dimension(self) -> None:
        settings = Settings()
        model = ViTBackbone(settings)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 5)


class TestHigherResolutionEval:
    """Evaluate a 224-trained checkpoint at a higher resolution (e.g. 384)."""

    def test_input_size_override_sizes_pos_embed(self) -> None:
        model = ViTBackbone(Settings(), input_size=384)
        assert model.pos_embed.shape[1] == 24 * 24 + 1

    def test_interpolate_pos_embed(self) -> None:
        settings = Settings()
        model = ViTBackbone(settings)
        model2 = ViTBackbone(settings, input_size=384)

        interpolated = interpolate_pos_embed(model.pos_embed, 24 * 24)
        assert interpolated.shape == model2.pos_embed.shape
        assert torch.equal(interpolated[:, 0:1], model.pos_embed[:, 0:1])

    def test_forward_at_384_with_interpolated_embed(self) -> None:
        settings = Settings()
        model = ViTBackbone(settings)
        model2 = ViTBackbone(settings, input_size=384)
        model2.load_state_dict(
            {k: interpolate_pos_embed(v, 24 * 24) if k == "pos_embed" else v for k, v in model.state_dict().items()},
            strict=False,
        )
        out = model2(torch.randn(1, 3, 384, 384))
        assert out.shape == (1, 5)

    def test_interpolate_returns_unchanged_when_grid_matches(self) -> None:
        model = ViTBackbone(Settings(), input_size=224)
        assert torch.equal(interpolate_pos_embed(model.pos_embed, 14 * 14), model.pos_embed)


class TestLoadCheckpoint:
    """Verify checkpoint loading behaviour."""

    def test_missing_file_raises(self) -> None:
        settings = Settings()
        model = ViTBackbone(settings)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            model.load_checkpoint("/nonexistent/path.pt")

    def test_load_bare_state_dict(self, tmp_path: Path) -> None:
        settings = Settings()
        model = ViTBackbone(settings)
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        model2 = ViTBackbone(settings)
        model2.load_checkpoint(path)
        assert model2.cls_token.shape == model.cls_token.shape

    def test_load_wrapped_state_dict(self, tmp_path: Path) -> None:
        settings = Settings()
        model = ViTBackbone(settings)
        path = tmp_path / "model.pt"
        torch.save({"state_dict": model.state_dict(), "model_config": {}}, path)

        model2 = ViTBackbone(settings)
        model2.load_checkpoint(path)
        assert model2.cls_token.shape == model.cls_token.shape

    def test_strict_false_allows_head_mismatch(self, tmp_path: Path) -> None:
        settings5 = Settings()
        model5 = ViTBackbone(settings5)
        path = tmp_path / "model.pt"
        torch.save(model5.state_dict(), path)

        settings3 = Settings(num_classes=3)
        model3 = ViTBackbone(settings3)
        model3.load_checkpoint(path, strict=False)
        assert model3.head.out_features == 3


class TestComputedModelName:
    """Verify the auto-generated model_name."""

    def test_default_name(self) -> None:
        settings = Settings()
        assert settings.model_name == "vit_p16_e768_d12_h12_c5"

    def test_custom_architecture_name(self) -> None:
        settings = Settings(patch_size=32, embed_dim=384, depth=6, num_heads=6, num_classes=3)
        assert settings.model_name == "vit_p32_e384_d6_h6_c3"


class TestImageSizeValidator:
    """Verify image_size is normalised to a tuple."""

    def test_scalar_becomes_square(self) -> None:
        settings = Settings(image_size=256)  # type: ignore[arg-type]
        assert settings.image_size == (256, 256)

    def test_tuple_passthrough(self) -> None:
        settings = Settings(image_size=(224, 192))
        assert settings.image_size == (224, 192)
