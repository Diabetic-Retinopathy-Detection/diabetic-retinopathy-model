from __future__ import annotations

from collections.abc import Generator
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import dr_model.serve.app as serve


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    monkeypatch.setattr(serve, "load_model", lambda: None)
    with TestClient(serve.app) as test_client:
        yield test_client


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_image(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    probabilities = {"No DR": 0.1, "Mild": 0.2, "Moderate": 0.3, "Severe": 0.15, "Proliferative DR": 0.25}
    monkeypatch.setattr(serve, "predict", lambda _image: probabilities)

    response = client.post(
        "/predict",
        files={"file": ("fundus.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "predicted_label": "Moderate",
        "probabilities": probabilities,
    }


def test_predict_rejects_unsupported_content_type(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("fundus.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported image type."


def test_predict_rejects_invalid_image(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("fundus.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file."
