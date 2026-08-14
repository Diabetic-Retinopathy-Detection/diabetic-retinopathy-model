from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from dr_model.config import Settings
from dr_model.inference import load_model, predict


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(
    title="Diabetic Retinopathy Model API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    probabilities = predict(image)
    predicted_label = max(probabilities, key=lambda label: probabilities[label])

    return {
        "status": "ok",
        "predicted_label": predicted_label,
        "probabilities": probabilities,
    }


def main() -> None:
    settings = Settings()
    uvicorn.run(
        "dr_model.serve.app:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
    )


if __name__ == "__main__":
    main()
