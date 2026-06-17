from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from dr_model.config import Settings

app = FastAPI(
    title="Diabetic Retinopathy Model API",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
