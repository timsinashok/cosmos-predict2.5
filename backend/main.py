from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import VideoRiskPredictor, load_from_env


app = FastAPI(title="Cosmos Risk API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PREDICTOR: VideoRiskPredictor | None = None


@app.on_event("startup")
def _startup():
    global PREDICTOR
    try:
        PREDICTOR = load_from_env()
    except Exception as e:
        # Keep server up but error on /predict with actionable message
        PREDICTOR = None
        app.state.startup_error = str(e)


@app.get("/health")
def health() -> dict[str, Any]:
    ok = PREDICTOR is not None
    return {
        "ok": ok,
        "startup_error": getattr(app.state, "startup_error", None),
    }


@app.post("/predict")
async def predict(video: UploadFile = File(...)) -> dict[str, Any]:
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail=getattr(app.state, "startup_error", "Predictor not initialized"))

    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = os.path.splitext(video.filename)[1].lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        # we still try, but warn client
        suffix = ".mp4"

    data = await video.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        out = PREDICTOR.predict_video_bytes(data, suffix=suffix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")

    # Frontend-friendly fields (plus raw fields)
    return {
        "risk": out["risk"],
        "status": out["status"],
        "impact_in": f'{out["impact_in_s"]:.1f}s',
        "confidence": f'{out["confidence_percent"]:.1f}%',
        "raw": out,
    }

