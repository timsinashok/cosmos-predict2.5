from __future__ import annotations

import logging
import os
import time
import sys
from typing import Any
import warnings

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.predictor import VideoRiskPredictor, load_from_env
from backend.agent import IssueRequest


# Use uvicorn's error logger so messages show in the terminal.
# Ensure INFO logs are enabled even if the root logger is higher.
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

app = FastAPI(title="Cosmos Risk API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PREDICTOR: VideoRiskPredictor | None = None
AGENT = None


def _print_predict_block(filename: str, out: dict[str, Any], elapsed_s: float) -> None:
    risk = out.get("risk", "?")
    status = out.get("status", "?")
    impact_in_s = out.get("impact_in_s", None)
    conf_pct = out.get("confidence_percent", None)
    p_pos = out.get("confidence", None)
    margin = out.get("xgb_margin", None)
    emb_dim = out.get("embedding_dim", None)

    # Demo-friendly block; keep it visually distinct from uvicorn access logs.
    lines = [
        "",
        "==================== COSMOS RISK PREDICTION ====================",
        f"Video      : {filename}",
        f"Risk       : {risk}",
        f"Status     : {status}",
        f"Impact In  : {impact_in_s:.2f}s" if isinstance(impact_in_s, (int, float)) else "Impact In  : n/a",
        f"Confidence : {conf_pct:.3f}%" if isinstance(conf_pct, (int, float)) else "Confidence : n/a",
        "---------------------- raw model outputs -----------------------",
        f"p_positive  : {p_pos:.6f}" if isinstance(p_pos, (int, float)) else "p_positive  : n/a",
        f"margin      : {margin}" if margin is not None else "margin      : n/a",
        f"embed_dim   : {emb_dim}" if emb_dim is not None else "embed_dim   : n/a",
        "--------------------------- timing ----------------------------",
        f"elapsed     : {elapsed_s:.3f}s",
        "===============================================================",
        "",
    ]
    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()


@app.on_event("startup")
def _startup():
    global PREDICTOR, AGENT
    # Reduce very noisy Torch warning about TF32 deprecation.
    warnings.filterwarnings(
        "ignore",
        message=r".*Please use the new API settings to control TF32 behavior.*",
    )
    try:
        PREDICTOR = load_from_env()
    except Exception as e:
        # Keep server up but error on /predict with actionable message
        PREDICTOR = None
        app.state.startup_error = str(e)

    # Optional agent: requires GOOGLE_API_KEY + pydantic-ai dependency.
    try:
        from backend.agent import build_agent

        AGENT = build_agent()
    except Exception as e:
        AGENT = None
        app.state.agent_startup_error = str(e)


@app.get("/health")
def health() -> dict[str, Any]:
    ok = PREDICTOR is not None
    return {
        "ok": ok,
        "startup_error": getattr(app.state, "startup_error", None),
        "agent_ok": AGENT is not None,
        "agent_startup_error": getattr(app.state, "agent_startup_error", None),
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

    t0 = time.time()
    try:
        out = PREDICTOR.predict_video_bytes(data, suffix=suffix)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {e}")
    elapsed_s = time.time() - t0

    # Hackathon/demo-friendly printout (more visible than logs).
    try:
        _print_predict_block(video.filename, out, elapsed_s)
    except Exception:
        # Never break the request if printing fails.
        pass

    # Frontend-friendly fields (plus raw fields)
    return {
        "risk": out["risk"],
        "status": out["status"],
        "impact_in": f'{out["impact_in_s"]:.1f}s',
        "confidence": f'{out["confidence_percent"]:.1f}%',
        "raw": out,
    }


@app.post("/agent/recommend")
async def agent_recommend(req: IssueRequest) -> dict[str, Any]:
    """
    PydanticAI action recommender.
    Send dummy telemetry + detector outputs; get recommended action.
    """
    if AGENT is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(app.state, "agent_startup_error", "Agent not initialized (set GOOGLE_API_KEY)."),
        )

    try:
        result = await AGENT.run(req.model_dump())
        rec = result.output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    # frontend-friendly strings + raw structured output
    return {
        "action": rec.action,
        "priority": rec.priority,
        "trigger_kill_switch": rec.trigger_kill_switch,
        "headline": rec.headline,
        "impact_in": (f"{rec.impact_in_s:.1f}s" if rec.impact_in_s is not None else None),
        "confidence": f"{rec.confidence * 100.0:.1f}%",
        "operator_instructions": rec.operator_instructions,
        "raw": rec.model_dump(),
    }

