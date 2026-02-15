from __future__ import annotations

import os
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Action(str, Enum):
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SLOW_DOWN = "SLOW_DOWN"
    HOLD_POSITION = "HOLD_POSITION"
    ALERT_OPERATOR = "ALERT_OPERATOR"
    CONTINUE = "CONTINUE"


class IssueRequest(BaseModel):
    """
    Payload you can send from the frontend.
    Keep it flexible: include whatever telemetry/UI context you have.
    """

    issue: str = Field(..., description="Short description of the situation/problem.")

    # Optional model outputs from /predict (or any other detector)
    risk: str | None = None
    status: str | None = None
    impact_in_s: float | None = Field(None, description="Seconds until impact / intersection, if known.")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Probability of hazard (0..1).")

    # Optional extra context (dummy telemetry is fine)
    speed_mps: float | None = None
    distance_to_intersection_m: float | None = None
    location: str | None = None
    notes: dict[str, Any] | None = None


class ActionRecommendation(BaseModel):
    action: Action
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    impact_in_s: float | None = Field(None, description="Model-estimated time-to-impact, if applicable.")

    headline: str
    rationale: str
    operator_instructions: list[str]

    # Safety/controls hints for frontend
    trigger_kill_switch: bool = False
    notify_operator: bool = True


def build_agent():
    """
    Returns a configured PydanticAI Agent or raises a RuntimeError if not configured.
    Uses Gemini 2.5 Flash via Generative Language API by default.
    """
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.google import GoogleModelSettings
    except ModuleNotFoundError as e:
        raise RuntimeError("pydantic-ai is not installed. Install: pip install \"pydantic-ai-slim[google]\"") from e

    # Gemini auth: Generative Language API expects GOOGLE_API_KEY env var.
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY to use /agent/recommend (Gemini).")

    # Keep it fast + deterministic-ish.
    settings = GoogleModelSettings(
        temperature=0.2,
        max_tokens=700,
        # Disable/limit thinking on older models by setting budget to 0.
        google_thinking_config={"thinking_budget": 0},
    )

    system_prompt = (
        "You are a safety decision agent for a physical AI / vehicle safety system.\n"
        "You will receive a structured issue report + optional detector outputs.\n"
        "Return a structured action recommendation.\n\n"
        "Rules:\n"
        "- Prefer conservative actions when risk/confidence is high and impact_in_s is short.\n"
        "- If recommending EMERGENCY_STOP / kill switch, include clear operator instructions.\n"
        "- Never claim certainty. Use the provided confidence field.\n"
        "- Keep output concise and actionable.\n"
        "- This is a demo system: recommend actions, do not attempt to directly control real hardware.\n"
    )

    # Model name per PydanticAI docs: 'google-gla:<model>'
    agent = Agent(
        "google-gla:gemini-2.5-flash",
        system_prompt=system_prompt,
        result_type=ActionRecommendation,
        model_settings=settings,
    )
    return agent

