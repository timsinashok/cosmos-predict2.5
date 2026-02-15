from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.video2world import resize_input

from new_embedding_generation import (  # type: ignore
    DEFAULT_EXP_14B,
    DEFAULT_EXP_2B,
    HF_14B_PRETRAINED,
    HF_2B_POSTTRAINED,
    _parse_hw,
    _to_bf16_minus1_1,
    load_tokenizer_only,
)


def _get_fps(meta: dict) -> float | None:
    for k in ("fps", "video_fps", "frame_rate", "avg_fps"):
        v = meta.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
            if fv > 0:
                return fv
        except Exception:
            continue
    return None


def _sample_indices_last_seconds_or_even(
    t: int,
    fps: float | None,
    seconds: float,
    fps_sample: float,
) -> list[int]:
    seconds = float(seconds)
    fps_sample = float(fps_sample)
    if seconds <= 0 or fps_sample <= 0:
        raise ValueError("seconds and fps_sample must be > 0")
    k = int(round(seconds * fps_sample))
    k = max(1, k)
    end = t - 1
    if t <= 0:
        return []

    if fps is None:
        if k == 1:
            return [end]
        idxs = np.linspace(0, end, k).round().astype(int).tolist()
        return [min(max(i, 0), end) for i in idxs]

    duration_s = (t - 1) / fps if fps > 0 else 0.0
    if duration_s + 1e-6 < seconds:
        # shorter than requested => evenly spaced across whole clip
        if k == 1:
            return [end]
        idxs = np.linspace(0, end, k).round().astype(int).tolist()
        return [min(max(i, 0), end) for i in idxs]

    idxs: list[int] = []
    for j in range(k):
        offset_s = (k - 1 - j) / fps_sample
        idx = int(round(end - offset_s * fps))
        idxs.append(min(max(idx, 0), end))
    return idxs


def _frames_to_clip_uint8_bcthw(frames_thwc_uint8: np.ndarray, idxs: list[int], resolution_hw: tuple[int, int]) -> torch.Tensor:
    sel = frames_thwc_uint8[idxs]  # (K,H,W,C) uint8
    tchw = torch.from_numpy(sel).permute(0, 3, 1, 2).contiguous()  # (K,C,H,W) uint8
    tchw = resize_input(tchw, [resolution_hw[0], resolution_hw[1]])
    return tchw.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1,C,K,H,W)


def _pool(latent_bcthw: torch.Tensor, mode: str) -> torch.Tensor:
    # latent: (B,C,T,H,W)
    if mode == "mean_thw":
        return latent_bcthw.mean(dim=(2, 3, 4))
    if mode == "mean_hw_keep_t":
        x = latent_bcthw.mean(dim=(3, 4))  # (B,C,T)
        return x.reshape(x.shape[0], -1)
    if mode == "max_hw_keep_t":
        x = latent_bcthw.amax(dim=(3, 4))  # (B,C,T)
        return x.reshape(x.shape[0], -1)
    raise ValueError(f"Unknown pool mode: {mode}")


def _default_experiment(model_size: str) -> str:
    return DEFAULT_EXP_14B if model_size == "14B" else DEFAULT_EXP_2B


def _default_ckpt(model_size: str) -> str:
    return HF_14B_PRETRAINED if model_size == "14B" else HF_2B_POSTTRAINED


@dataclass(frozen=True)
class PredictConfig:
    # embedding
    seconds: float = 5.0
    fps_sample: float = 1.0
    pool: str = "mean_hw_keep_t"
    resolution: str = "360,640"
    device: str = "cuda"

    # tokenizer loading
    model_size: str = "2B"
    experiment: str | None = None
    ckpt_path: str | None = None
    config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
    experiment_opt: tuple[str, ...] = ()


class VideoRiskPredictor:
    """
    Loads:
    - tokenizer (Cosmos) for embeddings
    - XGBoost model for classification
    """

    def __init__(self, xgb_model_path: str, cfg: PredictConfig):
        self.cfg = cfg
        self.resolution_hw = _parse_hw(cfg.resolution)

        # load xgb model
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as e:
            raise RuntimeError("xgboost is not installed (pip install -U xgboost)") from e

        model_path = Path(xgb_model_path)
        if not model_path.exists():
            raise FileNotFoundError(str(model_path))

        self.xgb = XGBClassifier()
        self.xgb.load_model(str(model_path))

        # load expected feature dim if meta exists (optional)
        self.expected_dim: int | None = None
        meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("feature_dim") is not None:
                    self.expected_dim = int(meta["feature_dim"])
            except Exception:
                self.expected_dim = None

        # load tokenizer once
        exp = cfg.experiment or _default_experiment(cfg.model_size)
        ckpt = cfg.ckpt_path or _default_ckpt(cfg.model_size)
        self.tokenizer, self.model_config, _ = load_tokenizer_only(
            experiment=exp,
            ckpt_path=ckpt,
            config_file=cfg.config_file,
            experiment_opts=list(cfg.experiment_opt),
            device=cfg.device,
        )

    def embed_video_path(self, video_path: str) -> np.ndarray:
        frames_np, meta = easy_io.load(str(video_path))  # (T,H,W,C) uint8
        t = int(frames_np.shape[0])
        if t < 1:
            raise ValueError("Decoded video has zero frames")
        fps = _get_fps(meta) if isinstance(meta, dict) else None

        idxs = _sample_indices_last_seconds_or_even(
            t=t,
            fps=fps,
            seconds=self.cfg.seconds,
            fps_sample=self.cfg.fps_sample,
        )
        clip = _frames_to_clip_uint8_bcthw(frames_np, idxs, self.resolution_hw)  # (1,C,K,H,W) uint8
        x = _to_bf16_minus1_1(clip, device=self.cfg.device)
        with torch.inference_mode():
            latent = self.tokenizer.encode(x)
            feat = _pool(latent, mode=self.cfg.pool)  # (1,D)
        feat_np = feat.float().cpu().numpy()[0]

        if self.expected_dim is not None and feat_np.shape[0] != self.expected_dim:
            raise ValueError(f"Feature dim mismatch: expected {self.expected_dim}, got {feat_np.shape[0]}")
        return feat_np

    def embed_video_bytes(self, video_bytes: bytes, suffix: str = ".mp4") -> np.ndarray:
        # easy_io expects a path, so write to a temp file.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(video_bytes)
            f.flush()
            return self.embed_video_path(f.name)

    def predict_from_embedding(self, feat: np.ndarray) -> dict[str, Any]:
        feat = np.asarray(feat, dtype=np.float32).reshape(1, -1)
        p_pos = float(self.xgb.predict_proba(feat)[:, 1][0])

        # UI mapping (simple heuristic):
        # - "confidence" = probability
        # - "impact_in" is not directly modeled; we map it into [0.5, seconds] with a small risk-dependent shift.
        impact_in = float(np.clip(self.cfg.seconds - p_pos * 1.0, 0.5, self.cfg.seconds))

        if p_pos >= 0.85:
            risk = "CRITICAL RISK"
            status = "INTERSECTION IMMINENT"
        elif p_pos >= 0.60:
            risk = "ELEVATED RISK"
            status = "INTERSECTION LIKELY"
        else:
            risk = "LOW RISK"
            status = "NO INTERSECTION DETECTED"

        return {
            "risk": risk,
            "status": status,
            "impact_in_s": round(impact_in, 2),
            "confidence": round(p_pos, 6),
            "confidence_percent": round(p_pos * 100.0, 3),
        }

    def predict_video_bytes(self, video_bytes: bytes, suffix: str = ".mp4") -> dict[str, Any]:
        feat = self.embed_video_bytes(video_bytes, suffix=suffix)
        out = self.predict_from_embedding(feat)
        out["embedding_dim"] = int(feat.shape[0])
        return out


def load_from_env() -> VideoRiskPredictor:
    # Required
    model_path = os.environ.get("XGB_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("Set XGB_MODEL_PATH to your saved XGBoost model (e.g. xgb_nexar.json)")

    cfg = PredictConfig(
        seconds=float(os.environ.get("PRED_SECONDS", "5.0")),
        fps_sample=float(os.environ.get("PRED_FPS_SAMPLE", "1.0")),
        pool=os.environ.get("PRED_POOL", "mean_hw_keep_t"),
        resolution=os.environ.get("PRED_RESOLUTION", "360,640"),
        device=os.environ.get("PRED_DEVICE", "cuda"),
        model_size=os.environ.get("COSMOS_MODEL_SIZE", "2B"),
        experiment=os.environ.get("COSMOS_EXPERIMENT") or None,
        ckpt_path=os.environ.get("COSMOS_CKPT") or None,
        config_file=os.environ.get("COSMOS_CONFIG_FILE", "cosmos_predict2/_src/predict2/configs/video2world/config.py"),
        experiment_opt=tuple(x for x in os.environ.get("COSMOS_EXPERIMENT_OPTS", "").split("||") if x),
    )
    return VideoRiskPredictor(xgb_model_path=model_path, cfg=cfg)

