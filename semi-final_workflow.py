#!/usr/bin/env python3
"""
Semi-final workflow: raw video -> embedding -> XGBoost prediction (+optional training).

Two modes:

1) Train and save XGB model from a saved embedding .pt:
   python semi-final_workflow.py train \
     --data nexar_last5s_1fps.pt --use_split test-private --test_size 0.1 --seed 0 \
     --model_out xgb_nexar.json

2) Predict on a single video:
   python semi-final_workflow.py predict \
     --video /path/to/video.mp4 \
     --model xgb_nexar.json

Embedding strategy (predict):
- Sample K frames at `fps_sample` over the last `seconds` of the video.
- If the video is shorter than `seconds` (and FPS is known), fall back to K evenly-spaced frames.
- Encode clip with tokenizer (no diffusion), then pool to a feature vector.

The model expects the same feature dimension D at predict time as at train time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.video2world import resize_input

from new_embedding_generation import (  # noqa: E402
    DEFAULT_EXP_14B,
    DEFAULT_EXP_2B,
    HF_14B_PRETRAINED,
    HF_2B_POSTTRAINED,
    _parse_hw,
    _to_bf16_minus1_1,
    load_tokenizer_only,
)


def _require_sklearn():
    try:
        from sklearn.model_selection import train_test_split  # noqa: F401
        from sklearn.metrics import accuracy_score  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name not in ("sklearn", "scikit_learn"):
            raise
        py = sys.executable
        raise SystemExit(
            "scikit-learn is not installed in the Python you're using.\n\n"
            f"You're running: {py}\n\n"
            "Install into THIS interpreter, e.g.:\n"
            f"  {py} -m pip install -U scikit-learn\n"
        )


def _require_xgboost():
    try:
        from xgboost import XGBClassifier  # noqa: F401
    except ModuleNotFoundError:
        py = sys.executable
        raise SystemExit(
            "xgboost is not installed in the Python you're using.\n\n"
            f"You're running: {py}\n\n"
            "Install into THIS interpreter, e.g.:\n"
            f"  {py} -m pip install -U xgboost\n"
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
        # no FPS => evenly spaced across the whole clip
        if k == 1:
            return [end]
        idxs = np.linspace(0, end, k).round().astype(int).tolist()
        return [min(max(i, 0), end) for i in idxs]

    duration_s = (t - 1) / fps if fps > 0 else 0.0
    if duration_s + 1e-6 < seconds:
        # shorter than requested window => evenly spaced across the whole clip
        if k == 1:
            return [end]
        idxs = np.linspace(0, end, k).round().astype(int).tolist()
        return [min(max(i, 0), end) for i in idxs]

    # Sample backwards from end at 1/fps_sample intervals.
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


def _meta_path_for_model(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".meta.json")


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def train_and_save(args: argparse.Namespace) -> None:
    _require_sklearn()
    _require_xgboost()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from xgboost import XGBClassifier

    d = torch.load(args.data, map_location="cpu")
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().astype(np.int64)

    if args.use_split != "all":
        if "split" not in d:
            raise ValueError("--use_split requested but data has no 'split' field.")
        s = np.array(d["split"])
        m = s == args.use_split
        if m.sum() == 0:
            raise ValueError(f"No samples for split={args.use_split}")
        X = X[m]
        y = y[m]

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )

    fit_kwargs = {}
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        fit_kwargs = {"eval_set": [(Xte, yte)], "verbose": False, "early_stopping_rounds": args.early_stopping_rounds}

    try:
        clf.fit(Xtr, ytr, **fit_kwargs)
    except TypeError:
        clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)

    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, prob) if len(np.unique(yte)) > 1 else float("nan")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(model_path))

    meta = {
        "trained_from": str(args.data),
        "use_split": args.use_split,
        "test_size": args.test_size,
        "seed": args.seed,
        "feature_dim": int(X.shape[1]),
        "class_counts": {
            "train_pos": int((ytr == 1).sum()),
            "train_neg": int((ytr == 0).sum()),
            "test_pos": int((yte == 1).sum()),
            "test_neg": int((yte == 0).sum()),
        },
        "metrics": {"accuracy": float(acc), "roc_auc": float(auc)},
        "xgb_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "min_child_weight": args.min_child_weight,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
    }
    _save_json(_meta_path_for_model(model_path), meta)

    print(f"Saved model: {model_path}")
    print(f"Saved meta:  {_meta_path_for_model(model_path)}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC:  {auc:.4f}")
    print()
    print(classification_report(yte, pred, digits=4))


def embed_video(args: argparse.Namespace) -> np.ndarray:
    resolution_hw = _parse_hw(args.resolution)

    frames_np, meta = easy_io.load(str(args.video))  # (T,H,W,C) uint8
    t = int(frames_np.shape[0])
    if t < 1:
        raise ValueError(f"Decoded T=0 for {args.video}")
    fps = _get_fps(meta) if isinstance(meta, dict) else None

    idxs = _sample_indices_last_seconds_or_even(
        t=t,
        fps=fps,
        seconds=args.seconds,
        fps_sample=args.fps_sample,
    )
    clip = _frames_to_clip_uint8_bcthw(frames_np, idxs, resolution_hw)  # (1,C,K,H,W) uint8

    if args.experiment is None:
        args.experiment = _default_experiment(args.model_size)
    if args.ckpt_path is None:
        args.ckpt_path = _default_ckpt(args.model_size)

    tokenizer, _model_config, _cfg = load_tokenizer_only(
        experiment=args.experiment,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        experiment_opts=args.experiment_opt,
        device=args.device,
    )

    x = _to_bf16_minus1_1(clip, device=args.device)
    with torch.inference_mode():
        latent = tokenizer.encode(x)
        feat = _pool(latent, mode=args.pool)  # (1,D) torch
    return feat.float().cpu().numpy()[0]


def predict(args: argparse.Namespace) -> None:
    _require_xgboost()
    from xgboost import XGBClassifier

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    # If meta exists and user didn't override, use it to sanity check feature dim.
    meta_path = _meta_path_for_model(model_path)
    trained_dim = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            trained_dim = int(meta.get("feature_dim")) if meta.get("feature_dim") is not None else None
        except Exception:
            trained_dim = None

    feat = embed_video(args)  # (D,)
    if trained_dim is not None and int(feat.shape[0]) != trained_dim:
        raise ValueError(f"Feature dim mismatch: model expects D={trained_dim}, got D={feat.shape[0]}")

    clf = XGBClassifier()
    clf.load_model(str(model_path))

    prob_pos = float(clf.predict_proba(feat.reshape(1, -1))[:, 1][0])
    pred = 1 if prob_pos >= args.threshold else 0

    # Print a compact result
    print(f"video: {args.video}")
    print(f"pred: {pred} (1=positive, 0=negative)")
    print(f"p_positive: {prob_pos:.4f}")
    print(f"p_negative: {1.0 - prob_pos:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train XGB on a saved embedding .pt and save model.")
    pt.add_argument("--data", required=True, help="Embedding .pt with X/y.")
    pt.add_argument("--use_split", default="all")
    pt.add_argument("--test_size", type=float, default=0.1)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--model_out", required=True, help="Output model path, e.g. xgb_model.json")

    pt.add_argument("--n_estimators", type=int, default=500)
    pt.add_argument("--max_depth", type=int, default=3)
    pt.add_argument("--learning_rate", type=float, default=0.05)
    pt.add_argument("--subsample", type=float, default=0.8)
    pt.add_argument("--colsample_bytree", type=float, default=0.8)
    pt.add_argument("--reg_lambda", type=float, default=1.0)
    pt.add_argument("--reg_alpha", type=float, default=0.0)
    pt.add_argument("--min_child_weight", type=float, default=1.0)
    pt.add_argument("--early_stopping_rounds", type=int, default=30)

    # predict
    pp = sub.add_parser("predict", help="Predict on a single video using a saved XGB model.")
    pp.add_argument("--video", required=True, help="Path to .mp4 video.")
    pp.add_argument("--model", required=True, help="Path to saved XGB model (json).")
    pp.add_argument("--threshold", type=float, default=0.5)

    # embedding params
    pp.add_argument("--seconds", type=float, default=5.0)
    pp.add_argument("--fps_sample", type=float, default=1.0)
    pp.add_argument("--pool", default="mean_hw_keep_t", choices=("mean_thw", "mean_hw_keep_t", "max_hw_keep_t"))
    pp.add_argument("--resolution", default="360,640")
    pp.add_argument("--device", default="cuda", choices=("cuda", "cpu"))

    # tokenizer loading params
    pp.add_argument("--model_size", default="2B", choices=("2B", "14B"))
    pp.add_argument("--experiment", default=None)
    pp.add_argument("--ckpt_path", default=None)
    pp.add_argument("--config_file", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    pp.add_argument("--experiment_opt", action="append", default=[])

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        train_and_save(args)
    elif args.cmd == "predict":
        predict(args)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

