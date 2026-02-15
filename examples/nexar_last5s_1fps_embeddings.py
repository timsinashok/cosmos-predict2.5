#!/usr/bin/env python3
"""
Temporal embeddings for Nexar dataset using a short video snippet.

Idea:
- Instead of a single middle frame, sample 1 frame/second over the last N seconds
  (default N=5 => 5 frames), build a small clip (T=K), and encode it with the tokenizer.
- This adds temporal information (motion / closing speed cues) while staying fast.

Dataset layout (expected):
  <root>/
    positive/*.mp4
    negative/*.mp4

Outputs a .pt:
  {
    "X": FloatTensor [N, D],   # pooled embeddings
    "y": LongTensor [N],       # 1=positive, 0=negative
    "paths": list[str],
    "split": list[str],
    "meta": dict
  }

Example:
  cd /home/asus/new/cosmos-predict2.5
  python examples/nexar_last5s_1fps_embeddings.py \
    --roots "/home/asus/dataset/nexar_collision_prediction/test-private" "/home/asus/dataset/nexar_collision_prediction/test-public" \
    --output "nexar_last5s_1fps.pt" \
    --seconds 5 --fps_sample 1 \
    --resolution 360,640 \
    --batch_size 8 \
    --pool mean_hw_keep_t \
    --benchmark
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.video2world import resize_input

# Reuse model-loading + normalization from the fast script.
from new_embedding_generation import (  # noqa: E402
    DEFAULT_EXP_14B,
    DEFAULT_EXP_2B,
    HF_14B_PRETRAINED,
    HF_2B_POSTTRAINED,
    _parse_hw,
    _to_bf16_minus1_1,
    load_tokenizer_only,
)

VIDEO_EXTS = {".mp4"}


@dataclass(frozen=True)
class Item:
    path: Path
    y: int  # 1=positive, 0=negative
    split: str


def _infer_label(p: Path) -> int | None:
    parts = {x.lower() for x in p.parts}
    if "positive" in parts:
        return 1
    if "negative" in parts:
        return 0
    return None


def _collect_videos(root: Path) -> list[Item]:
    if not root.exists():
        raise FileNotFoundError(str(root))
    split = root.name
    items: list[Item] = []
    for ext in VIDEO_EXTS:
        for p in sorted(root.rglob(f"*{ext}")):
            y = _infer_label(p)
            if y is None:
                continue
            items.append(Item(path=p, y=y, split=split))
    return items


def _get_fps(meta: dict) -> float | None:
    # Best-effort; depends on easy_io backend.
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


def _sample_last_seconds_indices(t: int, fps: float | None, seconds: float, fps_sample: float) -> list[int]:
    """
    Returns K indices (length K = seconds * fps_sample), sampled at 1/fps_sample sec intervals
    counting backwards from the end of the clip. If fps is unknown, falls back to evenly spaced.
    """
    seconds = float(seconds)
    fps_sample = float(fps_sample)
    if seconds <= 0 or fps_sample <= 0:
        raise ValueError("seconds and fps_sample must be > 0")
    k = int(round(seconds * fps_sample))
    k = max(1, k)

    end = t - 1
    if fps is None:
        # fallback: evenly spaced over the whole clip
        if k == 1:
            return [end]
        idxs = np.linspace(0, end, k).round().astype(int).tolist()
        return [min(max(i, 0), end) for i in idxs]

    # offsets in seconds: (k-1)/fps_sample ... 0
    idxs: list[int] = []
    for j in range(k):
        offset_s = (k - 1 - j) / fps_sample
        idx = int(round(end - offset_s * fps))
        idxs.append(min(max(idx, 0), end))
    return idxs


def _frames_to_clip_uint8_bcthw(frames_thwc_uint8: np.ndarray, idxs: list[int], resolution_hw: tuple[int, int]) -> torch.Tensor:
    """
    frames_thwc_uint8: (T,H,W,C) uint8
    returns: (1,C,K,H,W) uint8
    """
    # gather -> (K,H,W,C)
    sel = frames_thwc_uint8[idxs]
    tchw = torch.from_numpy(sel).permute(0, 3, 1, 2).contiguous()  # (K,C,H,W) uint8
    tchw = resize_input(tchw, [resolution_hw[0], resolution_hw[1]])
    return tchw.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1,C,K,H,W)


def _pool(latent_bcthw: torch.Tensor, mode: str) -> torch.Tensor:
    """
    latent: (B,C,T,H,W)
    - mean_thw: (B,C)
    - mean_hw_keep_t: (B,C*T)
    - max_hw_keep_t: (B,C*T)
    """
    if mode == "mean_thw":
        return latent_bcthw.mean(dim=(2, 3, 4))
    if mode == "mean_hw_keep_t":
        x = latent_bcthw.mean(dim=(3, 4))  # (B,C,T)
        return x.reshape(x.shape[0], -1)
    if mode == "max_hw_keep_t":
        x = latent_bcthw.amax(dim=(3, 4))  # (B,C,T)
        return x.reshape(x.shape[0], -1)
    raise ValueError(f"Unknown pool mode: {mode}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[
            "/home/asus/dataset/nexar_collision_prediction/test-private",
            "/home/asus/dataset/nexar_collision_prediction/test-public",
        ],
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--resolution", default="360,640")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--benchmark", action="store_true")

    ap.add_argument("--seconds", type=float, default=5.0, help="How many seconds from the end to sample.")
    ap.add_argument("--fps_sample", type=float, default=1.0, help="Frames per second to sample (1.0 => 1 frame/sec).")
    ap.add_argument(
        "--pool",
        default="mean_hw_keep_t",
        choices=("mean_thw", "mean_hw_keep_t", "max_hw_keep_t"),
        help="How to pool the latent. Keeping T often helps with temporal signal.",
    )

    ap.add_argument("--model_size", default="2B", choices=("2B", "14B"))
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--config_file", default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    ap.add_argument("--experiment_opt", action="append", default=[])
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    resolution_hw = _parse_hw(args.resolution)

    if args.experiment is None:
        args.experiment = DEFAULT_EXP_14B if args.model_size == "14B" else DEFAULT_EXP_2B
    if args.ckpt_path is None:
        args.ckpt_path = HF_14B_PRETRAINED if args.model_size == "14B" else HF_2B_POSTTRAINED

    items: list[Item] = []
    for r in roots:
        items.extend(_collect_videos(r))
    if not items:
        raise ValueError("No labeled videos found (expected positive/negative subdirs).")

    log.info(f"Found {len(items)} videos")
    log.info(f"Sampling: last {args.seconds:g}s @ {args.fps_sample:g} fps | pool={args.pool}")
    log.info(f"Resolution={resolution_hw} device={args.device} batch_size={args.batch_size}")

    t0 = time.time()
    tokenizer, model_config, _cfg = load_tokenizer_only(
        experiment=args.experiment,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        experiment_opts=args.experiment_opt,
        device=args.device,
    )
    log.info(f"Loaded tokenizer in {time.time() - t0:.2f}s")

    Xs: list[torch.Tensor] = []
    ys: list[int] = []
    paths: list[str] = []
    splits: list[str] = []
    errors: list[dict] = []

    encode_total = 0.0
    n_done = 0

    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        tb = time.time()

        clips: list[torch.Tensor] = []
        batch_items: list[Item] = []
        batch_meta: list[dict] = []

        # Decode on CPU
        for it in batch:
            try:
                frames_np, meta = easy_io.load(str(it.path))  # (T,H,W,C) uint8
                t = int(frames_np.shape[0])
                if t < 1:
                    raise ValueError("decoded T=0")
                fps = _get_fps(meta)
                idxs = _sample_last_seconds_indices(t=t, fps=fps, seconds=args.seconds, fps_sample=args.fps_sample)
                clip = _frames_to_clip_uint8_bcthw(frames_np, idxs, resolution_hw)
                clips.append(clip)
                batch_items.append(it)
                batch_meta.append({"fps": fps, "t": t, "idxs": idxs})
            except Exception as e:
                errors.append({"path": str(it.path), "error": repr(e)})

        if not clips:
            continue

        x_uint8 = torch.cat(clips, dim=0)  # (B,C,K,H,W) uint8
        x = _to_bf16_minus1_1(x_uint8, device=args.device)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t_enc0 = time.time()

        with torch.inference_mode():
            latent = tokenizer.encode(x)  # (B,C,T',H',W')
            feat = _pool(latent, mode=args.pool)  # (B,D)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t_enc = time.time() - t_enc0

        encode_total += t_enc
        n_done += feat.shape[0]

        Xs.append(feat.float().cpu())
        ys.extend([it.y for it in batch_items])
        paths.extend([str(it.path) for it in batch_items])
        splits.extend([it.split for it in batch_items])

        if args.benchmark:
            log.info(
                f"Batch {i//args.batch_size + 1}: vids={feat.shape[0]} encode={t_enc:.2f}s total={time.time()-tb:.2f}s"
            )

    if not Xs:
        raise RuntimeError("No embeddings were generated (all items failed?)")

    X = torch.cat(Xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)

    payload = {
        "X": X,
        "y": y,
        "paths": paths,
        "split": splits,
        "meta": {
            "roots": [str(r) for r in roots],
            "resolution_hw": resolution_hw,
            "device": args.device,
            "batch_size": args.batch_size,
            "experiment": args.experiment,
            "ckpt_path": args.ckpt_path,
            "seconds": float(args.seconds),
            "fps_sample": float(args.fps_sample),
            "pool": args.pool,
            "tokenizer_state_t": getattr(model_config, "state_t", None),
            "latent_channels_or_dim": int(X.shape[1]),
            "n_total": len(items),
            "n_done": int(X.shape[0]),
            "n_errors": len(errors),
            "class_counts": {"pos": int((y == 1).sum()), "neg": int((y == 0).sum())},
            "errors": errors[:50],
        },
    }

    torch.save(payload, outp)
    log.info(f"Saved embeddings: {outp}")
    log.info(f"X={tuple(X.shape)} y={tuple(y.shape)} done={n_done} errors={len(errors)}")
    if args.benchmark:
        log.info(f"Encode total: {encode_total:.2f}s for {n_done} -> {n_done/encode_total:.3f} vids/s")


if __name__ == "__main__":
    main()

