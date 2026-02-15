#!/usr/bin/env python3
"""
Generate classifier embeddings for the Nexar collision dataset.

Dataset layout (expected):
  <root>/
    positive/*.mp4
    negative/*.mp4

For each video:
- Decode with `easy_io.load()`
- Take the middle frame (t // 2)
- Encode a *single-frame* clip through the tokenizer (no diffusion)
- Mean-pool latent -> fixed vector

Outputs a .pt file:
  {
    "X": FloatTensor [N, C],     # pooled embeddings
    "y": LongTensor [N],         # labels: 1=positive, 0=negative
    "paths": list[str],          # video paths
    "split": list[str],          # "test-private" or "test-public"
    "meta": dict                # settings, counts, etc.
  }

Example:
  cd /home/asus/new/cosmos-predict2.5
  python examples/nexar_midframe_embeddings.py \
    --roots "/home/asus/dataset/nexar_collision_prediction/test-private" "/home/asus/dataset/nexar_collision_prediction/test-public" \
    --output "nexar_midframe_feats.pt" \
    --batch_size 16 \
    --latent_frames 1 \
    --resolution 360,640 \
    --benchmark
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.predict2.inference.video2world import resize_input

# Reuse model-loading + normalization/pooling logic from the fast script.
from new_embedding_generation import (  # noqa: E402
    DEFAULT_EXP_14B,
    DEFAULT_EXP_2B,
    HF_14B_PRETRAINED,
    HF_2B_POSTTRAINED,
    _parse_hw,
    _pool_mean_bcthw,
    _to_bf16_minus1_1,
    load_tokenizer_only,
)


VIDEO_EXTS = {".mp4"}


@dataclass(frozen=True)
class Item:
    path: Path
    y: int  # 1=positive, 0=negative
    split: str  # "test-private" | "test-public" | other root name


def _infer_label(p: Path) -> int | None:
    # Label from directory name.
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


def _middle_frame_uint8_chw(video_path: Path) -> torch.Tensor:
    frames_np, _ = easy_io.load(str(video_path))  # (T,H,W,C) uint8
    t = int(frames_np.shape[0])
    if t < 1:
        raise ValueError(f"No frames decoded: {video_path}")
    mid = t // 2
    frame = frames_np[mid]  # (H,W,C) uint8
    # to CHW uint8
    return torch.from_numpy(frame).permute(2, 0, 1).contiguous()


def _sample_frames_uint8_chw(video_path: Path, num_samples: int) -> list[torch.Tensor]:
    frames_np, _ = easy_io.load(str(video_path))  # (T,H,W,C) uint8
    t = int(frames_np.shape[0])
    if t < 1:
        raise ValueError(f"No frames decoded: {video_path}")
    num_samples = max(1, int(num_samples))
    if num_samples == 1:
        idxs = [t // 2]
    else:
        # evenly spaced indices across the full clip
        idxs = np.linspace(0, t - 1, num_samples).round().astype(int).tolist()
        # ensure within bounds
        idxs = [min(max(i, 0), t - 1) for i in idxs]
    out: list[torch.Tensor] = []
    for i in idxs:
        frame = frames_np[i]  # (H,W,C) uint8
        out.append(torch.from_numpy(frame).permute(2, 0, 1).contiguous())
    return out


def _frame_to_clip_uint8_bcthw(frame_chw_uint8: torch.Tensor, resolution_hw: tuple[int, int], pixel_frames: int):
    """
    frame_chw_uint8: (C,H,W) uint8
    returns: (1,C,T,H,W) uint8 with first frame = image, rest = zeros.
    """
    if frame_chw_uint8.dtype != torch.uint8:
        raise ValueError("Expected uint8 frame")
    pixel_frames = max(1, int(pixel_frames))

    tchw = frame_chw_uint8.unsqueeze(0)  # (1,C,H,W)
    tchw = resize_input(tchw, [resolution_hw[0], resolution_hw[1]])  # (1,C,H,W)

    if pixel_frames > 1:
        pad = torch.zeros(pixel_frames - 1, tchw.shape[1], tchw.shape[2], tchw.shape[3], dtype=torch.uint8)
        tchw = torch.cat([tchw, pad], dim=0)  # (T,C,H,W)

    return tchw.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1,C,T,H,W)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[
            "/home/asus/dataset/nexar_collision_prediction/test-private",
            "/home/asus/dataset/nexar_collision_prediction/test-public",
        ],
        help="One or more dataset roots to scan.",
    )
    ap.add_argument("--output", required=True, help="Output .pt file.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--resolution", default="360,640", help="Resize frame to H,W before encoding.")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument(
        "--frames_per_video",
        type=int,
        default=1,
        help="How many evenly-spaced frames to sample per video and average embeddings over. 1 = middle frame.",
    )

    ap.add_argument("--model_size", default="2B", choices=("2B", "14B"))
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument(
        "--config_file",
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
    )
    ap.add_argument("--experiment_opt", action="append", default=[])

    ap.add_argument(
        "--latent_frames",
        type=int,
        default=1,
        help="Target latent frames. Pixel frames = (latent_frames-1)*4+1. Use 1 for max speed.",
    )
    ap.add_argument(
        "--pixel_frames",
        type=int,
        default=None,
        help="Override pixel-frame count directly (T). If set, --latent_frames is ignored.",
    )

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
        raise ValueError("No labeled videos found under provided roots (expected positive/negative subdirs).")

    n_pos = sum(1 for it in items if it.y == 1)
    n_neg = len(items) - n_pos
    log.info(f"Found {len(items)} videos: pos={n_pos}, neg={n_neg}")
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

    if args.pixel_frames is not None:
        pixel_frames = int(args.pixel_frames)
        latent_frames = tokenizer.get_latent_num_frames(pixel_frames)
    else:
        latent_frames = int(args.latent_frames)
        pixel_frames = tokenizer.get_pixel_num_frames(latent_frames)
    log.info(f"Encoding with pixel_frames={pixel_frames} (~latent_frames={latent_frames})")

    Xs: list[torch.Tensor] = []
    ys: list[int] = []
    paths: list[str] = []
    splits: list[str] = []
    errors: list[dict] = []

    encode_total = 0.0
    n_done = 0

    # Simple batching over items
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        tb = time.time()

        clips: list[torch.Tensor] = []
        clip_item_idx: list[int] = []
        batch_items: list[Item] = []

        # Decode + preprocess on CPU
        for it in batch:
            try:
                frames = _sample_frames_uint8_chw(it.path, args.frames_per_video)
                item_local_idx = len(batch_items)
                batch_items.append(it)
                for fr in frames:
                    clip = _frame_to_clip_uint8_bcthw(fr, resolution_hw, pixel_frames)
                    clips.append(clip)
                    clip_item_idx.append(item_local_idx)
            except Exception as e:
                errors.append({"path": str(it.path), "error": repr(e)})

        if not clips:
            continue

        x_uint8 = torch.cat(clips, dim=0)  # (B,C,T,H,W) uint8
        x = _to_bf16_minus1_1(x_uint8, device=args.device)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t_enc0 = time.time()

        with torch.inference_mode():
            latent = tokenizer.encode(x)
            pooled = _pool_mean_bcthw(latent)  # (B,C)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t_enc = time.time() - t_enc0

        # Aggregate per-video by averaging sampled frame embeddings.
        pooled_cpu = pooled.float().cpu()  # (n_clips,C)
        C = pooled_cpu.shape[1]
        n_vid = len(batch_items)
        sums = torch.zeros((n_vid, C), dtype=torch.float32)
        counts = torch.zeros((n_vid,), dtype=torch.int32)
        for j, idx in enumerate(clip_item_idx):
            sums[idx] += pooled_cpu[j]
            counts[idx] += 1
        counts_f = counts.clamp_min(1).to(torch.float32).unsqueeze(1)
        per_vid = sums / counts_f  # (n_vid,C)

        encode_total += t_enc
        n_done += per_vid.shape[0]

        Xs.append(per_vid)
        ys.extend([it.y for it in batch_items])
        paths.extend([str(it.path) for it in batch_items])
        splits.extend([it.split for it in batch_items])

        if args.benchmark:
            log.info(
                f"Batch {i//args.batch_size + 1}: vids={per_vid.shape[0]} clips={pooled.shape[0]} encode={t_enc:.2f}s total={time.time()-tb:.2f}s"
            )

    if not Xs:
        raise RuntimeError("No embeddings were generated (all items failed to decode?)")

    X = torch.cat(Xs, dim=0)  # (N,C) float32
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
            "pixel_frames": pixel_frames,
            "latent_frames": latent_frames,
            "frames_per_video": int(args.frames_per_video),
            "tokenizer_state_t": getattr(model_config, "state_t", None),
            "latent_channels": int(X.shape[1]),
            "n_total": len(items),
            "n_done": int(X.shape[0]),
            "n_errors": len(errors),
            "class_counts": {"pos": int((y == 1).sum()), "neg": int((y == 0).sum())},
            "errors": errors[:50],  # keep file size reasonable; first 50 errors
        },
    }

    torch.save(payload, outp)
    log.info(f"Saved embeddings: {outp}")
    log.info(f"X={tuple(X.shape)} y={tuple(y.shape)} done={n_done} errors={len(errors)}")
    if args.benchmark:
        log.info(f"Encode total: {encode_total:.2f}s for {n_done} -> {n_done/encode_total:.3f} items/s")


if __name__ == "__main__":
    main()

