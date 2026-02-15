#!/usr/bin/env python3
"""
Fast embedding generation for classifier training.

Goal: image/video -> *pooled* latent feature vector as fast as possible.

Key speed choices (defaults):
- Use only the tokenizer encoder via `tokenizer.encode()` (no diffusion sampling).
- Use T=1 frame for images (`--num_frames 1`) to minimize compute.
- Pool on GPU to a small vector and only then move to CPU.
- Batch multiple images together (`--batch_size`).

Outputs:
- Saves a `.pt` file with:
  - features: float32 tensor [N, C] (pooled mean over T,H,W)
  - paths: list[str]
  - meta: dict of settings used

Example:
  cd /home/asus/new/cosmos-predict2.5
  python new_embedding_generation.py \
    --input "/home/asus/new/cosmos-predict2.5/images/img2vid.jpg" \
    --output feats.pt \
    --batch_size 4 \
    --num_frames 1 \
    --benchmark
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import torch
import torchvision
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.inference.video2world import resize_input
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4"}

# Known-good HF URIs for public (non-INTERNAL) environments.
HF_2B_POSTTRAINED = (
    "hf://nvidia/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)
HF_14B_PRETRAINED = (
    "hf://nvidia/Cosmos-Predict2.5-14B/base/pre-trained/"
    "54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt"
)

DEFAULT_EXP_2B = "Stage-c_pt_4-Index-26-Size-2B-Res-720-Fps-16-Note-HQ_V6_from_22"
DEFAULT_EXP_14B = "Stage-c_pt_4-Index-43-Size-14B-Res-720-Fps-16-Note-T24_HQV5_from_40"


def _parse_hw(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"resolution must be 'H,W', got: {s}")
    return int(parts[0]), int(parts[1])


def _list_inputs(p: Path) -> list[Path]:
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(str(p))
    out: list[Path] = []
    for ext in IMAGE_EXTS:
        out.extend(sorted(p.rglob(f"*{ext}")))
    for ext in VIDEO_EXTS:
        out.extend(sorted(p.rglob(f"*{ext}")))
    return out


def _chunks(xs: list[Path], bs: int) -> Iterable[list[Path]]:
    for i in range(0, len(xs), bs):
        yield xs[i : i + bs]


def _preprocess_image_uint8_bcthw(
    image_path: Path,
    resolution_hw: tuple[int, int],
    pixel_frames: int,
) -> torch.Tensor:
    """
    Returns uint8 tensor shaped (1, C, T, H, W) in [0,255].
    """
    img = Image.open(image_path).convert("RGB")
    img = torchvision.transforms.functional.to_tensor(img)  # (C,H,W) float [0,1]
    img = (img * 255.0).to(torch.uint8)  # (C,H,W) uint8
    img_tchw = img.unsqueeze(0)  # (T=1,C,H,W)

    # resize_input expects (T,C,H,W) uint8
    img_tchw = resize_input(img_tchw, [resolution_hw[0], resolution_hw[1]])

    # Build clip: first frame = image, remaining = zeros (if any).
    pixel_frames = max(1, int(pixel_frames))
    if pixel_frames > 1:
        pad = torch.zeros(
            pixel_frames - 1,
            img_tchw.shape[1],
            img_tchw.shape[2],
            img_tchw.shape[3],
            dtype=torch.uint8,
        )
        clip_tchw = torch.cat([img_tchw, pad], dim=0)  # (pixel_frames,C,H,W)
    else:
        clip_tchw = img_tchw  # (1,C,H,W)

    # (T,C,H,W) -> (B,C,T,H,W)
    return clip_tchw.unsqueeze(0).permute(0, 2, 1, 3, 4)


def _preprocess_video_uint8_bcthw(
    video_path: Path,
    resolution_hw: tuple[int, int],
    pixel_frames: int,
) -> torch.Tensor:
    """
    Returns uint8 tensor shaped (1, C, T, H, W) in [0,255].
    Uses the last `pixel_frames` frames (or fewer if video shorter),
    then pads by repeating the last available frame to reach `pixel_frames`.
    """
    from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

    frames_np, _ = easy_io.load(str(video_path))  # (T,H,W,C) uint8
    t = int(frames_np.shape[0])
    if t < 1:
        raise ValueError(f"Video has no frames: {video_path}")

    pixel_frames = max(1, int(pixel_frames))
    use_frames = min(pixel_frames, t)

    # take last frames
    frames_np = frames_np[-use_frames:]
    vid_tchw = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()  # (T,C,H,W) uint8
    vid_tchw = resize_input(vid_tchw, [resolution_hw[0], resolution_hw[1]])

    if pixel_frames > use_frames:
        last = vid_tchw[-1:].repeat(pixel_frames - use_frames, 1, 1, 1)
        vid_tchw = torch.cat([vid_tchw, last], dim=0)  # (pixel_frames,C,H,W)
    elif pixel_frames < use_frames:
        vid_tchw = vid_tchw[-pixel_frames:]

    return vid_tchw.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1,C,T,H,W)


def _to_bf16_minus1_1(x_uint8_bcthw: torch.Tensor, device: str) -> torch.Tensor:
    x = x_uint8_bcthw.to(device=device, non_blocking=True)
    x = x.float().div_(255.0).mul_(2.0).sub_(1.0)  # [-1,1]
    return x.to(dtype=torch.bfloat16)


def _pool_mean_bcthw(latent: torch.Tensor) -> torch.Tensor:
    # latent: (B,C,T,H,W) -> (B,C)
    return latent.mean(dim=(2, 3, 4))


def load_tokenizer_only(
    experiment: str,
    ckpt_path: str,
    config_file: str,
    experiment_opts: list[str] | None,
    device: str,
):
    if experiment_opts is None:
        experiment_opts = []
    if not INTERNAL:
        experiment_opts = experiment_opts + ["~data_train"]

    model, cfg = load_model_from_checkpoint(
        experiment_name=experiment,
        s3_checkpoint_dir=ckpt_path,
        config_file=config_file,
        load_ema_to_reg=True,
        experiment_opts=experiment_opts,
        to_device=device,
    )

    tokenizer = model.tokenizer
    model_config = model.config

    # Put underlying tokenizer model in eval, if present.
    if hasattr(tokenizer, "model") and hasattr(tokenizer.model, "eval"):
        tokenizer.model.eval()

    # Free unneeded components to reduce memory.
    for attr in ("net", "text_encoder", "conditioner"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tokenizer, model_config, cfg


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="Path to an image/video file or a directory of images/videos.")
    ap.add_argument("--output", required=True, help="Output .pt file to write features to.")

    ap.add_argument("--resolution", default="360,640", help="Resize input to H,W before encoding.")
    ap.add_argument(
        "--latent_frames",
        type=int,
        default=1,
        help="Number of latent frames to target. Pixel frames become (latent_frames-1)*4+1. Use 1 for fastest.",
    )
    ap.add_argument(
        "--pixel_frames",
        type=int,
        default=None,
        help="Override pixel-frame count directly (T). If set, --latent_frames is ignored.",
    )
    ap.add_argument("--batch_size", type=int, default=4, help="Batch size for throughput.")

    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--benchmark", action="store_true", help="Print per-batch and overall timing.")

    ap.add_argument(
        "--model_size",
        default="2B",
        choices=("2B", "14B"),
        help="Used only for choosing defaults if you don't override experiment/ckpt.",
    )
    ap.add_argument("--experiment", default=None, help="Hydra experiment name. If omitted, uses a reasonable default.")
    ap.add_argument(
        "--ckpt_path",
        default=None,
        help="Checkpoint URI: local path, s3://..., or hf://... . If omitted, uses a known HF default.",
    )
    ap.add_argument(
        "--config_file",
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        help="Config file used by the loader.",
    )
    ap.add_argument(
        "--experiment_opt",
        action="append",
        default=[],
        help="Extra hydra overrides (repeatable). Example: --experiment_opt 'some.key=value'",
    )

    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    resolution_hw = _parse_hw(args.resolution)

    if args.experiment is None:
        args.experiment = DEFAULT_EXP_14B if args.model_size == "14B" else DEFAULT_EXP_2B

    if args.ckpt_path is None:
        args.ckpt_path = HF_14B_PRETRAINED if args.model_size == "14B" else HF_2B_POSTTRAINED

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    torch.set_grad_enabled(False)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    paths = _list_inputs(inp)
    if not paths:
        raise ValueError(f"No images found under: {inp}")

    log.info(f"Inputs: {len(paths)}")
    log.info(f"Resolution: {resolution_hw}, batch_size={args.batch_size}")
    log.info(f"Experiment: {args.experiment}")
    log.info(f"Checkpoint: {args.ckpt_path}")

    t0 = time.time()
    tokenizer, model_config, cfg = load_tokenizer_only(
        experiment=args.experiment,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        experiment_opts=args.experiment_opt,
        device=args.device,
    )
    load_s = time.time() - t0
    log.info(f"Loaded tokenizer in {load_s:.2f}s")

    if args.pixel_frames is not None:
        pixel_frames = int(args.pixel_frames)
        latent_frames = tokenizer.get_latent_num_frames(pixel_frames)
    else:
        latent_frames = int(args.latent_frames)
        pixel_frames = tokenizer.get_pixel_num_frames(latent_frames)
    if pixel_frames < 1 or latent_frames < 1:
        raise ValueError("pixel_frames and latent_frames must be >= 1")

    log.info(
        f"Encoding T={pixel_frames} pixel frames (~{latent_frames} latent frames). "
        f"(temporal_compression_factor={getattr(tokenizer, 'temporal_compression_factor', 'unknown')})"
    )

    all_feats: list[torch.Tensor] = []
    all_paths: list[str] = []

    encode_total = 0.0
    n_total = 0

    for batch in _chunks(paths, args.batch_size):
        tb = time.time()
        # Preprocess on CPU
        xs = []
        for p in batch:
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                xs.append(_preprocess_image_uint8_bcthw(p, resolution_hw, pixel_frames))
            elif ext in VIDEO_EXTS:
                xs.append(_preprocess_video_uint8_bcthw(p, resolution_hw, pixel_frames))
            else:
                raise ValueError(f"Unsupported input type: {p}")
        x_uint8 = torch.cat(xs, dim=0)  # (B,C,T,H,W) uint8

        # Move + normalize on GPU
        x = _to_bf16_minus1_1(x_uint8, device=args.device)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t_enc0 = time.time()

        with torch.inference_mode():
            latent = tokenizer.encode(x)  # (B,C,T',H',W') bf16/ fp?
            pooled = _pool_mean_bcthw(latent)  # (B,C)

        if args.device == "cuda":
            torch.cuda.synchronize()
        t_enc = time.time() - t_enc0
        encode_total += t_enc
        n_total += pooled.shape[0]

        # Move small pooled vector to CPU float32 for training
        pooled_cpu = pooled.float().cpu()
        all_feats.append(pooled_cpu)
        all_paths.extend([str(p) for p in batch])

        if args.benchmark:
            dt = time.time() - tb
            log.info(f"Batch {len(batch)}: encode={t_enc:.2f}s total={dt:.2f}s ({len(batch)/t_enc:.2f} img/s)")

    feats = torch.cat(all_feats, dim=0)  # (N,C) float32 on CPU

    payload = {
        "features": feats,
        "paths": all_paths,
        "meta": {
            "resolution_hw": resolution_hw,
            "pixel_frames": pixel_frames,
            "latent_frames": latent_frames,
            "batch_size": args.batch_size,
            "experiment": args.experiment,
            "ckpt_path": args.ckpt_path,
            "device": args.device,
            "latent_channels": feats.shape[1],
            "tokenizer_state_t": getattr(model_config, "state_t", None),
        },
    }

    torch.save(payload, outp)

    if args.benchmark:
        log.info(f"Saved: {outp}")
        log.info(f"Features: {tuple(feats.shape)} float32")
        log.info(f"Encode total: {encode_total:.2f}s for {n_total} -> {n_total/encode_total:.3f} img/s")


if __name__ == "__main__":
    main()

