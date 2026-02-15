#!/usr/bin/env python3
"""
2D PCA visualization of saved embeddings (positive vs negative).

Input format: .pt produced by `examples/nexar_midframe_embeddings.py`
  - X: FloatTensor [N, C]
  - y: LongTensor [N]  (1=positive, 0=negative)
  - split: list[str]   (optional)

Outputs:
  - PNG scatter plot (colored by label)
  - Optional NPZ with PCA coordinates

Example:
  cd /home/asus/new/cosmos-predict2.5
  python pca2d.py --data nexar_midframe_feats.pt --out_png pca_midframe.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA via SVD.
    Returns:
      z: [N,2] projected coords
      evr: [2] explained variance ratios
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # SVD: x = U S Vt
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:2]  # [2,C]
    z = x @ comps.T  # [N,2]
    # explained variance ratio
    n = x.shape[0]
    if n > 1:
        eigvals = (s**2) / (n - 1)
        total = float(eigvals.sum()) if eigvals.size else 0.0
        evr = (eigvals[:2] / total) if total > 0 else np.zeros((2,), dtype=np.float64)
    else:
        evr = np.zeros((2,), dtype=np.float64)
    return z, evr


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to .pt with X/y (from embedding generation).")
    ap.add_argument("--out_png", default="pca2d.png", help="Where to save the plot PNG.")
    ap.add_argument("--out_npz", default=None, help="Optional: save PCA coords/labels to .npz")
    ap.add_argument(
        "--use_split",
        default="all",
        help="Filter by split name (e.g. 'test-private', 'test-public') or 'all'.",
    )
    ap.add_argument("--max_points", type=int, default=0, help="If >0, randomly subsample to this many points.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--standardize",
        action="store_true",
        default=True,
        help="Standardize features (recommended).",
    )
    ap.add_argument("--no_standardize", action="store_false", dest="standardize")
    args = ap.parse_args()

    d = torch.load(args.data, map_location="cpu")
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().astype(np.int64)

    if args.use_split != "all":
        if "split" not in d:
            raise ValueError("--use_split requested but data has no 'split' field.")
        split = np.array(d["split"])
        m = split == args.use_split
        if m.sum() == 0:
            raise ValueError(f"No samples for split={args.use_split}")
        X = X[m]
        y = y[m]

    if args.max_points and args.max_points > 0 and X.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(X.shape[0], size=args.max_points, replace=False)
        X = X[idx]
        y = y[idx]

    if args.standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-6
        Xp = (X - mu) / sd
    else:
        Xp = X

    z, evr = _pca_2d(Xp)

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        py = sys.executable
        raise SystemExit(
            "matplotlib is not installed in the Python you're using.\n\n"
            f"You're running: {py}\n\n"
            "Install into THIS interpreter, e.g.:\n"
            f"  {py} -m pip install -U matplotlib\n"
        )

    neg = y == 0
    pos = y == 1

    plt.figure(figsize=(7, 6))
    if neg.any():
        plt.scatter(z[neg, 0], z[neg, 1], s=10, alpha=0.6, label=f"negative (n={neg.sum()})")
    if pos.any():
        plt.scatter(z[pos, 0], z[pos, 1], s=10, alpha=0.6, label=f"positive (n={pos.sum()})")
    plt.title(f"PCA 2D (EVR: {evr[0]:.3f}, {evr[1]:.3f})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)

    print(f"Saved plot: {args.out_png}")
    print(f"N={len(y)} dim={X.shape[1]} | EVR PC1={evr[0]:.4f} PC2={evr[1]:.4f} (sum={evr.sum():.4f})")

    if args.out_npz:
        np.savez_compressed(args.out_npz, z=z.astype(np.float32), y=y, evr=evr.astype(np.float32))
        print(f"Saved coords: {args.out_npz}")


if __name__ == "__main__":
    main()

