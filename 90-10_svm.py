#!/usr/bin/env python3
"""
Linear SVM baseline with a stratified 90/10 split.

Works on embeddings saved by:
- examples/nexar_midframe_embeddings.py
- examples/nexar_last5s_1fps_embeddings.py

Input .pt format:
  - X: FloatTensor [N, D]
  - y: LongTensor [N] (1=positive, 0=negative)
  - split: list[str] (optional)

Example:
  cd /home/asus/new/cosmos-predict2.5
  python 90-10_svm.py --data nexar_last5s_1fps.pt --test_size 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to .pt with X/y.")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_split", default="all", help="Optional filter, e.g. test-private / test-public / all.")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class_weight", default="balanced", help="Use 'balanced' or leave empty for None.")
    args = ap.parse_args()

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

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test_size must be in (0,1)")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
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

    cw = None if args.class_weight.strip().lower() in ("", "none") else args.class_weight

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        SVC(C=args.C, kernel="linear", class_weight=cw, probability=True),
    )
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)

    acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, prob)
    except Exception:
        auc = float("nan")

    print(f"Data: {args.data}")
    if args.use_split != "all":
        print(f"Filtered split: {args.use_split}")
    print(f"Train n={len(Xtr)} pos={(ytr==1).sum()} neg={(ytr==0).sum()}")
    print(f"Test  n={len(Xte)} pos={(yte==1).sum()} neg={(yte==0).sum()}")
    print(f"Linear SVM: C={args.C:g}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print()
    print(classification_report(yte, pred, digits=4))


if __name__ == "__main__":
    main()

