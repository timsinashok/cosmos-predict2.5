#!/usr/bin/env python3
"""
Logistic regression baseline on embeddings from `examples/nexar_midframe_embeddings.py`.

Example:
  cd /home/asus/new/cosmos-predict2.5
  python logreg.py --data nexar_midframe_feats.pt --train_split test-private --eval_split test-public
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train_split", default="test-private")
    ap.add_argument("--eval_split", default="test-public")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength.")
    ap.add_argument("--class_weight", default="balanced", help="Use 'balanced' or leave empty for None.")
    ap.add_argument("--max_iter", type=int, default=5000)
    args = ap.parse_args()

    d = torch.load(args.data, map_location="cpu")
    X = d["X"].numpy().astype(np.float32)
    y = d["y"].numpy().astype(np.int64)
    split = np.array(d["split"])

    tr = split == args.train_split
    te = split == args.eval_split
    if tr.sum() == 0:
        raise ValueError(f"No samples for train_split={args.train_split}")
    if te.sum() == 0:
        raise ValueError(f"No samples for eval_split={args.eval_split}")

    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    except ModuleNotFoundError as e:
        if e.name not in ("sklearn", "scikit_learn"):
            raise
        py = sys.executable
        raise SystemExit(
            "scikit-learn is not installed in the Python you're using to run this script.\n\n"
            f"You're running: {py}\n\n"
            "Install into THIS interpreter, e.g.:\n"
            f"  {py} -m pip install -U scikit-learn\n"
        )

    cw = None if args.class_weight.strip().lower() in ("", "none") else args.class_weight

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            C=args.C,
            class_weight=cw,
            max_iter=args.max_iter,
            solver="lbfgs",
        ),
    )
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)

    acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, prob)
    except Exception:
        auc = float("nan")

    print(f"Train: {args.train_split} n={len(Xtr)} pos={(ytr==1).sum()} neg={(ytr==0).sum()}")
    print(f"Eval:  {args.eval_split} n={len(Xte)} pos={(yte==1).sum()} neg={(yte==0).sum()}")
    print(f"LogReg: C={args.C:g}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print()
    print(classification_report(yte, pred, digits=4))


if __name__ == "__main__":
    main()

