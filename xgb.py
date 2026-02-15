#!/usr/bin/env python3
"""
XGBoost baseline on saved embeddings.

Trains on a stratified 90% split and tests on 10% (within the same file).

Input format: .pt produced by `examples/nexar_*_embeddings.py`
  - X: FloatTensor [N, D]
  - y: LongTensor [N] (1=positive, 0=negative)
  - split: list[str] (optional)

Example:
  cd /home/asus/new/cosmos-predict2.5
  python xgb.py --data nexar_last5s_1fps.pt --test_size 0.1 --seed 0
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

    # XGBoost hyperparams (good starting point for small-ish embedding datasets)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--min_child_weight", type=float, default=1.0)

    ap.add_argument("--early_stopping_rounds", type=int, default=30)
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

    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError:
        py = sys.executable
        raise SystemExit(
            "xgboost is not installed in the Python you're using.\n\n"
            f"You're running: {py}\n\n"
            "Install into THIS interpreter, e.g.:\n"
            f"  {py} -m pip install -U xgboost\n"
        )

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

    # Try early stopping on a held-out eval set (the test set). For a stricter protocol,
    # you can later change this to a validation split inside Xtr.
    fit_kwargs = {}
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        fit_kwargs = {"eval_set": [(Xte, yte)], "verbose": False, "early_stopping_rounds": args.early_stopping_rounds}

    try:
        clf.fit(Xtr, ytr, **fit_kwargs)
    except TypeError:
        # API differences across xgboost versions
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
    print(
        "XGB params: "
        f"n_estimators={args.n_estimators} max_depth={args.max_depth} lr={args.learning_rate:g} "
        f"subsample={args.subsample:g} colsample={args.colsample_bytree:g} "
        f"l2={args.reg_lambda:g} l1={args.reg_alpha:g}"
    )
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print()
    print(classification_report(yte, pred, digits=4))


if __name__ == "__main__":
    main()

