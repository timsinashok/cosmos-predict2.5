#!/usr/bin/env python3
"""
Train/evaluate a small MLP on saved embeddings from `examples/nexar_midframe_embeddings.py`.

Notes:
- Works best with scaling (we use StandardScaler).
- With small datasets, use early stopping + regularization to avoid overfitting.

Example:
  cd /home/asus/new/cosmos-predict2.5
  python mlp.py --data nexar_midframe_feats.pt --train_split test-private --eval_split test-public
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def _parse_int_list(s: str) -> tuple[int, ...]:
    # "64,32" -> (64,32)
    s = s.strip()
    if not s:
        return tuple()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to .pt produced by nexar_midframe_embeddings.py")
    ap.add_argument("--train_split", default="test-private", help="Split name to train on.")
    ap.add_argument("--eval_split", default="test-public", help="Split name to eval on.")

    ap.add_argument("--hidden", default="64", help="Hidden layer sizes, e.g. '64' or '128,64'.")
    ap.add_argument("--activation", default="relu", choices=("relu", "tanh", "logistic"))
    ap.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--early_stopping", action="store_true", default=True)
    ap.add_argument("--no_early_stopping", action="store_false", dest="early_stopping")
    ap.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction for early stopping.")
    ap.add_argument("--patience", type=int, default=20, help="n_iter_no_change for early stopping.")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--class_weight", default="balanced", help="Use 'balanced' or leave empty for None.")
    args = ap.parse_args()

    d = torch.load(args.data, map_location="cpu")
    X = d["X"].numpy().astype(np.float32)  # (N,C)
    y = d["y"].numpy().astype(np.int64)  # (N,)
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
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
        from sklearn.utils.class_weight import compute_sample_weight
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

    hidden = _parse_int_list(args.hidden)
    if len(hidden) == 0:
        raise ValueError("--hidden must specify at least one layer size, e.g. --hidden 64")

    cw = None if args.class_weight.strip().lower() in ("", "none") else args.class_weight
    sample_weight = compute_sample_weight(class_weight=cw, y=ytr) if cw is not None else None

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=args.activation,
            alpha=args.alpha,
            learning_rate_init=args.lr,
            max_iter=args.max_iter,
            early_stopping=args.early_stopping,
            validation_fraction=args.val_frac,
            n_iter_no_change=args.patience,
            random_state=args.seed,
        ),
    )

    clf.fit(Xtr, ytr, mlpclassifier__sample_weight=sample_weight)

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)

    acc = accuracy_score(yte, pred)
    try:
        auc = roc_auc_score(yte, prob)
    except Exception:
        auc = float("nan")

    print(f"Train: {args.train_split} n={len(Xtr)} pos={(ytr==1).sum()} neg={(ytr==0).sum()}")
    print(f"Eval:  {args.eval_split} n={len(Xte)} pos={(yte==1).sum()} neg={(yte==0).sum()}")
    print(f"MLP hidden={hidden} activation={args.activation} alpha={args.alpha:g} lr={args.lr:g}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print()
    print(classification_report(yte, pred, digits=4))


if __name__ == "__main__":
    main()

