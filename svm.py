#!/usr/bin/env python3
"""
Train/evaluate a quick SVM on saved embeddings from `examples/nexar_midframe_embeddings.py`.

Requires:
  pip install scikit-learn

Example:
  cd /home/asus/new/cosmos-predict2.5
  python svm.py --data nexar_midframe_feats.pt --eval_split test-public --train_split test-private
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def _parse_float_list(s: str) -> list[float]:
    # "0.1,1,10" -> [0.1,1.0,10.0]
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _summarize_split(name: str, y: np.ndarray) -> str:
    return f"{name} n={len(y)} pos={(y==1).sum()} neg={(y==0).sum()}"


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="Path to .pt produced by nexar_midframe_embeddings.py")
    ap.add_argument("--train_split", default="test-private", help="Split name to train on.")
    ap.add_argument("--eval_split", default="test-public", help="Split name to eval on.")
    ap.add_argument("--kernel", default="rbf", choices=("linear", "rbf"))
    ap.add_argument("--C", type=float, default=1.0, help="Used when not doing --grid_search.")
    ap.add_argument("--gamma", default="scale", help="RBF gamma (e.g. 'scale' or '0.1'). Used when not doing --grid_search.")
    ap.add_argument("--class_weight", default="balanced", help="Use 'balanced' or leave empty for None.")
    ap.add_argument("--grid_search", action="store_true", help="Tune hyperparams on train split via CV (optimize ROC-AUC).")
    ap.add_argument("--cv", type=int, default=5, help="CV folds for --grid_search.")
    ap.add_argument("--Cs", default="0.01,0.1,1,10,100", help="Comma-separated C grid for --grid_search.")
    ap.add_argument(
        "--gammas",
        default="scale,0.001,0.01,0.1,1",
        help="Comma-separated gamma grid for RBF --grid_search. Use 'scale' to include sklearn default.",
    )
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
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
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

    def fit_eval(model: Pipeline):
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)
        acc = accuracy_score(yte, pred)
        try:
            auc = roc_auc_score(yte, prob)
        except Exception:
            auc = float("nan")
        return acc, auc, prob, pred

    def cv_auc_for_params(C: float, gamma):
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=0)
        aucs = []
        for tr_idx, va_idx in cv.split(Xtr, ytr):
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "svc",
                        SVC(
                            C=C,
                            kernel=args.kernel,
                            gamma=gamma if args.kernel == "rbf" else "scale",
                            class_weight=cw,
                            probability=True,
                        ),
                    ),
                ]
            )
            model.fit(Xtr[tr_idx], ytr[tr_idx])
            p = model.predict_proba(Xtr[va_idx])[:, 1]
            aucs.append(roc_auc_score(ytr[va_idx], p))
        return float(np.mean(aucs)), float(np.std(aucs))

    best = None
    best_cv_auc = -1.0
    best_cv_std = 0.0

    if args.grid_search:
        Cs = _parse_float_list(args.Cs)
        if args.kernel == "rbf":
            gammas: list[object] = []
            for g in (x.strip() for x in args.gammas.split(",")):
                if not g:
                    continue
                if g.lower() == "scale":
                    gammas.append("scale")
                else:
                    gammas.append(float(g))
        else:
            gammas = ["scale"]

        print(_summarize_split(f"Train({args.train_split})", ytr))
        print(_summarize_split(f"Eval({args.eval_split})", yte))
        print(f"Grid search: kernel={args.kernel} cv={args.cv} | Cs={Cs} | gammas={gammas if args.kernel=='rbf' else 'n/a'}")
        print()

        for C in Cs:
            for gamma in gammas:
                mean_auc, std_auc = cv_auc_for_params(C=C, gamma=gamma)
                print(f"CV AUC {mean_auc:.4f} ± {std_auc:.4f} | C={C:g} gamma={gamma}")
                if mean_auc > best_cv_auc:
                    best_cv_auc = mean_auc
                    best_cv_std = std_auc
                    best = (C, gamma)

        assert best is not None
        C_best, gamma_best = best
        print()
        print(f"Best CV AUC: {best_cv_auc:.4f} ± {best_cv_std:.4f} | C={C_best:g} gamma={gamma_best}")

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "svc",
                    SVC(
                        C=C_best,
                        kernel=args.kernel,
                        gamma=gamma_best if args.kernel == "rbf" else "scale",
                        class_weight=cw,
                        probability=True,
                    ),
                ),
            ]
        )
    else:
        gamma = args.gamma
        if args.kernel == "rbf" and isinstance(gamma, str) and gamma.lower() != "scale":
            gamma = float(gamma)
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("svc", SVC(C=args.C, kernel=args.kernel, gamma=gamma, class_weight=cw, probability=True)),
            ]
        )

    acc, auc, prob, pred = fit_eval(clf)

    print()
    print(f"Train: {args.train_split} n={len(Xtr)} pos={(ytr==1).sum()} neg={(ytr==0).sum()}")
    print(f"Eval:  {args.eval_split} n={len(Xte)} pos={(yte==1).sum()} neg={(yte==0).sum()}")
    if args.grid_search:
        print(f"Chosen: kernel={args.kernel} C={clf.named_steps['svc'].C:g} gamma={clf.named_steps['svc'].gamma}")
    else:
        print(f"Model:  kernel={args.kernel} C={args.C:g} gamma={args.gamma}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")
    print()
    print(classification_report(yte, pred, digits=4))


if __name__ == "__main__":
    main()

