#!/usr/bin/env python
"""
Numerical experiments supporting Wilson (2025) claims:

  1. Benign overfitting in linear regression
  2. Double descent in kernel ridge regression
  3. Double descent in gradient-boosted decision trees
  4. PAC-Bayes bound vs. generalization gap (linear model)
  5. Representation learning: CNN vs. Random Fourier Features

Usage:
    python run_experiments.py --exp 1
    python run_experiments.py --exp 2 --n_trials 10
    python run_experiments.py --exp 5 --device cuda
"""

import argparse, os, random, math, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# ==== common utilities ======================================================

RNG = np.random.default_rng(123)

def train_test_split(X, y, test_frac=0.2):
    n = len(X)
    idx = RNG.permutation(n)
    cut = int(n * (1 - test_frac))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def save_plot(fig, name: str):
    out = Path("figs")
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")

# ==== 1. Benign overfitting: linear regression ==============================
# https://www.arxiv.org/abs/2112.02809v4
def exp1_lin_benign(
    d: int = 5000,
    n_list: Tuple[int] = (100, 200, 400, 800, 1600),
    sparsity: int = 50,
    noise_std: float = 0.5,
    debug=False
):
    from sklearn.linear_model import Ridge,LinearRegression

    # true sparse weight vector
    w_true = np.zeros(d)
    w_true[RNG.choice(d, sparsity, replace=False)] = RNG.normal(size=sparsity)

    results = []
    for n in n_list:
        X = RNG.normal(size=(n, d))
        y = X @ w_true + RNG.normal(scale=noise_std, size=n)
        Xtr, Xte, ytr, yte = train_test_split(X, y)

        # α → 0 interpolation regime
        for i,salpha in enumerate[1e-12,1e-10,1e-8,1e-6]:
            [(,lin_interps[i]) = Ridge(alpha=salpha, fit_intercept=False).fit(Xtr, ytr)

        # slightly regularized baseline
        lin_reg = Ridge(alpha=1.0, fit_intercept=False).fit(Xtr, ytr)

        #for tag, mdl in [("interp", lin_interp)]+[("ridge1", lin_reg)]:
        for tag, mdl in [("interp", lin_interp)]+[("ridge1", lin_reg)]:
            results.append(
                dict(
                    n=n,
                    regime=tag,
                    train_mse=mse(ytr, mdl.predict(Xtr)),
                    test_mse=mse(yte, mdl.predict(Xte)),
                )
            )

    if(debug):
        for tag in ("interp","ridge1"):
            xs = [r["n"] for r in results if r["regime"] == tag]
            ys = [r["test_mse"] for r in results if r["regime"] == tag]
        print(tag,xs,ys)

    # plot
    fig, ax = plt.subplots()
    for tag in ("interp", "ridge1"):
        xs = [r["n"] for r in results if r["regime"] == tag]
        ys = [r["test_mse"] for r in results if r["regime"] == tag]
        ax.plot(xs, ys, marker="o", label=tag)
   
    ax.axhline(noise_std**2, ls="--", lw=1, color="gray", label="noise variance")
    ax.set_xscale("log")
    ax.set_xlabel("train samples n")
    ax.set_ylabel("test MSE")
    ax.set_title("Benign overfitting (linear regression)")
    ax.legend()
    save_plot(fig, "exp1_benign")
    print("Experiment 1 finished → figs/exp1_benign.png")

# ==== 2. Double descent: kernel ridge regression ============================

def exp2_krr_dd(
    n_train: int = 1000,
    dataset: str = "fashion",
    bandwidths=np.logspace(-2, 2, 20),
    reg_lambda: float = 1e-6,
):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.datasets import fetch_openml

    # --- data ---------------------------------------------------------------
    if dataset == "fashion":
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
        y = (y.astype(int) % 10)  # 0-9
    elif dataset == "cifar_10_small":
        from torchvision.datasets import CIFAR10
        import torchvision.transforms as T, torch

        tf = T.Compose([T.ToTensor()])
        tmp = CIFAR10(root="data", download=True, transform=tf)
        X = torch.stack([tf(img) for img, _ in tmp]) \
                .view(len(tmp), -1).numpy()
        y = np.array([int(lbl) for _, lbl in tmp])
    else:
        raise ValueError("unknown dataset")

    X = X / 255.0
    sel = RNG.choice(len(X), n_train, replace=False)
    X, y = X[sel], y[sel]
    Xtr, Xte, ytr, yte = train_test_split(X, y)

    # --- sweep --------------------------------------------------------------
    tr_err, te_err = [], []
    for bw in bandwidths:
        mdl = KernelRidge(
            alpha=reg_lambda, kernel="rbf", gamma=1.0 / (2 * bw**2)
        ).fit(Xtr, ytr)
        tr_err.append(np.mean((mdl.predict(Xtr).round() != ytr)))
        te_err.append(np.mean((mdl.predict(Xte).round() != yte)))

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(bandwidths, te_err, marker="o", label="test")
    ax.plot(bandwidths, tr_err, marker="x", label="train")
    ax.set_xscale("log")
    ax.set_xlabel("RBF bandwidth σ")
    ax.set_ylabel("classification error")
    ax.set_title("Double descent in KRR")
    ax.legend()
    save_plot(fig, "exp2_krr_dd")
    print("Experiment 2 finished → figs/exp2_krr_dd.png")

# ==== 3. Double descent: Gradient Boosted Trees ============================

def exp3_gbdt_dd(
    n_estimators_list=(10, 20, 40, 80, 160, 320, 640, 1280),
    dataset="higgs",
):
    from sklearn.metrics import log_loss
    from sklearn.datasets import fetch_openml
    import xgboost as xgb

    if dataset == "higgs":
        X, y = fetch_openml("HIGGS", version=1, as_frame=False, return_X_y=True)
        y = y.astype(int)
        sel = RNG.choice(len(X), 100_000, replace=False)  # subsample ↓
        X, y = X[sel], y[sel]
    else:
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y)

    tr_err, te_err = [], []
    for n_est in n_estimators_list:
        mdl = xgb.XGBClassifier(
            n_estimators=n_est,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            n_jobs=8,
            random_state=42,
        ).fit(Xtr, ytr, eval_set=[(Xtr, ytr)], verbose=False)

        tr_err.append(log_loss(ytr, mdl.predict_proba(Xtr)[:, 1]))
        te_err.append(log_loss(yte, mdl.predict_proba(Xte)[:, 1]))

    fig, ax = plt.subplots()
    ax.plot(n_estimators_list, te_err, marker="o", label="test")
    ax.plot(n_estimators_list, tr_err, marker="x", label="train")
    ax.set_xscale("log")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("log-loss")
    ax.set_title("Double descent in GBDT")
    ax.legend()
    save_plot(fig, "exp3_gbdt_dd")
    print("Experiment 3 finished → figs/exp3_gbdt_dd.png")

# ==== 4. PAC-Bayes bound vs. generalization gap (linear) ====================

def exp4_pacbayes(
    n: int = 500,
    d: int = 300,
    noise_std: float = 1.0,
    delta: float = 0.05,
):
    # --- synthetic linear-Gaussian model ------------------------------------
    X = RNG.normal(size=(n, d))
    w0 = RNG.normal(size=d)
    y = X @ w0 + RNG.normal(scale=noise_std, size=n)

    Xtr, Xte, ytr, yte = train_test_split(X, y)

    # MAP under prior ~ N(0, τ²I) and σ² noise  ⇒ ridge regression closed form
    tau2 = 1.0
    sigma2 = noise_std**2
    A = Xtr.T @ Xtr + (sigma2 / tau2) * np.eye(d)
    w_map = np.linalg.solve(A, Xtr.T @ ytr)

    # ----- empirical quantities --------------------------------------------
    train_mse = mse(ytr, Xtr @ w_map)
    test_mse = mse(yte, Xte @ w_map)
    gap = test_mse - train_mse
    print(f"empirical gap = {gap:.4f}")

    # ----- PAC-Bayes Catoni bound ------------------------------------------
    # posterior Q = N(w_map, σ² A^{-1});   prior P = N(0, τ² I)
    # KL(Q||P) = ½ (tr(Σ_p^{-1} Σ_q) + w_mapᵀ Σ_p^{-1} w_map − d + ln|Σ_p|/|Σ_q|)
    Σq = sigma2 * np.linalg.inv(A)
    Σp_inv = (1 / tau2) * np.eye(d)
    kl = 0.5 * (
        np.trace(Σp_inv @ Σq)
        + w_map @ Σp_inv @ w_map
        - d
        + np.log(np.linalg.det(tau2 * np.eye(d)))  # |Σ_p|
        - np.log(np.linalg.det(Σq))
    )
    bound = math.sqrt((kl + math.log(2 * math.sqrt(n) / delta)) / (2 * (n - 1)))
    print(f"PAC-Bayes Δ = {bound:.4f}")

# ==== 5. Representation learning: CNN vs. RFF ===============================

def exp5_repr(
    n_train: int = 10_000,
    n_probe_train: int = 1000,
    device: str = "cpu",
):
    import torch, torchvision
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = torch.device(device)

    # ---- CIFAR-10 ----------------------------------------------------------
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    cifar_train = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    X = torch.stack([img for img, _ in cifar_train])[:n_train]
    y = torch.tensor([lbl for _, lbl in cifar_train])[:n_train]

    # ---- pretrained ResNet18 ----------------------------------------------
    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    resnet.fc = torch.nn.Identity()
    resnet.to(device).eval()

    with torch.no_grad():
        feats = resnet(X.to(device)).cpu().numpy()

    # ---- linear probe ------------------------------------------------------
    Xtr_p, Xte_p, ytr_p, yte_p = train_test_split(
        feats, y.numpy(), test_frac=0.5
    )
    probe = LogisticRegression(max_iter=1000, n_jobs=8).fit(Xtr_p, ytr_p)
    acc_probe = accuracy_score(yte_p, probe.predict(Xte_p))
    print(f"Linear probe accuracy (ResNet features) = {acc_probe:.3f}")

    # ---- Random Fourier Features baseline ---------------------------------
    from sklearn.kernel_approximation import RBFSampler

    rff = RBFSampler(gamma=1.0, n_components=2048, random_state=42)
    feats_rff = rff.fit_transform(X.view(n_train, -1).numpy())

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        feats_rff, y.numpy(), test_frac=0.5
    )
    clf_rff = LogisticRegression(max_iter=1000, n_jobs=8).fit(Xtr_r, ytr_r)
    acc_rff = accuracy_score(yte_r, clf_rff.predict(Xte_r))
    print(f"RFF + logreg accuracy = {acc_rff:.3f}")

    print(
        f"Representation gain = {(acc_probe - acc_rff)*100:.1f} pp"
    )

# ==== driver ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_trials", type=int, default=1)
    args = parser.parse_args()

    if args.exp == 1:
        exp1_lin_benign(debug=True)
    elif args.exp == 2:
        exp2_krr_dd()
    elif args.exp == 3:
        exp3_gbdt_dd()
    elif args.exp == 4:
        exp4_pacbayes()
    elif args.exp == 5:
        exp5_repr(device=args.device)

if __name__ == "__main__":
    main()
