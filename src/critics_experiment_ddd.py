"""
critics_experiments_ddd.py

Adds Deep Double Descent (DDD) considerations:
  F) Reproduce deep double descent: width sweep x label-noise sweep on MNIST/CIFAR10
  B) (updated) Parity complexity + label-noise sweep + width sweep
  C) (updated) Soft vs Hard bias under label-noise sweep
  D) (updated) SGD simplicity bias with label noise

Usage examples:
  python critics_experiments_ddd.py --exp F --dataset mnist --noise_list 0.0 0.2 0.4
  python critics_experiments_ddd.py --exp B --complexities 1 2 4 8 16 --noise_list 0.0 0.2
  python critics_experiments_ddd.py --exp C --ntrain_list 200 500 1000 5000 --noise_list 0.0 0.2 0.4
  python critics_experiments_ddd.py --exp D --noise_list 0.0 0.2
"""

import os, math, argparse, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

import torchvision
import torchvision.transforms as T
from util import RNG,set_seed,get_mnist,SmallCNN,MLPNoConv,train_test_split,save_plot,train_torch,accuracy_torch,build_loader_from_subset
# ===================================================
# B) UPDATED: Parity complexity + noise + width sweep
# ===================================================

def gen_parity_dataset(n: int, d: int, k: int, noise=0.0):
    X = RNG.normal(size=(n, d))
    S = RNG.choice(d, size=k, replace=False)
    bits = (X[:, S] > 0).astype(np.int32)
    y = (np.sum(bits, axis=1) % 2).astype(np.int64)
    if noise > 0:
        flip = RNG.random(n) < noise
        y = y ^ flip.astype(np.int64)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def exp_B(device="cpu", complexities=(1,2,4,8,16), n_train=5000, n_test=2000, d=100,
          noise_list=(0.0, 0.2, 0.4), widths=(64,128,256,512), epochs=15, lr=1e-3):
    set_seed(1)
    for noise in noise_list:
        res = []
        for k in complexities:
            # under- to over-param by width; record best and peak behavior
            test_err_vs_width = []
            for w in widths:
                in_dim = d
                model = nn.Sequential(
                    nn.Linear(in_dim, w), nn.ReLU(),
                    nn.Linear(w, w), nn.ReLU(),
                    nn.Linear(w, 2)
                ).to(device)
                Xtr, ytr = gen_parity_dataset(n_train, d, k, noise=noise)
                Xte, yte = gen_parity_dataset(n_test,  d, k, noise=noise)
                tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
                te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)
                _ = train_torch(model, tr_loader, te_loader, device, epochs=epochs, lr=lr, wd=0.0)
                te_acc = accuracy_torch(model, te_loader, device)
                test_err_vs_width.append(1 - te_acc)
            # Store minimal test error and location of peak/criticality proxy
            min_err = float(np.min(test_err_vs_width))
            peak_err = float(np.max(test_err_vs_width))
            peak_at  = widths[int(np.argmax(test_err_vs_width))]
            res.append((k, min_err, peak_err, peak_at))
            print(f"[B] noise={noise:.2f}, k={k}: min={min_err:.3f}, peak={peak_err:.3f} @ width={peak_at}")
        # plot: peak_at vs k (should shift left with more noise)
        fig, ax = plt.subplots()
        ax.plot([r[0] for r in res], [r[3] for r in res], marker="o")
        ax.set_xlabel("parity complexity k")
        ax.set_ylabel("width at peak test error (critical region proxy)")
        ax.set_title(f"B) critical-region shift vs k @ noise={noise}")
        save_plot(fig, f"B_parity_critical_shift_noise{int(100*noise)}")

# ===================================================
# C) UPDATED: Soft vs Hard bias with label noise
# ===================================================

def get_mnist_subset(n, translate_pixels):
    tr, te = get_mnist(n_train=n, n_test=2000, translate_pixels=translate_pixels)
    return tr, te

def exp_C(device="cpu", ntrain_list=(200, 500, 1000, 5000), translate_pixels=3,
          noise_list=(0.0, 0.2, 0.4), epochs=6, lr=1e-3):
    """
    Compare soft bias (MLP+augmentation) vs hard bias (CNN) under label noise.
    Expectation: with noise, test error increases and the data-efficiency gap can widen;
    also augmentation interacts with noise (too strong aug may hurt when labels are noisy).
    """
    set_seed(2)
    for noise in noise_list:
        out = []
        for ntr in ntrain_list:
            tr_soft_sub, te_sub = get_mnist_subset(ntr, translate_pixels=translate_pixels)
            tr_hard_sub, _      = get_mnist_subset(ntr, translate_pixels=0)
            tr_soft = build_loader_from_subset(tr_soft_sub, batch_size=128, noise=noise, n_classes=10, shuffle=True)
            tr_hard = build_loader_from_subset(tr_hard_sub, batch_size=128, noise=noise, n_classes=10, shuffle=True)
            te = build_loader_from_subset(te_sub, batch_size=256, noise=0.0, n_classes=10, shuffle=False)

            mlp = MLPNoConv(in_shape=(1,28,28), n_classes=10, width=1024, depth=2)
            cnn = SmallCNN(n_classes=10, base=32)

            _, te_acc_s = train_torch(mlp, tr_soft, te, device, epochs=epochs, lr=lr, wd=1e-4)
            _, te_acc_h = train_torch(cnn, tr_hard, te, device, epochs=epochs, lr=lr, wd=1e-4)
            out.append((ntr, te_acc_s, te_acc_h))
            print(f"[C] noise={noise:.2f}, n={ntr}: soft={te_acc_s:.3f} | hard={te_acc_h:.3f}")

        fig, ax = plt.subplots()
        ax.plot([o[0] for o in out], [1-o[1] for o in out], marker="o", label="soft: MLP+aug")
        ax.plot([o[0] for o in out], [1-o[2] for o in out], marker="x", label="hard: CNN")
        ax.set_xscale("log")
        ax.set_xlabel("train size")
        ax.set_ylabel("test error")
        ax.set_title(f"C) soft vs hard under label noise={noise}")
        ax.legend()
        save_plot(fig, f"C_soft_vs_hard_noise{int(100*noise)}")

# ===================================================
# D) UPDATED: SGD simplicity bias with label noise
# ===================================================

def make_gaussian(n=4000, d=100, margin=1.0, noise=0.0):
    Xpos = RNG.normal(size=(n//2, d)); Xneg = RNG.normal(size=(n//2, d))
    Xpos[:,0] += margin; Xneg[:,0] -= margin
    X = np.vstack([Xpos, Xneg]).astype(np.float32)
    y = np.hstack([np.ones(n//2), np.zeros(n//2)]).astype(np.int64)
    # flip some labels
    if noise > 0:
        m = int(round(noise * n))
        idx = RNG.choice(n, m, replace=False)
        y[idx] = 1 - y[idx]
    perm = RNG.permutation(n)
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])

class LinearClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return x @ self.w + self.b
    def logits(self, x):
        return torch.stack([torch.zeros_like(self.forward(x)), self.forward(x)], dim=1)

def logistic_train_sgd(Xtr, ytr, Xte, yte, lr, bs, epochs=30, wd=0.0, device="cpu"):
    n, d = Xtr.shape
    model = LinearClassifier(d).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model.logits(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model.logits(Xte.to(device)).argmax(1).cpu().numpy()
        acc = (pred == yte.numpy()).mean()
        wnorm = torch.cat([model.w, model.b]).norm().item()
    return acc, wnorm

def exp_D(device="cpu", noise_list=(0.0, 0.2), margin=1.0):
    set_seed(3)
    for noise in noise_list:
        X, y = make_gaussian(n=4000, d=100, margin=margin, noise=noise)
        Xtr, Xte, ytr, yte = train_test_split(X.numpy(), y.numpy())
        Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
        ytr, yte = torch.tensor(ytr, dtype=torch.long), torch.tensor(yte, dtype=torch.long)

        configs = [
            ("SGD small-batch", 0.1, 32),
            ("SGD large-batch", 0.1, 512),
            ("Full-batch GD", 0.05, len(Xtr)),
            ("High LR", 0.5, 32),
        ]
        rows = []
        for name, lr, bs in configs:
            acc, wnorm = logistic_train_sgd(Xtr, ytr, Xte, yte, lr=lr, bs=bs, device=device)
            rows.append((name, 1-acc, wnorm))
            print(f"[D] noise={noise:.2f} {name}: test_err={1-acc:.3f}, ||w||={wnorm:.2f}")

        # plot: test error vs ||w|| for each optimizer config
        fig, ax = plt.subplots()
        for name, terr, wnorm in rows:
            ax.scatter(wnorm, terr); ax.text(wnorm, terr, name)
        ax.set_xlabel("||w||2")
        ax.set_ylabel("test error")
        ax.set_title(f"D) simplicity-bias under noise={noise}")
        save_plot(fig, f"D_simplicity_noise{int(100*noise)}")

# --------------------- main ---------------------
def main_all():
    if( torch.cuda.is_available()):
        device ="cuda"
        print("device is ",device)        
        torch.device(device)
        exp_B(device=device)
        exp_C(device=device)
        exp_D(device=device)
    else:
        device="cpu"
        print("device is ",device)        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=list("BCDF"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--widths", type=int, nargs="*", default=[32,64,128,256,512,1024,2048])
    parser.add_argument("--noise_list", type=float, nargs="*", default=[0.0, 0.2, 0.4])
    parser.add_argument("--complexities", type=int, nargs="*", default=[1,2,4,8,16])
    parser.add_argument("--ntrain_list", type=int, nargs="*", default=[200,500,1000,5000])
    args = parser.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    print("device is ",device)        
    if args.exp == "B":
        exp_B(device=device, complexities=tuple(args.complexities), noise_list=tuple(args.noise_list))
    elif args.exp == "C":
        exp_C(device=device, ntrain_list=tuple(args.ntrain_list), noise_list=tuple(args.noise_list))
    elif args.exp == "D":
        exp_D(device=device, noise_list=tuple(args.noise_list))
    else:
        main_all()
        
if __name__ == "__main__":
    main()
