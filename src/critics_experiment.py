"""
critics_experiments.py
Implements numerical experiments inspired by "Critics / Alternative Ideas" of soft_inductive_bias:

 A) PAC-MDL (compression-as-prior) vs generalization
 B) Data algorithmic complexity sweep (parity / k-DNF like)
 C) Soft vs Hard inductive bias: augmentation vs convolutional weight sharing
 D) SGD simplicity bias: margin & weight norm under LR/BS schedules
 E) Small-net Hessian degeneracy (flatness) vs generalization

Usage examples:
  python critics_experiments.py --exp A
  python critics_experiments.py --exp B --complexities 1 2 4 8 16
  python critics_experiments.py --exp C --ntrain_list 200 500 1000 5000
  python critics_experiments.py --exp D
  python critics_experiments.py --exp E --device cuda
"""
import os, math, argparse, random, time
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Optional imports (will be used when available)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.kernel_approximation import RBFSampler
import lanczos 
import hpv
import pandas as pd
import seaborn as sns
from util import RNG,set_seed,get_mnist,uniform_quantize,mdl_code_length_bits,SmallCNN,train_test_split,save_plot,train_torch
import lightning_vision as lv

# ------------- A) PAC-MDL (compression) -------------------------------------

def exp_A(device="cpu", model_type="mlp", keep_ratio_list=(0.1,0.2,0.4,0.8), quant_bits=(8,6,4),batch_size=128):
    """
    Train a small model, compress (prune+quantize), convert compression to MDL bits,
    and compare empirical generalization gap with Catoni-style PAC-Bayes using KL≈bits/ln2.
    """
    set_seed(0)
    train_ds, test_ds = get_mnist(n_train=10000, n_test=2000, translate_pixels=0)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=2)

    if model_type == "mlp":
        model = MLP(in_dim=28*28, n_classes=10, hidden=512)
    elif model_type == "resnet18":
        model = nn.model.MLP(in_dim=28*28, n_classes=10, hidden=512)
    elif model_type == "cnn":
        model = SmallCNN(n_classes=10)
    else:
        model = SmallCNN(n_classes=10)

    tr_acc, te_acc = train_torch(model, train_loader, test_loader, device, epochs=6, lr=1e-3, wd=1e-4)
    print(f"[A] trained base: train_acc={tr_acc:.3f}, test_acc={te_acc:.3f}")
    # empirical generalization gap by cross-entropy on test/train
    def ce_on(loader):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                ce = F.cross_entropy(model(x), y, reduction="sum").item()
                tot += ce; n += y.numel()
        return tot/n

    ce_tr = ce_on(train_loader); ce_te = ce_on(test_loader)
    gap = ce_te - ce_tr
    # flatten parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(f"critics_A_pacmdl_{model_type}.log","w") as fp:
        print(f"[A] empirical CE gap = {gap:.4f}",file=fp)
        print(f"[batch_size={batch_size}",file=fp)
        print(f"[A] trained base: train_acc={tr_acc:.3f}, test_acc={te_acc:.3f}",file=fp)
        rows = []
        for kr in keep_ratio_list:
            _, kept = global_magnitude_prune(model, keep_ratio=kr)
            nz = kept.numel()
            for qb in quant_bits:
                _, mse = uniform_quantize(kept.clone(), qb)
                bits = mdl_code_length_bits(n_params=n_params, nz=nz, value_bits=qb, index_bits_exact=False)
                # PAC-Bayes bound: Δ ≈ sqrt((KL+ln(2√n/δ))/2(n-1)), KL≈bits/ln2
                n = len(train_ds)
                delta = 0.05 #?
                KL = bits / math.log(2)   # convert bits -> nats
                bound = math.sqrt((KL + math.log(2*math.sqrt(n)/delta)) / (2*(n-1)))
                #rows.append((kr, qb, nz, bits, bound, mse))
                rows.append({"keep_ratio":kr, "quantbit":qb, "kept":nz, "mdlbit":bits, "mdlbound":bound, "mse":mse})
                print(f"[A] keep={kr:.2f}, q={qb}bits -> nz={nz}, bits={bits}, Δ≈{bound:.4f}, qMSE={mse:.2e}",file=fp)
    
    df=pd.DataFrame.from_records(rows)
    pg=sns.pairplot(df)
    pg.savefig("")

    # quick plot: bound vs keep ratio
    fig, ax = plt.subplots()
    for qb in quant_bits:
        xs = [r["keep_ratio"] for r in rows if r["quantbit"]==qb]
        ys = [r["mdlbound"] for r in rows if r["quantbit"]==qb]
        ax.plot(xs, ys, marker="o", label=f"{qb} bits")
    ax.set_xlabel("keep ratio (global magnitude prune)")
    ax.set_ylabel("PAC-MDL bound Δ is(approx)")
    ax.set_title("A) Compression-as-prior PAC-MDL bound")
    ax.legend()
    save_plot(fig, f"critics_A_pacmdl_{model_type}")

# ===================================================
# B)  Parity complexity + noise + width sweep
# ===================================================

def gen_parity_dataset(n: int, d: int, k: int, noise=0.0):
    """
    X ~ N(0,1)^d; label y = XOR[ sign(x_i>0) for i in subset S of size k ]
    Parity complexity grows with k (more bits to specify subset & larger Fourier mass at high order).
    """
    X = RNG.normal(size=(n, d))
    S = RNG.choice(d, size=k, replace=False)
    bits = (X[:, S] > 0).astype(np.int32)
    y = (np.sum(bits, axis=1) % 2).astype(np.int64)
    # flip with noise
    if noise > 0:
        flip = RNG.random(n) < noise
        y = y ^ flip.astype(np.int64)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_simple_mlp_classifier(Xtr, ytr, Xte, yte, device, width=512, epochs=20, lr=1e-3, wd=0.0):
    in_dim = Xtr.size(1)
    model = nn.Sequential(
        nn.Linear(in_dim, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, 2)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tr_ds = TensorDataset(Xtr, ytr); te_ds = TensorDataset(Xte, yte)
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        tr_acc = (model(Xtr.to(device)).argmax(1) == ytr.to(device)).float().mean().item()
        te_acc = (model(Xte.to(device)).argmax(1) == yte.to(device)).float().mean().item()
    return tr_acc, te_acc

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
    return get_mnist(n_train=n, n_test=2000, translate_pixels=translate_pixels)

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

# ------------- D) SGD simplicity bias ---------------------------------------

def make_separable_gaussian(n=1000, d=50, margin=1.0):
    """
    Two Gaussians with means +/- m along the first axis to create separability with margin.
    """
    Xpos = RNG.normal(size=(n//2, d)); Xneg = RNG.normal(size=(n//2, d))
    Xpos[:,0] += margin; Xneg[:,0] -= margin
    X = np.vstack([Xpos, Xneg]).astype(np.float32)
    y = np.hstack([np.ones(n//2), np.zeros(n//2)]).astype(np.int64)
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

def logistic_train_sgd(Xtr, ytr, Xte, yte, lr, bs, epochs=20, wd=0.0, device="cpu"):
    n, d = Xtr.shape
    model = LinearClassifier(d).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model.logits(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        w = torch.cat([model.w, model.b])
        wnorm = w.norm().item()
        # geometric margin on test: y∈{0,1} -> map to {-1,+1}
        x = Xte.to(device)
        yy = (2*yte-1).to(device).float()
        margin = torch.min( (x@model.w + model.b).flatten() * yy ).item()
        pred = model.logits(x).argmax(1).cpu().numpy()
        acc = (pred == yte.numpy()).mean()
    return acc, wnorm, margin

def exp_D(device="cuda"):
    set_seed(3)
    X, y = make_separable_gaussian(n=4000, d=100, margin=1.0)
    Xtr, Xte, ytr, yte = train_test_split(X.numpy(), y.numpy())
    Xtr, Xte, ytr, yte = map(lambda a: torch.tensor(a, dtype=torch.float32) if a.ndim==2 else torch.tensor(a, dtype=torch.long), [Xtr, Xte, ytr, yte])

    configs = [
        ("SGD small-batch", 0.1, 32),
        ("SGD large-batch", 0.1, 512),
        ("Full-batch GD", 0.05, len(Xtr)),
        ("High LR", 0.5, 32),
    ]
    rows = []
    for name, lr, bs in configs:
        acc, wnorm, margin = logistic_train_sgd(Xtr, ytr, Xte, yte, lr=lr, bs=bs, device=device)
        rows.append((name, acc, wnorm, margin))
        print(f"[D] {name}: acc={acc:.3f}, ||w||={wnorm:.2f}, margin={margin:.3f}")

    # Plot norm vs margin
    fig, ax = plt.subplots()
    for name, acc, wnorm, margin in rows:
        ax.scatter(wnorm, margin); ax.text(wnorm, margin, name)
    ax.set_xlabel("||w||2"); ax.set_ylabel("geometric margin (test min)")
    ax.set_title("D) SGD simplicity bias: norm vs margin")
    save_plot(fig, "critics_D_margin_norm")

# ------------- E) Hessian degeneracy ----------------------------------------

def topk_eigs_symmetric(H, k=20):
    # simple power-iteration based Lanczos via torch.lobpcg if available
    try:
        vals, _ = torch.lobpcg(H, k=min(k, H.size(0)-2), largest=False)
        return vals.cpu().numpy()
    except Exception:
        # fallback: numpy eigh (dense) for small problems
        return np.linalg.eigvalsh(H.cpu().numpy())

def hessian_matrix(model, loss_fn, x, y):
    # compute H = ∇^2_w L at a minibatch (for small nets)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True)
    gvec = torch.cat([g.contiguous().view(-1) for g in grads])
    H_rows = []
    for i in range(gvec.numel()):
        gi = torch.autograd.grad(gvec[i], [p for p in model.parameters() if p.requires_grad], retain_graph=True)
        gi_vec = torch.cat([g.contiguous().view(-1) for g in gi]).detach()
        H_rows.append(gi_vec)
        if i >= 3000:  # cap for speed
            break
    H = torch.stack(H_rows, dim=0)
    return H

def exp_E_Hessian_deg(device="cpu", n_train=2000, n_test=1000,itenum=80,maxepochs=6,simple=False,seed=4):
    """
    Small MLP on MNIST subset; compute approximate Hessian rows to estimate small eigenvalues.
    Correlate fraction of near-zero eigenvalues with generalization gap across checkpoints.
    """
    assert torchvision is not None, "torchvision required"
    set_seed(seed)
    train_ds, test_ds = get_mnist(n_train=n_train, n_test=n_test, translate_pixels=0)
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=256, shuffle=False)
    
    #model = MLP(in_dim=28*28, n_classes=10, hidden=256).to(device)
    model = MLP(in_dim=28*28, depth=3,n_classes=10, hidden=256).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn= F.cross_entropy
    checkpoints = []
    for ep in range(maxepochs):
        model.train()
        for x,y in tr: 
            x,y = x.to(device), y.to(device)
            loss = loss_fn(model(x),y)
            opt.zero_grad(); 
            loss.backward(); 
            opt.step()
        tr_acc = accuracy_torch(model, tr, device)
        te_acc = accuracy_torch(model, te, device)
        checkpoints.append((ep+1, tr_acc, te_acc))
        print(f"[E] epoch {ep+1}: train={tr_acc:.3f}, test={te_acc:.3f}")

    if(simple): #simple approach
        # pick last minibatch to build Hessian approx
        x, y = next(iter(tr))
        x, y = x.to(device)[:64], y.to(device)[:64]
        # approximate spectrum of H^T H to see number of tiny curvatures
        H = hessian_matrix(model, lambda logits, yy: loss_fn(logits, yy), x, y)  # m x P
        HT_H = H.T @ H
        vals = topk_eigs_symmetric(HT_H.cpu(), k=50)
    else:
        #v=hpv.getvec(model,loss,tr)
        rank_est,vals,vecs= lanczos._get_eigenvecs(device,model,loss_fn,tr,topk=50,m=itenum)

    tiny = (vals < 1e-6).sum()
    frac_tiny = tiny / len(vals)
    print(f"[E] tiny-eig fraction (approx) = {frac_tiny:.3f}")
    print(f'rank estimation{rank_est}')
          
    # plot curve of (epoch vs gen gap) and annotate tiny-eig fraction
    fig, ax = plt.subplots()
    ep = [c[0] for c in checkpoints]
    gap = [c[1]-c[2] for c in checkpoints]  # train acc - test acc
    ax.plot(ep, gap, marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("train-test acc gap")
    ax.set_title(f"E) Hessian degeneracy proxy: tiny-eig≈{frac_tiny:.2f}")
    save_plot(fig, f"critics_E_hessian_deg_epoch{maxepochs}")

# ===================================================
# F) Deep Double Descent: width sweep × label-noise
# ===================================================

def exp_F(dataset="mnist", device="cpu",
          widths=(32, 64, 128, 256, 512, 1024, 2048),
          noise_list=(0.0, 0.2, 0.4),
          n_train=20000, n_test=5000, epochs=8, lr=1e-3):
    """
    Reproduce DDD by sweeping model width (capacity) at multiple label-noise rates.
    Expectation: higher noise -> test error peak higher and shifts to smaller width (earlier).
    """
    set_seed(0)
    if dataset.lower() == "mnist":
        train_sub, test_sub = get_mnist(n_train=n_train, n_test=n_test)
        n_classes = 10
        in_dim = 28*28
    else:
        raise ValueError("Only mnist is included for a fast demo.")

    fig, ax = plt.subplots()
    for noise in noise_list:
        tr_loader = build_loader_from_subset(train_sub, batch_size=128, noise=noise, n_classes=n_classes, shuffle=True)
        te_loader = build_loader_from_subset(test_sub,  batch_size=256, noise=0.0,   n_classes=n_classes, shuffle=False)  # test is clean
        test_errs, train_errs = [], []
        for w in widths:
            model = MLP(in_dim=in_dim, n_classes=n_classes, width=w, depth=3).to(device)
            tr_acc, te_acc = train_torch(model, tr_loader, te_loader, device, epochs=epochs, lr=lr, wd=1e-4)
            train_errs.append(1 - tr_acc);  test_errs.append(1 - te_acc)
            print(f"[F] noise={noise:.2f}, width={w}: train={tr_acc:.3f}, test={te_acc:.3f}")
        ax.plot(widths, test_errs, marker="o", label=f"noise={noise}")
    ax.set_xscale("log")
    ax.set_xlabel("width (model capacity)")
    ax.set_ylabel("test error")
    ax.set_title("F) Deep Double Descent: width × label-noise (MNIST)")
    ax.legend()
    save_plot(fig, "F_ddd_width_noise")

# ------------- main ----------------------------------------------------------
def main_all():
    if( torch.cuda.is_available()):
        device ="cuda"
        print("device is ",device)        
        torch.device(device)
        exp_A(device=device)
        exp_B(device=device)
        exp_C(device=device)
        exp_D(device=device)
        exp_E_Hessian_deg(device=device)
        exp_F(device=device)
    else:
        device="cpu"
        print("device is ",device)        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=list("ABCDE"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_complexity", type=int, default=64)
    parser.add_argument("--complexities", type=int, nargs="*", default=[1,2,4,8,16])
    parser.add_argument("--ntrain_list", type=int, nargs="*", default=[200,500,1000,5000])
    parser.add_argument("--maxepochs", type=int, default=6)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    if args.exp == "A":
        exp_A(device=device, keep_ratio_list=(0.1,0.2,0.4,0.8), quant_bits=(8,6,4,1),batch_size=128,model_type="mlp")
        exp_A(device=device, keep_ratio_list=(0.1,0.2,0.4,0.8), quant_bits=(8,6,4,1),batch_size=128,model_type="cnn")
    elif args.exp == "B":
        exp_B(device=device, max_complexity=args.max_complexity)
    elif args.exp == "C":
        exp_C(device=device, ntrain_list=args.ntrain_list)
    elif args.exp == "D":
        exp_D(device=device)
    elif args.exp == "E":
        exp_E_Hessian_deg(device=device,maxepochs=args.maxepochs)
    elif args.exp == "F":
        exp_F(device=device)        
    else:
        main_all()

if __name__ == "__main__":
    main()

