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
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional imports (will be used when available)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
try:
    import torchvision
    import torchvision.transforms as T
except Exception:
    torchvision = None
    T = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.kernel_approximation import RBFSampler
except Exception:
    LogisticRegression = None

import lanczos 
import hpv

RNG = np.random.default_rng(42)

# ------------- utils ---------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_plot(fig, name: str):
    ensure_dir("figs")
    fig.savefig(Path("figs") / f"{name}.png", dpi=180, bbox_inches="tight")

def train_test_split(X, y, test_frac=0.2):
    n = len(X)
    idx = RNG.permutation(n)
    cut = int(n * (1 - test_frac))
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]

def accuracy_torch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

# ------------- MDL (compression) helpers ------------------------------------

def global_magnitude_prune(model: nn.Module, keep_ratio: float = 0.2):
    # return mask dict and flattened weight vector after pruning
    with torch.no_grad():
        all_weights = torch.cat([p.view(-1).abs() for p in model.parameters() if p.requires_grad])
        k = max(1, int(len(all_weights) * keep_ratio))
        thresh = torch.topk(all_weights, k, largest=True).values.min()
        nz_bits = 0
        masks = []
        flat_vals = []
        for p in model.parameters():
            if not p.requires_grad: 
                masks.append(torch.ones_like(p, dtype=torch.bool))
                flat_vals.append(p.view(-1))
                continue
            m = (p.abs() >= thresh)
            masks.append(m)
            flat_vals.append(p[m].view(-1))
        return masks, torch.cat(flat_vals)

def uniform_quantize(x: torch.Tensor, nbits: int = 8):
    if x.numel() == 0: 
        return x, 0.0
    with torch.no_grad():
        xmin, xmax = x.min().item(), x.max().item()
        if xmin == xmax:  # constant
            return torch.zeros_like(x), 0.0
        qlevels = 2 ** nbits
        scale = (xmax - xmin) / (qlevels - 1)
        q = torch.clamp(((x - xmin) / scale).round(), 0, qlevels - 1)
        x_hat = q * scale + xmin
        mse = F.mse_loss(x_hat, x).item()
        return x_hat, mse

def elias_gamma_bits(n: int) -> int:
    if n <= 0: return 1
    l = int(math.floor(math.log2(n))) + 1
    return 2 * l - 1

def mdl_code_length_bits(n_params: int, nz: int, value_bits: int, index_bits_exact=False) -> int:
    """
    A simple upper bound of code length:
      1) encode nz using Elias-gamma
      2) encode indices: nz * ceil(log2(n_params)) bits (upper bound), or log2(nCr) (tighter)
      3) encode nonzero values: nz * value_bits
    """
    b1 = elias_gamma_bits(max(nz,1))
    if index_bits_exact and 0 < nz < n_params:
        # Stirling approx for log2(nCr)
        def log2fact(m):
            if m == 0 or m == 1: return 0.0
            return (m*math.log2(m) - m/math.log(2) + 0.5*math.log2(2*math.pi*m))
        b2 = int(round(log2fact(n_params) - log2fact(nz) - log2fact(n_params - nz)))
    else:
        b2 = nz * math.ceil(math.log2(max(n_params,2)))
    b3 = nz * value_bits
    return b1 + b2 + b3

# ------------- simple models -------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim=784, n_classes=10, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

class MLPNoConv(nn.Module):
    """For C: a non-convolutional MLP to compare with CNN on translation invariance."""
    def __init__(self, in_shape=(1,28,28), n_classes=10, width=1024):
        super().__init__()
        in_dim = int(np.prod(in_shape))
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, n_classes)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# ------------- datasets ------------------------------------------------------

def get_mnist(n_train=10000, n_test=2000, translate_pixels=0, flatten=False):
    assert torchvision is not None, "torchvision is required for MNIST"
    aug = []
    if translate_pixels > 0:
        aug.append(T.RandomAffine(degrees=0, translate=(translate_pixels/28, translate_pixels/28)))
    aug.append(T.ToTensor())
    tf_train = T.Compose(aug)
    tf_test  = T.Compose([T.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=tf_train)
    test_ds  = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=tf_test)
    # subsample
    train_idx = RNG.choice(len(train_ds), n_train, replace=False)
    test_idx  = RNG.choice(len(test_ds), n_test, replace=False)
    train_ds = Subset(train_ds, train_idx)
    test_ds  = Subset(test_ds,  test_idx)
    return train_ds, test_ds

# ------------- training loop -------------------------------------------------

def train_torch(model, train_loader, test_loader, device, epochs=5, lr=1e-3, wd=0.0):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    tr_acc = accuracy_torch(model, train_loader, device)
    te_acc = accuracy_torch(model, test_loader, device)
    return tr_acc, te_acc

# ------------- A) PAC-MDL (compression) -------------------------------------

def exp_A(device="cpu", model_type="mlp", keep_ratio_list=(0.1,0.2,0.4,0.8), quant_bits=(8,6,4)):
    """
    Train a small model, compress (prune+quantize), convert compression to MDL bits,
    and compare empirical generalization gap with Catoni-style PAC-Bayes using KL≈bits/ln2.
    """
    set_seed(0)
    train_ds, test_ds = get_mnist(n_train=10000, n_test=2000, translate_pixels=0)
    bs = 128
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2)

    if model_type == "mlp":
        model = MLP(in_dim=28*28, n_classes=10, hidden=512)
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
    print(f"[A] empirical CE gap = {gap:.4f}")

    # flatten parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rows = []
    for kr in keep_ratio_list:
        masks, kept = global_magnitude_prune(model, keep_ratio=kr)
        nz = kept.numel()
        for qb in quant_bits:
            qvals, mse = uniform_quantize(kept.clone(), qb)
            bits = mdl_code_length_bits(n_params=n_params, nz=nz, value_bits=qb, index_bits_exact=False)
            # PAC-Bayes bound: Δ ≈ sqrt((KL+ln(2√n/δ))/2(n-1)), KL≈bits/ln2
            n = len(train_ds)
            delta = 0.05
            KL = bits / math.log(2)   # convert bits -> nats
            bound = math.sqrt((KL + math.log(2*math.sqrt(n)/delta)) / (2*(n-1)))
            rows.append((kr, qb, nz, bits, bound, mse))
            print(f"[A] keep={kr:.2f}, q={qb}bits -> nz={nz}, bits={bits}, Δ≈{bound:.4f}, qMSE={mse:.2e}")

    # quick plot: bound vs keep ratio
    fig, ax = plt.subplots()
    for qb in quant_bits:
        xs = [r[0] for r in rows if r[1]==qb]
        ys = [r[4] for r in rows if r[1]==qb]
        ax.plot(xs, ys, marker="o", label=f"{qb} bits")
    ax.set_xlabel("keep ratio (global magnitude prune)")
    ax.set_ylabel("PAC-MDL bound Δ (approx)")
    ax.set_title("A) Compression-as-prior PAC-MDL bound")
    ax.legend()
    save_plot(fig, "critics_A_pacmdl")

# ------------- B) Algorithmic complexity sweep ------------------------------

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

def exp_B(device="cpu", complexities=(1,2,4,8,16), n_train=5000, n_test=2000, d=100, noise=0.0):
    set_seed(1)
    res = []
    for k in complexities:
        Xtr, ytr = gen_parity_dataset(n_train, d, k, noise=noise)
        Xte, yte = gen_parity_dataset(n_test,  d, k, noise=noise)
        tr_acc, te_acc = train_simple_mlp_classifier(Xtr, ytr, Xte, yte, device, width=512, epochs=15)
        res.append((k, tr_acc, te_acc))
        print(f"[B] k={k}: train_acc={tr_acc:.3f}, test_acc={te_acc:.3f}")
    # plot
    fig, ax = plt.subplots()
    ax.plot([r[0] for r in res], [1-r[2] for r in res], marker="o")
    ax.set_xlabel("parity complexity k (number of bits)")
    ax.set_ylabel("test error")
    ax.set_title("B) Data algorithmic complexity sweep")
    save_plot(fig, "critics_B_complexity")

# ------------- C) Soft vs Hard bias (augmentation vs conv) -------------------

def exp_C(device="cpu", ntrain_list=(200, 500, 1000, 5000), translate_pixels=3):
    """
    Compare:
      - MLP without weight sharing, but with translation augmentation (soft bias)
      - CNN with weight sharing, minimal/no augmentation (hard bias)
    Measure sample efficiency on MNIST.
    """
    set_seed(2)
    out = []
    for ntr in ntrain_list:
        # soft: MLP with strong translation augmentation
        train_ds_soft, test_ds = get_mnist(n_train=ntr, n_test=2000, translate_pixels=translate_pixels)
        train_ds_hard, _       = get_mnist(n_train=ntr, n_test=2000, translate_pixels=0)
        bs = 128
        tr_soft = DataLoader(train_ds_soft, batch_size=bs, shuffle=True)
        tr_hard = DataLoader(train_ds_hard, batch_size=bs, shuffle=True)
        te = DataLoader(test_ds, batch_size=bs, shuffle=False)

        mlp = MLPNoConv(in_shape=(1,28,28), n_classes=10, width=1024)
        cnn = SmallCNN(n_classes=10)

        tr_acc_s, te_acc_s = train_torch(mlp, tr_soft, te, device, epochs=6, lr=1e-3, wd=1e-4)
        tr_acc_h, te_acc_h = train_torch(cnn, tr_hard, te, device, epochs=6, lr=1e-3, wd=1e-4)

        out.append((ntr, te_acc_s, te_acc_h))
        print(f"[C] n={ntr}: soft(MLP+aug) test={te_acc_s:.3f} | hard(CNN) test={te_acc_h:.3f}")

    fig, ax = plt.subplots()
    ax.plot([o[0] for o in out], [o[1] for o in out], marker="o", label="soft: MLP+aug")
    ax.plot([o[0] for o in out], [o[2] for o in out], marker="x", label="hard: CNN")
    ax.set_xscale("log")
    ax.set_xlabel("train size")
    ax.set_ylabel("test accuracy")
    ax.set_title("C) Soft vs Hard inductive bias (sample efficiency)")
    ax.legend()
    save_plot(fig, "critics_C_soft_vs_hard")

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

def exp_D(device="cpu"):
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

def exp_E(device="cpu", n_train=2000, n_test=1000,itenum=80,simple=False,seed=4):
    """
    Small MLP on MNIST subset; compute approximate Hessian rows to estimate small eigenvalues.
    Correlate fraction of near-zero eigenvalues with generalization gap across checkpoints.
    """
    assert torchvision is not None, "torchvision required"
    set_seed(seed)
    train_ds, test_ds = get_mnist(n_train=n_train, n_test=n_test, translate_pixels=0)
    tr = DataLoader(train_ds, batch_size=128, shuffle=True)
    te = DataLoader(test_ds, batch_size=256, shuffle=False)
    model = MLP(in_dim=28*28, n_classes=10, hidden=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn= F.cross_entropy
    checkpoints = []
    for ep in range(6):
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
    save_plot(fig, "critics_E_hessian_deg")

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
        exp_E(device=device)
    else:
        device="cpu"
        print("device is ",device)        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=list("ABCDE"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--complexities", type=int, nargs="*", default=[1,2,4,8,16])
    parser.add_argument("--ntrain_list", type=int, nargs="*", default=[200,500,1000,5000])
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    if args.exp == "A":
        exp_A(device=device)
    elif args.exp == "B":
        exp_B(device=device, complexities=args.complexities)
    elif args.exp == "C":
        exp_C(device=device, ntrain_list=args.ntrain_list)
    elif args.exp == "D":
        exp_D(device=device)
    elif args.exp == "E":
        exp_E(device=device)
    else:
        main_all()

if __name__ == "__main__":
    main()

