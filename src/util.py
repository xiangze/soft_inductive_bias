import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os, math, argparse, random

# Optional imports (will be used when available)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from pathlib import Path
RNG = np.random.default_rng(42)

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

def corrupt_labels(y: torch.Tensor, noise: float, n_classes: int):
    """Flip labels uniformly at rate 'noise'. y: LongTensor (N,) in [0, n_classes-1]."""
    if noise <= 0: return y.clone()
    y = y.clone()
    N = y.numel()
    m = int(round(noise * N))
    idx = torch.from_numpy(RNG.choice(N, m, replace=False))
    noise_vals = torch.from_numpy(RNG.integers(0, n_classes, size=m)).long()
    # avoid trivial "same label" by resampling if equal
    same = noise_vals == y[idx]
    while same.any():
        noise_vals[same] = torch.from_numpy(RNG.integers(0, n_classes, size=int(same.sum()))).long()
        same = noise_vals == y[idx]
    y[idx] = noise_vals
    return y

def build_loader_from_subset(subset, batch_size, noise=0.0, n_classes=10, shuffle=True):
    xs, ys = [], []
    for i in range(len(subset)):
        x, y = subset[i]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    y = corrupt_labels(y, noise=noise, n_classes=n_classes)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)

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

def build_loader_from_subset(subset, batch_size, noise=0.0, n_classes=10, shuffle=True):
    xs, ys = [], []
    for i in range(len(subset)):
        x, y = subset[i]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    y = corrupt_labels(y, noise=noise, n_classes=n_classes)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)

# --------------------- training ---------------------

def train_torch(model, train_loader, test_loader, device, epochs=10, lr=1e-3, wd=0.0):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    tr_acc = accuracy_torch(model, train_loader, device)
    te_acc = accuracy_torch(model, test_loader, device)
    return tr_acc, te_acc

# ------------- simple models -------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim=784, n_classes=10, hidden=512,depth=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
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

def get_cifar10(n_train=10000, n_test=2000, translate_pixels=0, flatten=False):
    assert torchvision is not None, "torchvision is required for CIFAR"    


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

# ------------- plots -------------------------------------------------
def save_plot(fig, name: str):
    ensure_dir("figs")
    fig.savefig(Path("figs") / f"{name}.png", dpi=180, bbox_inches="tight")

def plots(xs,ys,title,xlabel,ylabel,labels,outputfilename):
    fig, ax = plt.subplots()
    for i,xy in enumerate(zip(xs,ys)):
        x,y=xy
        ax.plot(x, y, marker="o", label=labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_plot(fig, outputfilename)


def plot2(files=["result/critics_A_pacmdl_cnn.log","result/critics_A_pacmdl_mlp.log"]):
    #Δ vs MSE
    for f in files:
        with open(f) as fp:
            ls=fp.readlines()
        #CEgap=l[0].split(",")[0].split(" ")[-1]
        #accuracy=l[1].split()[2:]
        ls=[l.split(",") for l in ls]
        delta=[float(l[0]) for l in ls]
        mse=[float(l[1]) for l in ls]
        plt.plot(delta,mse,marker="o")
        plt.title("MDLΔ vs MSE")
        plt.xlabel("PAC-MDL bound Δ is(approx)")
        plt.ylabel("mean square error")
        plt.savefig(f.replace("log","png"))

def plotA(files=["result/critics_A_pacmdl_cnn.log","result/critics_A_pacmdl_mlp.log"]):
    for f in files:
        with open(f) as fp:
            ls=fp.readlines()
        CEgap=ls[0].split(",")[0].split(" ")[-1]
        accuracy=ls[1].split()[2:]
        ls=[l.replace("≈","=").split(",") for l in ls[2:]]
        ds=[{a.split("=")[0]:float(a.split("=")[-1]) for a in l} for l in ls]
        df=pd.DataFrame.from_records(ds)        
        #print(df)
        pg = sns.pairplot(df)
        pg.savefig(f'{f.split(".")[0]}.png')
        

    