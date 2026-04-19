from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os, math, argparse, random
from typing import Iterable, List, Optional, Sequence, Tuple, Union

# Optional imports (will be used when available)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
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

# ----------------------------
# Utilities
# ----------------------------
def _pair(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    assert len(x) == 2
    return (int(x[0]), int(x[1]))

def _act(name: str) -> nn.Module:
    name = name.lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("tanh",):
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

def _norm_1d(n: int, kind: str) -> nn.Module:
    kind = kind.lower()
    if kind in ("none", ""):
        return nn.Identity()
    if kind in ("bn", "batchnorm", "batchnorm1d"):
        return nn.BatchNorm1d(n)
    if kind in ("ln", "layernorm"):
        return nn.LayerNorm(n)
    raise ValueError(f"Unknown norm kind: {kind}")

def _norm_2d(c: int, kind: str) -> nn.Module:
    kind = kind.lower()
    if kind in ("none", ""):
        return nn.Identity()
    if kind in ("bn", "batchnorm", "batchnorm2d"):
        return nn.BatchNorm2d(c)
    if kind in ("gn", "groupnorm"):
        # good default: 8 groups if divisible, else 4, else 1
        for g in (8, 4, 2, 1):
            if c % g == 0:
                return nn.GroupNorm(g, c)
        return nn.GroupNorm(1, c)
    raise ValueError(f"Unknown norm kind: {kind}")


def make_image_transform(image_size, translate_pixels=0, flatten=False):
    H, W = image_size
    aug = []
    if translate_pixels > 0:
        aug.append( T.RandomAffine(degrees=0, translate=(translate_pixels / W, translate_pixels / H) )   )
    aug.append(T.Resize((H, W)))
    aug.append(T.ToTensor())
    if flatten:
        aug.append(T.Lambda(lambda x: x.view(-1)))
    return T.Compose(aug)

# ----------------------------
# MLP classifier
# ----------------------------
class MLPClassifier(nn.Module):
    """
    Fully-connected MLP for image classification.
    - input_shape: (C, H, W) or (H, W, C) depending on channels_first
    - depth: number of hidden layers
    - width: hidden dimension (int) or list of hidden dims (len=depth)
    """
    def __init__(
        self,
        num_classes: int,
        image_size: Union[int, Tuple[int, int]],
        in_channels: int = 3,
        channels_first: bool = True,
        depth: int = 3,
        width: Union[int, Sequence[int]] = 512,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm: str = "ln",  # "ln", "bn", "none"
    ):
        super().__init__()
        H, W = _pair(image_size)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.image_size = (H, W)
        self.channels_first = bool(channels_first)

        in_dim = self.in_channels * H * W

        if isinstance(width, int):
            hidden_dims = [int(width)] * int(depth)
        else:
            hidden_dims = [int(x) for x in width]
            if len(hidden_dims) != int(depth):
                raise ValueError(f"len(width) must equal depth. Got len(width)={len(hidden_dims)}, depth={depth}")

        layers: List[nn.Module] = []
        prev = in_dim
        act = _act(activation)

        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, hdim))
            layers.append(_norm_1d(hdim, norm))
            layers.append(act if i == 0 else _act(activation))  # keep separate module instances
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = hdim

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) if channels_first else (B,H,W,C)
        if not self.channels_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x.flatten(1)  # (B, C*H*W)
        x = self.backbone(x)
        logits = self.head(x)
        return logits

# ----------------------------
# CNN classifier (stacked conv blocks)
# ----------------------------
class CNNClassifier(nn.Module):
    """
    Simple configurable multi-layer CNN for image classification.

    - depth: number of conv blocks.
    - channels: base channel count (int) -> grows by channel_mult each stage
               or explicit list of out_channels per block (len=depth).
    - downsample_every: apply stride-2 conv every N blocks (0/None disables)
    - head: global avg pool -> linear

    Works for any input H,W thanks to AdaptiveAvgPool2d.
    """
    def __init__(
        self,
        num_classes: int,
        image_size: Union[int, Tuple[int, int]],  # kept for API symmetry; not strictly required
        in_channels: int = 3,
        depth: int = 4,
        channels: Union[int, Sequence[int]] = 64,
        channel_mult: int = 2,
        kernel_size: int = 3,
        activation: str = "relu",
        norm: str = "bn",  # "bn", "gn", "none"
        dropout: float = 0.0,  # spatial dropout after activations
        downsample_every: Optional[int] = 2,  # e.g., 2 -> every 2 blocks do stride-2
        final_pool: str = "avg",  # "avg" only (kept extensible)
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.image_size = _pair(image_size)

        if isinstance(channels, int):
            out_channels_list = []
            c = int(channels)
            for i in range(int(depth)):
                out_channels_list.append(c)
                # grow after each block
                c = c * int(channel_mult)
        else:
            out_channels_list = [int(c) for c in channels]
            if len(out_channels_list) != int(depth):
                raise ValueError(f"len(channels) must equal depth. Got len(channels)={len(out_channels_list)}, depth={depth}")

        k = int(kernel_size)
        pad = k // 2
        act_name = activation

        blocks: List[nn.Module] = []
        c_in = self.in_channels

        for i, c_out in enumerate(out_channels_list):
            stride = 1
            if downsample_every and downsample_every > 0 and (i > 0) and (i % int(downsample_every) == 0):
                stride = 2

            blocks.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=pad, bias=(norm in ("none", "", None))))
            blocks.append(_norm_2d(c_out, norm))
            blocks.append(_act(act_name))
            if dropout > 0:
                blocks.append(nn.Dropout2d(p=float(dropout)))
            c_in = c_out

        self.features = nn.Sequential(*blocks)

        if final_pool.lower() != "avg":
            raise ValueError("final_pool currently supports only 'avg'")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c_in, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        x = self.features(x)
        x = self.pool(x).flatten(1)  # (B, C)
        logits = self.head(x)
        return logits

# ----------------------------
# Resnet18 classifier (stacked conv blocks)
# ----------------------------
class ResNet18Classifier(nn.Module):
    """
    torchvision.models.resnet18 を
    MLPClassifier / CNNClassifier と同じAPIで使えるようにしたラッパー
    """
    def __init__(
        self,
        num_classes: int,
        image_size,              # unused (for API compatibility)
        in_channels: int = 3,
        channels_first: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.channels_first = channels_first
        self.in_channels = in_channels

        # load base model
        self.backbone = torch.model.resnet18(pretrained=pretrained)

        # 入力チャンネルが3以外の場合は最初のConvを書き換え
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        # classifier head を差し替え
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,W,C) → (B,C,H,W)
        if not self.channels_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        return self.backbone(x)
# ------------- datasets ------------------------------------------------------
def subsample(train, test):
    train_idx = RNG.choice(len(train["data"]), train["num"], replace=False)
    test_idx  = RNG.choice(len(test["data"]), test["num"], replace=False)
    train_ds = Subset(train["data"], train_idx)
    test_ds  = Subset(test["data"],  test_idx)
    return train_ds, test_ds

def get_mnist(n_train=10000, n_test=2000, image_size=(32,32),translate_pixels=0, flatten=False):
    tf_train = make_image_transform(image_size, translate_pixels, flatten)
    tf_test  = make_image_transform(image_size, 0, flatten)
    train_ds = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=tf_train)
    test_ds  = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=tf_test)
    return subsample({"data":train_ds,"num":n_train},{"data":test_ds,"num":n_test})

def get_cifar10(n_train=10000, n_test=2000, image_size=(32,32),translate_pixels=0, flatten=False):
    tf_train = make_image_transform(image_size, translate_pixels, flatten)
    tf_test  = make_image_transform(image_size, 0, flatten)
    train_ds = torchvision.datasets.CIFAR10( root="data", train=True, download=True, transform=tf_train,)
    test_ds = torchvision.datasets.CIFAR10( root="data", train=False, download=True, transform=tf_test, )
    return subsample({"data":train_ds,"num":n_train},{"data":test_ds,"num":n_test})

def get_cifar100(n_train=10000, n_test=2000, image_size=(32,32), translate_pixels=0, flatten=False):
    tf_train = make_image_transform(image_size, translate_pixels, flatten)
    tf_test  = make_image_transform(image_size, 0, flatten)
    train_ds = torchvision.datasets.CIFAR100(root="data",  train=True,  download=True, transform=tf_train)
    test_ds  = torchvision.datasets.CIFAR100(root="data",  train=False, download=True, transform=tf_test)
    return subsample({"data":train_ds,"num":n_train},{"data":test_ds,"num":n_test})

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
        

# ----------------------------
#  NN test
# ----------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W)

    mlp = MLPClassifier(num_classes=10, image_size=(H, W), in_channels=C, depth=3, width=256, activation="gelu", dropout=0.1, norm="ln")
    cnn = CNNClassifier(num_classes=10, image_size=(H, W), in_channels=C, depth=4, channels=32, channel_mult=2, activation="relu", norm="bn", dropout=0.1, downsample_every=2)
    print("MLP:", mlp(x).shape)  # (B,10)
    print("CNN:", cnn(x).shape)  # (B,10)

    