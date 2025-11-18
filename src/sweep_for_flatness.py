# sweep_sgd_temp_flatness.py
import argparse, csv, os, random, time, math
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import lanczos
# -----------------------
# Utils
# -----------------------

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Trueだと遅い
    torch.backends.cudnn.benchmark = True

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def to_device(batch, device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

# -----------------------
# Label noise wrapper
# -----------------------
class FixedLabelNoise(Dataset):
    """
    訓練セットのラベルを一定比率で一度だけ乱転（固定）させるラッパー。
    """
    def __init__(self, base: Dataset, num_classes: int, noise_ratio: float, seed: int=0):
        self.base = base
        self.num_classes = num_classes
        self.noise_ratio = noise_ratio
        self.indices = list(range(len(self.base)))
        rng = random.Random(seed)
        self.flip_mask = [False]*len(self.base)
        if noise_ratio > 0:
            num_flip = int(len(self.base)*noise_ratio)
            flip_idx = rng.sample(self.indices, num_flip)
            for i in flip_idx: self.flip_mask[i] = True
        # pre-generate random wrong labels for flipped indices
        self.rand_labels = {}
        for i, f in enumerate(self.flip_mask):
            if f:
                orig = self.base[i][1]
                # choose a label != orig
                cand = list(range(num_classes)); cand.remove(orig)
                self.rand_labels[i] = rng.choice(cand)

    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.flip_mask[idx]:
            y = self.rand_labels[idx]
        return x, y

# -----------------------
# Data
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_cifar10(root: str, img_size: int=224, batch_size: int=128, num_workers: int=4,
                noise_ratio: float=0.0, seed:int=0) -> Tuple[DataLoader, DataLoader, int]:
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_set = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    if noise_ratio>0:
        train_set = FixedLabelNoise(train_set, 10, noise_ratio, seed=seed)
    test_set  = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10

# -----------------------
# Models
# -----------------------
def create_model(name: str, num_classes: int, pretrained: bool=True) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {name}")
    return m

# -----------------------
# Train / Eval
# -----------------------
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = to_device((x,y), device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += loss.item()*x.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total += x.size(0)
    err = 1.0 - correct/total
    return err, loss_sum/total

def train_one(model, train_loader, test_loader, device, epochs, lr, weight_decay,
              use_sgdr: bool, use_swa: bool, swa_start_frac: float=0.75):
    model.to(device)
    ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)

    if use_sgdr:
        # SGDR: Warm Restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(10, epochs//4), T_mult=2)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if use_swa:
        swa_model = AveragedModel(model)
        swa_start = int(epochs * swa_start_frac)
        swa_scheduler = SWALR(optimizer, anneal_strategy="cos", anneal_epochs=max(1, epochs//10), swa_lr=lr*0.5)
    else:
        swa_model = None
        swa_start = math.inf
        swa_scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = to_device((x,y), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Scheduler update
        if use_sgdr:
            scheduler.step(ep + 1)  # warm restartsはepochを渡すのがわかりやすい
        else:
            scheduler.step()

        # SWA update
        if use_swa and ep >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        # print(f"Epoch {ep+1}/{epochs}, lr {[g['lr'] for g in optimizer.param_groups][0]:.4e}")

    # BN update for SWA model
    if use_swa:
        update_bn(train_loader, swa_model, device=device)
        return swa_model
    else:
        return model

# -----------------------
# Flatness metrics
# -----------------------
@torch.no_grad()
def local_sharpness(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor],
                    loss_fn: nn.Module, device: str, rho: float=1e-3) -> float:
    """
    1ステップのシャープネス近似：g = grad L, 方向 g に ρ だけ動かしたときの損失増分。
    """
    model.eval()
    x, y = to_device(batch, device)
    # need grad -> temporarily enable
    for p in model.parameters():
        p.requires_grad_(True)
    logits = model(x)
    loss = loss_fn(logits, y)
    grad = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=False)
    # flatten grad and compute norm
    flat_g = torch.cat([g.detach().flatten() for g in grad])
    gnorm = flat_g.norm() + 1e-12
    # apply perturbation: w' = w + rho * g/||g||
    idx = 0
    with torch.enable_grad():
        for p, g in zip(model.parameters(), grad):
            sz = p.numel()
            delta = (rho/gnorm) * g.detach()
            p.add_(delta)
        logits2 = model(x)
        loss2 = loss_fn(logits2, y)
        # revert
        for p, g in zip(model.parameters(), grad):
            delta = (rho/gnorm) * g.detach()
            p.sub_(delta)
    return (loss2 - loss).item()

def hessian_largest_eig(model: nn.Module, loader: DataLoader, device: str,
                        max_iters: int=10, samples: int=1, params_ratio: float=1.0,method="hpv") -> float:
    """
    lanzos法(パワー法)Hessian最大固有値を近似。CrossEntropyを使用。
    ・samples バッチ数だけ平均化
    ・params_ratio < 1.0 の場合はパラメータのサブサンプリング（最後の層のみ等）を簡易実装
    """
    ce = nn.CrossEntropyLoss()
    model.eval()

    # Collect parameter tensors we differentiate through
    params = [p for p in model.parameters() if p.requires_grad]
    if params_ratio < 1.0:
        # keep only a suffix of parameters totalling given ratio of elements
        total = sum(p.numel() for p in params)
        target = int(total * params_ratio)
        kept = []
        acc = 0
        for p in reversed(params):
            kept.append(p)
            acc += p.numel()
            if acc >= target: break
        params = list(reversed(kept))

    # Initialize v (concatenate shape)
    print("Initialize v (concatenate shape)")
    shapes = [p.shape for p in params]
    v = torch.randn(sum(p.numel() for p in params), device=device)
    v = v / (v.norm() + 1e-12)

    def _flatten(grads: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.reshape(-1) for g in grads]).to(device)

    # iterator for limited number of batches
    it = iter(loader)
    used_batches = 0
    eig_estimates = []
    while used_batches < samples:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = to_device((x,y), device)
        used_batches += 1
        
        print(f"batch num{used_batches}")
        print(f"method {method}")
        # power-iteration
        if(method=="power"):
            v_vec = v.clone()
            for _ in range(max_iters):
                # compute Hv
                # 1) grad = ∇L
                logits = model(x)
                loss  = ce(logits, y)
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

                # 2) g^T v  -> scalar
                flat_g = _flatten(grads)
                gv = (flat_g * v_vec).sum()

                # 3) ∇(g^T v) = H v
                Hv = torch.autograd.grad(gv, params, retain_graph=False)
                flat_Hv = _flatten([h.detach() for h in Hv])

                # Rayleigh quotient
                rq = (v_vec * flat_Hv).sum() / (v_vec.norm()**2 + 1e-12)

                # normalize
                v_vec = flat_Hv / (flat_Hv.norm() + 1e-12)

            eig_estimates.append(rq.item())
        else: #hpv
            logits = model(x)
            loss  = ce(logits, y)
            _,vals,_= lanczos._get_eigenvecs(device,model,loss,loader,topk=50,m=max_iters)
            eig_estimates=vals

        return sum(eig_estimates)/len(eig_estimates)

# -----------------------
# Sweep
# -----------------------
def run_condition(args, model_name, batch_size, lr, noise_ratio, use_swa, use_sgdr, device, writer):
    # Data
    train_loader, test_loader, num_classes = get_cifar10(args.data_root, args.img_size,
                                                         batch_size, args.num_workers,
                                                         noise_ratio, seed=args.seed)

    # Model
    model = create_model(model_name, num_classes, pretrained=not args.no_pretrain)

    # Train
    model = train_one(model, train_loader, test_loader, device, args.epochs, lr,
                      args.weight_decay, bool(use_sgdr), bool(use_swa), args.swa_start_frac)

    # Eval
    test_err, test_loss = evaluate(model, test_loader, device)

    # Flatness
    # 1) Hessian spectral norm (approx)
    lam_max = hessian_largest_eig(model, test_loader, device,
                                  max_iters=args.hess_power_iters,
                                  samples=args.hess_samples,
                                  params_ratio=args.hess_params_ratio)

    # 2) local sharpness (delta loss within radius)
    # use a single mini-batch from test loader
    test_iter = iter(test_loader)
    batch_for_sharp = next(test_iter)
    sharp = local_sharpness(model, batch_for_sharp, nn.CrossEntropyLoss(), device, rho=args.sharpness_radius)

    # Log
    row = {
        "timestamp": now(),
        "model": model_name,
        "batch_size": batch_size,
        "lr": lr,
        "noise_ratio": noise_ratio,
        "use_swa": use_swa,
        "use_sgdr": use_sgdr,
        "epochs": args.epochs,
        "test_error": test_err,
        "test_loss": test_loss,
        "hess_lambda_max": lam_max,
        "local_sharpness_delta": sharp,
        "seed": args.seed
    }
    writer.writerow(row)
    print(f"[{now()}] {row}")

def main():
    p = argparse.ArgumentParser(description="Sweep: (batch_size, lr) × noise × (SWA/SGDR) with flatness metrics")
    p.add_argument("--models", nargs="+", default=["resnet18", "resnet50", "mobilenet_v2"],
                   help="Model names: resnet18 resnet50 mobilenet_v2 efficientnet_b0")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 256])
    p.add_argument("--lrs", nargs="+", type=float, default=[0.1, 0.025])
    p.add_argument("--noise-ratios", nargs="+", type=float, default=[0.0, 0.2])
    p.add_argument("--use-swa", nargs="+", type=int, default=[0, 1])
    p.add_argument("--use-sgdr", nargs="+", type=int, default=[0, 1])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--swa-start-frac", type=float, default=0.75)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no-pretrain", action="store_true", help="Do not load ImageNet pretrained weights")
    # Hessian/Sharpness options
    p.add_argument("--hess-power-iters", type=int, default=8)
    p.add_argument("--hess-samples", type=int, default=2, help="num test mini-batches for Hessian estimate")
    p.add_argument("--hess-params-ratio", type=float, default=1.0, help="0<r<=1: use last r fraction of parameters")
    p.add_argument("--sharpness-radius", type=float, default=1e-3)
    # Output
    p.add_argument("--out-csv", type=str, default="sweep_results.csv")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    new_file = not os.path.exists(args.out_csv)

    with open(args.out_csv, "a", newline="") as f:
        fieldnames = ["timestamp","model","batch_size","lr","noise_ratio","use_swa","use_sgdr","epochs",
                      "test_error","test_loss","hess_lambda_max","local_sharpness_delta","seed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for m in args.models:
            for bs in args.batch_sizes:
                for lr in args.lrs:
                    for nz in args.noise_ratios:
                        for swa in args.use_swa:
                            for sgdr in args.use_sgdr:
                                run_condition(args, m, bs, lr, nz, swa, sgdr, device, writer)

if __name__ == "__main__":
    main()
