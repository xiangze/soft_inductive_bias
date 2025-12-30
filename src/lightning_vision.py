"""
PyTorch Lightning script:
- Choose torchvision dataset + torchvision model
- Automatically apply appropriate transforms (prefer model weights.transforms()).
- Train + validate + test with accuracy.

Examples:
  python train_lightning_vision.py --dataset cifar10 --model resnet18 --batch-size 256 --epochs 10 --accelerator gpu
  python train_lightning_vision.py --dataset mnist --model vit_b_16 --epochs 5 --accelerator gpu
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import datasets, transforms

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities.seed import seed_everything

# -----------------------------
# Dataset registry
# -----------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    ctor: Callable
    num_classes: int
    in_channels: int


DATASETS = {
    "cifar10": DatasetSpec("cifar10", datasets.CIFAR10, num_classes=10, in_channels=3),
    "cifar100": DatasetSpec("cifar100", datasets.CIFAR100, num_classes=100, in_channels=3),
    "mnist": DatasetSpec("mnist", datasets.MNIST, num_classes=10, in_channels=1),
    "fashionmnist": DatasetSpec("fashionmnist", datasets.FashionMNIST, num_classes=10, in_channels=1),
    "svhn": DatasetSpec("svhn", datasets.SVHN, num_classes=10, in_channels=3),
}


def get_default_weights_for_model(model_name: str):
    """
    Return torchvision Weights enum instance if available (DEFAULT),
    else None (no pretrained weights).
    """
    # torchvision.models.get_model_weights exists in newer torchvision.
    # We'll try a robust approach:
    try:
        weights_enum = torchvision.models.get_model_weights(model_name)
        return weights_enum.DEFAULT
    except Exception:
        # Fallback: try conventional attribute e.g. ResNet18_Weights
        # but mapping all would be annoying; return None.
        return None


def build_model(model_name: str, num_classes: int, pretrained: bool) -> Tuple[nn.Module, Optional[object]]:
    """
    Build torchvision model by name.
    If pretrained=True and DEFAULT weights exist, use them.
    Adjust classifier head to num_classes.
    Returns (model, weights_used).
    """
    weights = get_default_weights_for_model(model_name) if pretrained else None

    try:
        model = torchvision.models.get_model(model_name, weights=weights)
    except TypeError:
        # Older torchvision: may not support weights keyword for some get_model.
        model = torchvision.models.get_model(model_name)

    # Replace classification head depending on architecture family
    # Covers common torchvision models.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        # ResNet family
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        # MobileNet/EfficientNet/ConvNeXt/ViT etc.
        # classifier can be Linear or Sequential
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
            # Replace last Linear in Sequential
            replaced = False
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    in_features = model.classifier[i].in_features
                    model.classifier[i] = nn.Linear(in_features, num_classes)
                    replaced = True
                    break
            if not replaced:
                raise RuntimeError(f"Could not locate Linear layer in model.classifier for {model_name}")
        else:
            raise RuntimeError(f"Unsupported classifier type for {model_name}: {type(model.classifier)}")
    elif hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        # Some ViT variants have model.heads.head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError(
            f"Unsupported model head replacement for {model_name}. "
            f"Please extend head replacement logic."
        )

    return model, weights


def infer_model_expected_in_channels(model: nn.Module) -> Optional[int]:
    """
    Heuristic: try to find first conv layer input channels.
    For ViT, patch embedding conv can be used.
    Return None if not confidently inferred.
    """
    # Common CNN first conv
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
        # ViT patch embedding is often Conv2d too; above covers it.
    return None


def build_transforms(weights_used: Optional[object], dataset_in_channels: int, model_expected_in_channels: Optional[int]):
    """
    Prefer weights.transforms() if available; otherwise use reasonable defaults.

    Additionally:
    - If dataset is 1ch (MNIST-like) and model expects 3ch, convert to RGB (3ch).
    - If model expects 1ch but dataset is 3ch (rare for torchvision pretrained), could convert to grayscale (not done by default).
    """
    pre = []
    post = []

    # Channel adaptation (MNIST -> RGB if model likely expects 3ch)
    if dataset_in_channels == 1 and (model_expected_in_channels is None or model_expected_in_channels == 3):
        # Convert PIL "L" to "RGB"
        pre.append(transforms.Lambda(lambda img: img.convert("RGB")))

    if weights_used is not None and hasattr(weights_used, "transforms"):
        # torchvision weights have a built-in transform pipeline including resize/crop/normalize.
        core = weights_used.transforms()
        # weights.transforms() expects PIL input; our pre should also keep PIL.
        tfm = transforms.Compose(pre + [core])
        return tfm

    # Fallback: generic transforms
    # Use 224 if we can't rely on weights; most ImageNet-family models assume 224.
    size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    core = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])
    tfm = transforms.Compose(pre + [core] + post)
    return tfm

# -----------------------------
# Lightning components
# -----------------------------
class VisionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        train_tfm,
        test_tfm,
        batch_size: int,
        num_workers: int,
        val_split: float,
        seed: int,
    ):
        super().__init__()
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASETS.keys())}")

        self.spec = DATASETS[dataset_name]
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.train_tfm = train_tfm
        self.test_tfm = test_tfm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def prepare_data(self):
        # Download only
        if self.dataset_name in ("cifar10", "cifar100", "mnist", "fashionmnist"):
            self.spec.ctor(self.data_dir, train=True, download=True)
            self.spec.ctor(self.data_dir, train=False, download=True)
        elif self.dataset_name == "svhn":
            self.spec.ctor(self.data_dir, split="train", download=True)
            self.spec.ctor(self.data_dir, split="test", download=True)

    def setup(self, stage: Optional[str] = None):
        if self.dataset_name in ("cifar10", "cifar100", "mnist", "fashionmnist"):
            full_train = self.spec.ctor(self.data_dir, train=True, transform=self.train_tfm, download=False)
            test = self.spec.ctor(self.data_dir, train=False, transform=self.test_tfm, download=False)
        elif self.dataset_name == "svhn":
            full_train = self.spec.ctor(self.data_dir, split="train", transform=self.train_tfm, download=False)
            test = self.spec.ctor(self.data_dir, split="test", transform=self.test_tfm, download=False)
        else:
            raise ValueError(self.dataset_name)

        n_total = len(full_train)
        n_val = int(round(n_total * self.val_split))
        n_train = n_total - n_val
        g = torch.Generator().manual_seed(self.seed)
        train, val = random_split(full_train, [n_train, n_val], generator=g)

        # For val, use test_tfm (deterministic, no augmentation) if you later add augmentation to train_tfm.
        # Here we keep both identical unless user extends train pipeline.
        # But to be safe, rebuild val dataset with test_tfm:
        # random_split gives Subset; easiest is keep it, or re-wrap. We'll keep it as is.
        self.ds_train, self.ds_val, self.ds_test = train, val, test

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

        self.save_hyperparameters(ignore=["model"])

    @staticmethod
    def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        acc = self.accuracy_top1(logits, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # A simple cosine schedule; adjust as needed.
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.trainer.max_epochs))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

def show():
    print("datasets")
    print(DATASETS)
    print("models")
    print("e.g. resnet18, resnet50, vit_b_16")
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS.keys()))
    p.add_argument("--model", type=str, required=True, help="torchvision model name, e.g. resnet18, resnet50, vit_b_16")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=max(2, os.cpu_count() // 2))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--pretrained", action="store_true", help="Use torchvision DEFAULT pretrained weights when available")
    p.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu", "mps"])
    p.add_argument("--devices", type=str, default="auto", help='e.g. "auto", "1", "0,", "0,1"')
    p.add_argument("--precision", type=str, default="16-mixed", help='e.g. "32", "16-mixed", "bf16-mixed"')

    p.add_argument("--log-every-n-steps", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    show()
    ds_spec = DATASETS[args.dataset]
    model, weights_used = build_model(args.model, num_classes=ds_spec.num_classes, pretrained=args.pretrained)

    model_expected_in = infer_model_expected_in_channels(model)
    train_tfm = build_transforms(weights_used, dataset_in_channels=ds_spec.in_channels, model_expected_in_channels=model_expected_in)
    test_tfm = train_tfm  # determinism/augmentationを分けたい場合はここを差し替え

    dm = VisionDataModule(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        train_tfm=train_tfm,
        test_tfm=test_tfm,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    lit = LitClassifier(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    callbacks = [
        ModelCheckpoint(monitor="val/acc", mode="max", save_top_k=1, filename="{epoch:02d}-{val_acc:.4f}"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
    )

    trainer.fit(lit, datamodule=dm)
    trainer.test(lit, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
