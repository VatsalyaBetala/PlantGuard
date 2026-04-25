"""
ViT training script (timm + PyTorch) that saves artifacts in the format your app expects:

artifacts/<model_name>/
  model.pt       (checkpoint dict with model_state + metadata)
  labels.json    (class names in the exact order used during training)
  config.json    (training config snapshot)

Usage (local):
  python train/train_vit.py \
    --train_dir /path/to/train --val_dir /path/to/val \
    --model_name vit_base_patch16_224 --timm_name vit_base_patch16_224 \
    --artifacts_dir artifacts --epochs 20 --batch_size 64

Usage (HPC):
  put this in your repo, push, clone on HPC, then run via PBS/interactive.

Notes:
- Uses ImageFolder (train_dir/val_dir must be:
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
  )
- If you pass --labels_json, the script ENFORCES that exact class ordering.
  (Highly recommended for avoiding label mismatch in your app/paper comparisons.)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import timm
from timm.data import create_transform
from timm.utils import accuracy


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_labels(path: str) -> List[str]:
    labels = json.loads(Path(path).read_text())
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("labels_json must be a JSON array of strings.")
    return labels


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


# -------------------------
# Dataset / Label order
# -------------------------
def build_datasets(
    train_dir: str,
    val_dir: str,
    image_size: int,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    # timm-style train augmentations (works well for ViTs)
    train_tfms = create_transform(
        input_size=image_size,
        is_training=True,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
    )

    # standard val transforms
    val_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    return train_ds, val_ds


def enforce_label_order(
    train_ds: datasets.ImageFolder,
    val_ds: datasets.ImageFolder,
    labels: List[str],
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """
    Force dataset targets to follow the ordering in labels[].
    Requires folder names == labels.
    """
    train_classes = set(train_ds.classes)
    val_classes = set(val_ds.classes)
    label_set = set(labels)

    missing_train = label_set - train_classes
    extra_train = train_classes - label_set
    missing_val = label_set - val_classes
    extra_val = val_classes - label_set

    if missing_train or extra_train:
        raise ValueError(
            f"Train folder mismatch.\nMissing: {sorted(missing_train)}\nExtra: {sorted(extra_train)}"
        )
    if missing_val or extra_val:
        raise ValueError(
            f"Val folder mismatch.\nMissing: {sorted(missing_val)}\nExtra: {sorted(extra_val)}"
        )

    class_to_idx = {name: i for i, name in enumerate(labels)}

    # Overwrite mapping + rebuild samples/targets
    def remap(ds: datasets.ImageFolder) -> None:
        # ds.samples contains (path, old_target) where old_target indexes ds.classes
        old_classes = ds.classes
        ds.class_to_idx = class_to_idx
        ds.samples = [(p, class_to_idx[old_classes[y]]) for p, y in ds.samples]
        ds.targets = [y for _, y in ds.samples]
        ds.classes = labels

    remap(train_ds)
    remap(val_ds)
    return train_ds, val_ds


def build_loaders(
    train_ds,
    val_ds,
    batch_size: int,
    workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    losses, accs = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = crit(logits, y)
        acc1 = accuracy(logits, y, topk=(1,))[0].item()
        losses.append(loss.item())
        accs.append(acc1)

    return float(np.mean(losses)), float(np.mean(accs))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    label_smoothing: float,
    grad_clip_norm: float,
) -> Tuple[float, float]:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if label_smoothing and label_smoothing > 0:
        crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        crit = nn.CrossEntropyLoss()

    losses, accs = [], []
    pbar = tqdm(loader, desc="train", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = crit(logits, y)

        scaler.scale(loss).backward()

        if grad_clip_norm and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        acc1 = accuracy(logits, y, topk=(1,))[0].item()
        losses.append(loss.item())
        accs.append(acc1)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc1=f"{acc1:.2f}")

    return float(np.mean(losses)), float(np.mean(accs))


def save_artifacts(
    artifacts_dir: Path,
    model_name: str,
    model: nn.Module,
    labels: List[str],
    cfg: Dict,
    best_acc1: float,
    best_epoch: int,
) -> None:
    out_dir = artifacts_dir / model_name
    ensure_dir(out_dir)

    # labels.json (your app expects this)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    # config snapshot for reproducibility
    save_json(out_dir / "config.json", cfg)

    # model.pt checkpoint dict (safe and future-proof)
    ckpt = {
        "model_state": model.state_dict(),
        "timm_name": cfg["timm_name"],
        "pretrained": cfg["pretrained"],
        "num_classes": len(labels),
        "image_size": cfg["image_size"],
        "labels": labels,
        "best_acc1": best_acc1,
        "best_epoch": best_epoch,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(ckpt, out_dir / "model.pt")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, type=str)
    ap.add_argument("--val_dir", required=True, type=str)

    ap.add_argument("--model_name", default="vit_base_patch16_224", type=str,
                    help="Artifact folder name under artifacts/.")
    ap.add_argument("--timm_name", default="vit_base_patch16_224", type=str,
                    help="timm model id, e.g. vit_base_patch16_224, deit_small_patch16_224, etc.")
    ap.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights.")
    ap.add_argument("--no_pretrained", action="store_true", help="Force pretrained=False (overrides --pretrained).")

    ap.add_argument("--artifacts_dir", default=os.getenv("MODEL_ARTIFACTS_DIR", "artifacts"), type=str)

    ap.add_argument("--image_size", default=224, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--weight_decay", default=0.05, type=float)
    ap.add_argument("--label_smoothing", default=0.1, type=float)
    ap.add_argument("--grad_clip_norm", default=1.0, type=float)

    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--seed", default=42, type=int)

    ap.add_argument("--amp", action="store_true", help="Use mixed precision (recommended on GPU).")
    ap.add_argument("--labels_json", default="", type=str,
                    help="Optional: path to a JSON array of class names. Enforces class ordering.")

    args = ap.parse_args()

    # pretrained logic
    pretrained = bool(args.pretrained)
    if args.no_pretrained:
        pretrained = False

    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    ensure_dir(artifacts_dir)

    # Build datasets
    train_ds, val_ds = build_datasets(args.train_dir, args.val_dir, args.image_size)

    # If labels_json provided, enforce explicit label ordering
    if args.labels_json.strip():
        labels = load_labels(args.labels_json)
        train_ds, val_ds = enforce_label_order(train_ds, val_ds, labels)
        final_labels = labels
    else:
        # Use ImageFolder ordering (risky if you need fixed label order across runs)
        final_labels = train_ds.classes

    train_loader, val_loader = build_loaders(train_ds, val_ds, args.batch_size, args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} timm_name={args.timm_name} pretrained={pretrained}")
    print(f"[INFO] classes={len(final_labels)} image_size={args.image_size}")
    print(f"[INFO] artifacts_dir={artifacts_dir} model_name={args.model_name}")

    # Create model
    model = timm.create_model(
        args.timm_name,
        pretrained=pretrained,
        num_classes=len(final_labels),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    cfg_snapshot = {
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "model_name": args.model_name,
        "timm_name": args.timm_name,
        "pretrained": pretrained,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "grad_clip_norm": args.grad_clip_norm,
        "workers": args.workers,
        "seed": args.seed,
        "amp": bool(args.amp),
        "labels_json": args.labels_json.strip(),
    }

    best_acc1 = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp=bool(args.amp),
            label_smoothing=args.label_smoothing,
            grad_clip_norm=args.grad_clip_norm,
        )
        va_loss, va_acc1 = evaluate(model, val_loader, device=device, amp=bool(args.amp))

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc1 {tr_acc1:.2f} | "
            f"val loss {va_loss:.4f} acc1 {va_acc1:.2f}"
        )

        if va_acc1 > best_acc1:
            best_acc1 = va_acc1
            best_epoch = epoch
            save_artifacts(
                artifacts_dir=artifacts_dir,
                model_name=args.model_name,
                model=model,
                labels=final_labels,
                cfg=cfg_snapshot,
                best_acc1=best_acc1,
                best_epoch=best_epoch,
            )

    print(f"[DONE] best val acc1={best_acc1:.2f} at epoch={best_epoch}")
    print(f"[DONE] artifacts saved to: {artifacts_dir / args.model_name}")


if __name__ == "__main__":
    main()
