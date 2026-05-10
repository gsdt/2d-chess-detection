"""Train a piece classifier (13 classes) on synthetic per-square crops."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from tqdm import tqdm


# Folder names produced by gen_classifier (case-safe). Sorted alphabetically by
# torchvision.datasets.ImageFolder, this is the index → class mapping.
CLASS_DIRS = sorted(["empty"] + [f"{c}{p}" for c in "wb" for p in "PNBRQK"])
DIRNAME_TO_LABEL: dict[str, str] = {"empty": "empty"}
for d in CLASS_DIRS:
    if d == "empty":
        continue
    color, piece = d[0], d[1]
    DIRNAME_TO_LABEL[d] = piece if color == "w" else piece.lower()


def build_model(num_classes: int = 13, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    return model


def make_loaders(root: Path, img_size: int, batch_size: int, workers: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=2, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ds, train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds, train_loader, val_loader = make_loaders(args.data, args.img_size, args.batch_size, args.workers)
    print(f"train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}  classes: {train_ds.classes}")

    model = build_model(num_classes=len(train_ds.classes), pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ep {ep+1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": train_ds.classes,
                    "img_size": args.img_size,
                    "val_acc": val_acc,
                },
                args.out,
            )
            print(f"  ✓ saved {args.out} (acc {val_acc:.4f})")

    print(f"best val_acc: {best_acc:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/classifier"))
    ap.add_argument("--out", type=Path, default=Path("data/checkpoints/classifier.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true", default=True)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
