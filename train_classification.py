from __future__ import annotations

import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from resganet.datasets import ISIC2018ClassificationDataset
from resganet.model import ResGANet
from resganet.metrics import classification_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.append(outputs.detach())
        all_targets.append(labels.detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = classification_metrics(all_preds, all_targets)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.append(outputs.detach())
        all_targets.append(labels.detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = classification_metrics(all_preds, all_targets)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ResGANet for Classification")
    parser.add_argument("--data_root", type=str, required=True, help="ISIC2018 数据集根目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save", type=str, default="./checkpoints/resganet_classification.pth")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # 自动选择最佳设备：优先 MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"使用设备: mps (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用设备: cuda (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print(f"使用设备: cpu")

    # 数据加载（MPS 不支持 pin_memory）
    use_pin_memory = device.type == "cuda"
    train_dataset = ISIC2018ClassificationDataset(args.data_root, split="train", target_size=args.img_size)
    val_dataset = ISIC2018ClassificationDataset(args.data_root, split="val", target_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin_memory)

    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # 模型
    model = ResGANet(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"prec={val_metrics['precision']:.4f} rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "accuracy": best_acc,
                    "metrics": val_metrics,
                    "epoch": epoch + 1,
                },
                args.save,
            )
            print(f"✓ 保存最佳模型 (Acc: {best_acc:.4f})")

    print(f"\n训练完成！最佳准确率: {best_acc:.4f}，模型保存至 {args.save}")


if __name__ == "__main__":
    main()

