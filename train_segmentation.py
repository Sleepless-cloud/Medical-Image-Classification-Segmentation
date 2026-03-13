from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from resganet.datasets import ISIC2018SegmentationDataset
from resganet.segmentation import ResGANetSegmentation
from resganet.metrics import dice_coefficient, iou_score, pixel_accuracy


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # 指标计算
        preds = torch.sigmoid(outputs)
        running_loss += loss.item() * images.size(0)
        running_dice += dice_coefficient(preds, masks) * images.size(0)
        running_iou += iou_score(preds, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_dice / n, running_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, masks)

        preds = torch.sigmoid(outputs)
        running_loss += loss.item() * images.size(0)
        running_dice += dice_coefficient(preds, masks) * images.size(0)
        running_iou += iou_score(preds, masks) * images.size(0)
        running_acc += pixel_accuracy(preds, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_dice / n, running_iou / n, running_acc / n


def main():
    parser = argparse.ArgumentParser(description="Train ResGANet for Segmentation")
    parser.add_argument("--data_root", type=str, required=True, help="ISIC2018 数据集根目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save", type=str, default="./checkpoints/resganet_segmentation.pth")
    parser.add_argument("--img_size", type=int, default=256)
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
    train_dataset = ISIC2018SegmentationDataset(args.data_root, split="train", target_size=args.img_size)
    val_dataset = ISIC2018SegmentationDataset(args.data_root, split="val", target_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=use_pin_memory)

    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # 模型
    model = ResGANetSegmentation(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_dice = 0.0
    for epoch in range(args.epochs):
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_dice)

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} dice={train_dice:.4f} iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f} acc={val_acc:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "dice": best_dice,
                    "iou": val_iou,
                    "epoch": epoch + 1,
                },
                args.save,
            )
            print(f"✓ 保存最佳模型 (Dice: {best_dice:.4f})")

    print(f"\n训练完成！最佳 Dice: {best_dice:.4f}，模型保存至 {args.save}")


if __name__ == "__main__":
    main()

