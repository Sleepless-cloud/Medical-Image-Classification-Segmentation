from __future__ import annotations

import torch
from torch import Tensor


def dice_coefficient(pred: Tensor, target: Tensor, smooth: float = 1e-6) -> float:
    """计算 Dice 系数（F1 for segmentation）。
    
    Args:
        pred: 预测分割图 [B, 1, H, W] 或 [B, H, W]，值域 [0, 1]
        target: 真实标签 [B, 1, H, W] 或 [B, H, W]，值域 {0, 1}
        smooth: 平滑项避免除零
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def iou_score(pred: Tensor, target: Tensor, smooth: float = 1e-6) -> float:
    """计算 IoU（Jaccard Index）。"""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def pixel_accuracy(pred: Tensor, target: Tensor) -> float:
    """像素准确率。"""
    pred = (pred > 0.5).float()
    target = target.float()
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def classification_metrics(pred: Tensor, target: Tensor) -> dict:
    """分类任务指标：Accuracy, Precision, Recall, F1。
    
    Args:
        pred: [B, num_classes] logits
        target: [B] 类别标签
    """
    pred_labels = pred.argmax(dim=1)
    correct = (pred_labels == target).sum().item()
    total = target.size(0)
    accuracy = correct / total

    # 简化版：宏平均
    num_classes = pred.size(1)
    precision_list, recall_list, f1_list = [], [], []

    for c in range(num_classes):
        tp = ((pred_labels == c) & (target == c)).sum().item()
        fp = ((pred_labels == c) & (target != c)).sum().item()
        fn = ((pred_labels != c) & (target == c)).sum().item()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": accuracy,
        "precision": sum(precision_list) / num_classes,
        "recall": sum(recall_list) / num_classes,
        "f1": sum(f1_list) / num_classes,
    }


__all__ = [
    "dice_coefficient",
    "iou_score",
    "pixel_accuracy",
    "classification_metrics",
]

