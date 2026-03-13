from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ISIC2018SegmentationDataset(Dataset):
    """ISIC 2018 Task1 分割数据集。
    
    结构：
    - images: ISIC2018_Task1-2_Training_Input/ISIC_*.jpg
    - masks: ISIC2018_Task1_Training_GroundTruth/ISIC_*_segmentation.png
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_size: int = 256,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        if split == "train":
            self.image_dir = self.root / "ISIC2018_Task1-2_Training_Input"
            self.mask_dir = self.root / "ISIC2018_Task1_Training_GroundTruth"
        elif split == "val":
            self.image_dir = self.root / "ISIC2018_Task1-2_Validation_Input"
            self.mask_dir = self.root / "ISIC2018_Task1_Validation_GroundTruth"
        elif split == "test":
            self.image_dir = self.root / "ISIC2018_Task1-2_Test_Input"
            self.mask_dir = self.root / "ISIC2018_Task1_Test_GroundTruth"
        else:
            raise ValueError(f"split 必须是 train/val/test，得到 {split}")

        # 扫描所有图像
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
        self.ids = [f.replace(".jpg", "") for f in self.image_files]

        # 默认变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]
        img_path = self.image_dir / f"{img_id}.jpg"
        mask_path = self.mask_dir / f"{img_id}_segmentation.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.mask_transform(mask)
        # 二值化：>0.5 为前景
        mask = (mask > 0.5).float()

        return image, mask


class ISIC2018ClassificationDataset(Dataset):
    """ISIC 2018 Task3 分类数据集。
    
    7类皮肤病变：MEL, NV, BCC, AKIEC, BKL, DF, VASC
    """

    CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_size: int = 224,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        if split == "train":
            self.image_dir = self.root / "ISIC2018_Task3_Training_Input"
            csv_path = self.root / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv"
        elif split == "val":
            self.image_dir = self.root / "ISIC2018_Task3_Validation_Input"
            csv_path = self.root / "ISIC2018_Task3_Validation_GroundTruth" / "ISIC2018_Task3_Validation_GroundTruth.csv"
        elif split == "test":
            self.image_dir = self.root / "ISIC2018_Task3_Test_Input"
            csv_path = self.root / "ISIC2018_Task3_Test_GroundTruth" / "ISIC2018_Task3_Test_GroundTruth.csv"
        else:
            raise ValueError(f"split 必须是 train/val/test，得到 {split}")

        # 读取标签CSV
        df = pd.read_csv(csv_path)
        self.image_ids = df["image"].values
        self.labels = df[self.CLASS_NAMES].values.argmax(axis=1)

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_id = self.image_ids[idx]
        img_path = self.image_dir / f"{img_id}.jpg"
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


__all__ = ["ISIC2018SegmentationDataset", "ISIC2018ClassificationDataset"]

