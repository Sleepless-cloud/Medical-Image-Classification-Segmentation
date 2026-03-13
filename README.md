# ResGANet 医学图像分割与分类

基于 **ResGANet (Residual Group Attention Network)** 的医学图像分析模型实现，支持皮肤病变的自动分割与分类。论文参考：*用于医学图像分析的残差群体注意力网络*，数据集：**ISIC 2018 皮肤损伤数据集**。

---

## 功能概览

| 任务 | 说明 |
|------|------|
| **分割** | 自动分割皮肤病变区域，输出分割掩码与热图 |
| **分类** | 识别 7 类皮肤病变类型（MEL、NV、BCC、AKIEC、BKL、DF、VASC）|

---

## 效果展示

### 分割任务

上传皮肤镜图像后，模型自动分割病变区域，并展示分割掩码与热力图。

![segmentation_demo](https://github.com/user-attachments/assets/cf982ee2-33c4-4ea7-8c06-cbc2483e9c65)

### 分类任务

上传皮肤疾病图像，模型预测病变类型及各类别置信度。

![classification_demo](https://github.com/user-attachments/assets/37cf9085-b1a8-4962-a9d4-15f291b740fa)

---

## 项目结构

```
├── resganet/                    # 核心模块
│   ├── attention.py             # 通道注意力(CAM) + 空间注意力(SAM) + 分组注意力块(GAB)
│   ├── blocks.py                # ResGANetBlock (Bottleneck 变体)
│   ├── model.py                 # ResGANet-50 分类网络
│   ├── segmentation.py          # ResGANet-UNet 分割网络
│   ├── datasets.py              # ISIC 2018 数据加载器
│   ├── metrics.py               # 评估指标 (Dice/IoU/Accuracy/F1)
│   └── __init__.py
├── train_classification.py      # 分类训练脚本
├── train_segmentation.py        # 分割训练脚本
├── requirements.txt             # 依赖包
└── README.md
```

---

## 环境安装

```bash
pip install -r requirements.txt
```

**依赖概览：** PyTorch、torchvision、numpy、Pillow、pandas、scikit-learn、tqdm、matplotlib、gradio

---

## 数据准备

将 ISIC 2018 数据集放置在 `dataset/` 目录下：

```
dataset/
├── ISIC2018_Task1-2_Training_Input/       # 分割训练图像
├── ISIC2018_Task1_Training_GroundTruth/   # 分割训练掩码
├── ISIC2018_Task3_Training_Input/         # 分类训练图像
├── ISIC2018_Task3_Training_GroundTruth/   # 分类训练标签 CSV
├── ISIC2018_Task1-2_Validation_Input/     # 验证集
└── ...
```

---

## 训练模型

### 分割任务

```bash
python train_segmentation.py \
    --data_root ./dataset \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-3 \
    --img_size 256
```

### 分类任务

```bash
python train_classification.py \
    --data_root ./dataset \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --img_size 224
```

**设备：** 支持 CUDA、Apple Silicon (MPS) 与 CPU，脚本会自动选择可用设备。

---

## 病变类型（7 类）

| 缩写 | 名称 |
|------|------|
| MEL | 黑色素瘤 (Melanoma) |
| NV | 色素痣 (Melanocytic Nevus) |
| BCC | 基底细胞癌 (Basal Cell Carcinoma) |
| AKIEC | 光化性角化病 (Actinic Keratosis) |
| BKL | 良性角化病 (Benign Keratosis) |
| DF | 皮肤纤维瘤 (Dermatofibroma) |
| VASC | 血管病变 (Vascular Lesion) |

---

## 参考文献

```bibtex
@article{cheng2022resganet,
  title={Residual group attention network for medical image classification and segmentation},
  author={Cheng, Jun and others},
  journal={Medical Image Analysis},
  year={2022},
  publisher={Elsevier}
}
```

---

## 致谢

- 论文作者：Cheng et al. (2022)
- 数据集：ISIC 2018 Challenge
- 框架：PyTorch
