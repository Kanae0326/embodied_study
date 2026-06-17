"""
dataset.py — 数据加载（参考接口）
==================================

【单元 1.2 任务；单元 1.1 只需建空文件占位。理论见 ../U1.2_min_training_loop/study_guide.md】

单元 1.2 时要实现 get_dataloaders：
  - 返回 (train_loader, val_loader)
  - 训练集 shuffle=True，验证集 shuffle=False
  - batch_size、num_workers 作为参数

数据集：FashionMNIST（torchvision 自带），10 类灰度 28x28。
归一化：用 transforms.ToTensor() 转成 [0, 1] 浮点即可（本单元先不额外做 Normalize）。

卡住时查：
  - torchvision.datasets.FashionMNIST 文档
  - torch.utils.data.DataLoader 文档
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """构造 FashionMNIST 的 (train_loader, val_loader)。

    验收：
      - train_loader 每个 batch: images (B, 1, 28, 28) float32 in [0, 1]
      - val_loader 同 shape，shuffle=False
      - labels: (B,) int64
    """
    # TODO (单元 1.2)
    pass


# ============================================================
# 测试代码（单元 1.2 用）
# ============================================================

if __name__ == "__main__":
    print("Testing dataset loading...")

    train_loader, val_loader = get_dataloaders(batch_size=64)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    images, labels = next(iter(train_loader))
    print(f"\nFirst batch:")
    print(f"  images shape: {images.shape}")
    print(f"  images dtype: {images.dtype}")
    print(f"  images range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"  labels shape: {labels.shape}")
    print(f"  labels dtype: {labels.dtype}")
    print(f"  label values: {labels[:10]}")
    print(f"  label names:  {[CLASS_NAMES[l] for l in labels[:5].tolist()]}")

    assert images.shape == (64, 1, 28, 28), f"Unexpected shape: {images.shape}"
    assert labels.shape == (64,), f"Unexpected shape: {labels.shape}"
    assert 0 <= images.min() and images.max() <= 1, "images should be in [0, 1]"
    assert labels.dtype == torch.int64, f"labels dtype should be int64, got {labels.dtype}"

    print("\nPASSED: data loading works correctly!")
