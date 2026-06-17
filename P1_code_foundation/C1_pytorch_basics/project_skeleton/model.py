"""
model.py — 模型定义（参考接口）
================================

【单元 1.2 任务；单元 1.1 只需建空文件占位。理论见 ../U1.2_min_training_loop/study_guide.md】

单元 1.2 时，你需要实现一个简单 CNN 用于 FashionMNIST 分类：
  - 输入 shape: (B, 1, 28, 28) —— 灰度图
  - 输出 shape: (B, 10) —— 10 类的 logits
  - 最后一层不要加 softmax（CrossEntropyLoss 内部会处理）

**设计自由度：** 用几层卷积、kernel_size、hidden 维度、是否用 BatchNorm 都由你定。
唯一硬约束就是输入输出 shape。

卡住时的学习路径：
  1. 先读 PyTorch Build Model 官方教程
  2. 自己画一张网络结构图（纸/白板）
  3. 按图写代码，用 print(x.shape) 逐层验证形状
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN 分类器。__init__ 里定义层，forward 里按顺序调用。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # TODO (单元 1.2): 定义你的网络层
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) -> logits: (B, num_classes)"""
        # TODO (单元 1.2)
        pass


# ============================================================
# 测试代码（单元 1.2 用）
# ============================================================

if __name__ == "__main__":
    print("Testing SimpleCNN...")

    model = SimpleCNN(num_classes=10)

    x = torch.randn(4, 1, 28, 28)
    print(f"Input shape:  {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")

    assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"
    print("PASSED: output shape is correct (batch_size, 10)")

    from utils import count_parameters
    n = count_parameters(model)
    print(f"Total trainable parameters: {n:,}")
