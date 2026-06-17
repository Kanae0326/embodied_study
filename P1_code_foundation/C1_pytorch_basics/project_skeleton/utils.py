"""
utils.py — 训练工具函数（参考接口）
====================================

这是"接口参考"文件，不是填空作业。
请在 ../project/utils.py 里自己从空文件开始写这 5 个函数，
然后把本文件底部的 if __name__ == "__main__": 块复制过去自验。

5 个函数：
  1. set_seed(seed)         —— 固定所有随机种子
  2. accuracy(output, target) —— 计算分类准确率
  3. save_checkpoint(model, optimizer, epoch, loss, path)
  4. load_checkpoint(path, model, optimizer=None)
  5. count_parameters(model) —— 统计可训练参数数

实现顺序建议：
  set_seed → count_parameters → accuracy → save_checkpoint → load_checkpoint

**不要看本文件的实现细节再去写你自己的。先自己动手 20–30 分钟。**
"""

import os
import random
import tempfile
import torch
import torch.nn as nn
import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定 Python random / NumPy / PyTorch(CPU+CUDA) 的随机种子，确保实验可复现。"""
    # TODO: 你自己实现
    pass


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """分类准确率。output: (B, C) logits; target: (B,) 类别索引; 返回 [0, 1] 浮点。"""
    # TODO
    pass


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """保存 {'epoch', 'model_state_dict', 'optimizer_state_dict', 'loss'} 到 path。
    路径目录不存在时要自动创建。"""
    # TODO
    pass


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    """从 path 加载 checkpoint，把权重加载到 model，
    如果 optimizer 不为 None 也加载其状态。返回整个 checkpoint 字典。"""
    # TODO
    pass


def count_parameters(model: nn.Module) -> int:
    """统计模型中 requires_grad=True 的参数总数。"""
    # TODO
    pass


# ============================================================
# 测试代码 —— 复制到你自己的 project/utils.py 底部，然后 python utils.py 自验
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Testing utils.py")
    print("=" * 50)

    # --- Test set_seed ---
    print("\n[1/5] Testing set_seed...")
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b), f"set_seed failed: {a} != {b}"
    print("  PASSED: set_seed produces reproducible results")

    # --- Test accuracy ---
    print("\n[2/5] Testing accuracy...")
    output = torch.tensor([[2.0, 0.1, 0.3],
                           [0.1, 3.0, 0.2],
                           [0.5, 0.3, 2.0]])
    target = torch.tensor([0, 1, 0])
    acc = accuracy(output, target)
    assert abs(acc - 2/3) < 1e-6, f"accuracy failed: expected ~0.6667, got {acc}"
    print(f"  PASSED: accuracy = {acc:.4f}")

    # --- Test count_parameters ---
    print("\n[3/5] Testing count_parameters...")
    model = nn.Linear(10, 5)  # weight: 10*5=50, bias: 5, total: 55
    n = count_parameters(model)
    assert n == 55, f"count_parameters failed: expected 55, got {n}"
    print(f"  PASSED: count_parameters = {n}")

    # --- Test save_checkpoint + load_checkpoint ---
    print("\n[4/5] Testing save_checkpoint...")
    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    test_path = os.path.join(tempfile.gettempdir(), "test_ckpt.pth")
    save_checkpoint(model, optimizer, epoch=3, loss=0.456, path=test_path)
    assert os.path.exists(test_path), f"save_checkpoint failed: file not found at {test_path}"
    print(f"  PASSED: checkpoint saved to {test_path}")

    print("\n[5/5] Testing load_checkpoint...")
    model2 = nn.Linear(10, 5)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    ckpt = load_checkpoint(test_path, model2, optimizer2)
    assert ckpt['epoch'] == 3, f"load_checkpoint epoch failed: {ckpt['epoch']}"
    assert abs(ckpt['loss'] - 0.456) < 1e-6, f"load_checkpoint loss failed: {ckpt['loss']}"
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "load_checkpoint weight mismatch"
    print(f"  PASSED: checkpoint loaded, epoch={ckpt['epoch']}, loss={ckpt['loss']:.4f}")

    os.remove(test_path)

    print("\n" + "=" * 50)
    print("All 5 tests PASSED!")
    print("=" * 50)
