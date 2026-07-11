"""
train.py — 训练主入口（参考接口）
==================================

【单元 1.2 任务；单元 1.1 只需建空文件占位、不跑 smoke test。理论见 ../U1.2_min_training_loop/study_guide.md】

单元 1.2 要实现：
  1. train_one_epoch(model, loader, criterion, optimizer, device) -> (avg_loss, avg_acc)
  2. evaluate(model, loader, criterion, device) -> (avg_loss, avg_acc)
  3. main() —— set_seed、创建 dataloader / model / criterion / optimizer、跑多个 epoch、打印日志
     （argparse 参数化与保存 best 是可选加分，单元 1.3 才正式做）

训练步骤（五步循环）：
  zero_grad → forward → loss → backward → step
  本质规则：每次 backward 前清掉上一轮梯度；绝不能清在 backward 与 step 之间
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一个 epoch，返回 (平均 loss, 平均 accuracy)。记得 model.train()。"""
    # TODO (单元 1.2)
    raise NotImplementedError("单元 1.2 任务")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """验证集评估，返回 (平均 loss, 平均 accuracy)。记得 model.eval() + torch.no_grad()。"""
    # TODO (单元 1.2)
    raise NotImplementedError("单元 1.2 任务")


def main():
    """set_seed → dataloader → model → criterion/optimizer → 训练循环 → 打印日志；（可选加分）argparse 参数化与保存 best。"""
    # TODO (单元 1.2)
    raise NotImplementedError("单元 1.2 任务")


if __name__ == "__main__":
    main()
