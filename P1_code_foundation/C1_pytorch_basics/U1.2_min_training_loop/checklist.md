# 单元 1.2 验收清单 / 出关条件

> v4 原则：单元 1.2 做「训练最小闭环」。前置单元 1.1 必须已出关（`project/utils.py` 可用）。
> 本单元专注：**自己写出 dataset / model / 训练循环，跑通一个会收敛的 FashionMNIST 分类器**。

---

## 前置确认

- [ ] 单元 1.1 已出关：`project/utils.py` 的 5 个函数通过自测
- [ ] `conda activate embodied` 环境可用，`import torch / torchvision` 正常

---

## 数据（project/dataset.py）

- [ ] 实现 `get_dataloaders`，返回 `(train_loader, val_loader)`
- [ ] 训练集 `shuffle=True`，验证集 `shuffle=False`
- [ ] 跑脚手架自带测试通过：`cd project/ && python dataset.py`
  - [ ] 一个 batch：`images` 形状 `(B,1,28,28)`、`float32`、范围 `[0,1]`
  - [ ] `labels` 形状 `(B,)`、dtype `int64`(long)
- [ ] 能说出 train/val 为什么要分开

## 模型（project/model.py）

- [ ] 实现 `SimpleCNN`：输入 `(B,1,28,28)` → 输出 `(B,10)` logits
- [ ] **最后一层没有 softmax**
- [ ] 跑脚手架自带测试通过：`python model.py`（输出 shape `(4,10)`）
- [ ] 能用 `print(x.shape)` 解释 forward 每一层的形状变化
- [ ] 用 `count_parameters` 报得出参数量

## 训练循环（project/train.py）

- [ ] 实现 `train_one_epoch`：含 `model.train()` + 五步（zero_grad→forward→loss→backward→step）
- [ ] 实现 `evaluate`：含 `model.eval()` + `torch.no_grad()`，**不**反传不更新
- [ ] 实现 `main`：set_seed → dataloaders → model → criterion/optimizer → 循环 epoch → 打印日志
- [ ] 统计量用 `.item()` 累加（不是直接加张量）

## 跑通与日志

- [ ] 先用小子集 + 1 个 epoch 跑通（确认 loss 在动、形状没错）
- [ ] 再正式训练若干 epoch，得到逐 epoch 的 `train_loss/acc | val_loss/acc` 日志
- [ ] **val accuracy 明显高于 10%**（像样的小 CNN 通常 88–92%）
- [ ] loss 整体在下降

---

## 概念自测（不看笔记回答）

- [ ] 五步顺序？`zero_grad` 为什么不能省？能放在 backward 和 step 之间吗？
  > zero_grad→forward→loss→backward→step；梯度默认累加，每次 backward 前必须清掉上一轮梯度，否则上个 batch 的梯度叠进来、训练乱掉；**绝不能**清在 backward 与 step 之间——会把刚算出的梯度抹掉，参数不再更新。
- [ ] `model.train()` vs `model.eval()`？评估为什么还要 `torch.no_grad()`？
  > train/eval 切换 dropout/BN 行为；`no_grad` 关闭梯度追踪，省显存且更快，评估不需要梯度。
- [ ] `CrossEntropyLoss` 的两个输入形状/类型？要不要 softmax？
  > logits `(B,C)` + 类别索引 `(B,)` long；**不要** softmax，loss 内部已含。
- [ ] 为什么用 `loss.item()` 累加？
  > 直接 `+= loss` 会留住计算图，显存/内存一路涨；`.item()` 取成 Python float。
- [ ] train acc 高、val acc 停滞说明什么？本单元要解决吗？
  > 过拟合；本单元只要求能看出来，不要求解决（后续单元学早停/正则）。

---

## 完成标准

**最低标准（弱算力出关）：**
- dataset / model 的脚手架测试通过
- 训练闭环完整跑通（可用小子集 + 少量 epoch），日志显示 loss 在下降

**标准出关：**
- 以上 + 正式训练若干 epoch，val acc 达到合理水平（远高于随机）
- 能流畅回答全部概念自测
- 第一版完整训练日志已保存

**可选加分：**
- 画 train/val 的 loss、acc 曲线
- 用 `save_checkpoint` 存 best
- 加 `argparse`，命令行可改 lr/batch/epochs（这其实是单元 1.3 的预热）

---

## 遇到困难时

1. **先查形状**：`print(x.shape)`（90% 的 bug 是形状/维度）
2. 对照 `study_guide.md` 第九节「常见 bug 清单」按症状定位
3. 自己尝试 20–30 分钟后再用 AI；AI 用来**解释报错 / 审查逻辑**，不是直接代写整个 train.py
4. 拆小验证：先单独跑通 dataset.py、再 model.py，最后才连 train.py
