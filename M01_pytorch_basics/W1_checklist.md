# M1 W1 验收清单

> v3.1 原则：W1 只做"环境与骨架"。model/dataset/train.py 的 smoke test 已从 W1 移到 W2。
> 本周专注：**环境就绪 + shell 熟练 + tensor 练透 + utils.py 自己写出来**。

---

## 环境验证

- [ ] `conda activate embodied` 正常激活
- [ ] `python -c "import torch; print(torch.__version__)"` 输出版本号
- [ ] `python -c "import torchvision; print('OK')"` 输出 OK
- [ ] `git status` 正常工作
- [ ] VS Code 能识别 `embodied` 环境并提供代码补全

---

## Shell 基础（新增，必做）

参见 `W1_shell_basics.md`。

- [ ] 通读文档一遍
- [ ] 练习 1–6 全部完成
- [ ] 能不查笔记写出 `python x.py > out.log 2>&1 &` 每一部分的含义
- [ ] 能独立用 tmux 跑一个后台任务并 detach / attach
- [ ] 知道 `tail -f` 看实时日志的用法

---

## 张量练习 (tensor_playground.py)

- [ ] Part 1-2 示例代码能正常运行，输出理解无误
- [ ] Part 3: unsqueeze/squeeze 四个 TODO 全部完成，输出 shape 正确
- [ ] Part 4: broadcasting 三个 TODO 全部完成
  - [ ] 特别是 4.3：能解释为什么 channel_mean 需要 reshape 成 `(1, 3, 1, 1)`
- [ ] Part 5: indexing 五个 TODO 全部完成
- [ ] Part 6: autograd 三个 TODO 全部完成
  - [ ] 6.1: x.grad 输出 8.0
  - [ ] 6.2: w.grad 输出 `[2., 4., 6.]`
  - [ ] 6.3: 能解释为什么不清零时 grad 会累加到 8.0
- [ ] Part 7: 综合练习两个 TODO 完成（建议标准，非必做）
  - [ ] 7.1: W.grad shape 为 `(10, 5)`，b.grad shape 为 `(5,)`
  - [ ] 7.2: `output_test.requires_grad` 输出 `False`

---

## 项目骨架（W1 只做 utils.py）

**重点：按 `project_skeleton/SETUP.md` 自己从空目录建文件结构。不要直接改我给的 skeleton —— skeleton 是参考，你要自己写一版。**

- [ ] 自己建出 `project/` 目录和四个文件 `utils.py / model.py / dataset.py / train.py`（model/dataset/train 本周只留空骨架，W2 再填）
- [ ] `utils.py`: 5 个函数全部实现
  - [ ] `set_seed`
  - [ ] `accuracy`
  - [ ] `count_parameters`
  - [ ] `save_checkpoint`
  - [ ] `load_checkpoint`
- [ ] `python utils.py` 运行后输出 "All 5 tests PASSED!"（使用 `project_skeleton/tests/test_utils.py` 验证）

> **注：** `model.py`、`dataset.py`、`train.py` 的实现和 smoke test 已移到 **W2**。本周不做。

---

## 概念自测（不看笔记回答）

- [ ] 什么是张量？和 numpy array 的主要区别？
  > 关键点：张量可以在 GPU 上计算，支持 autograd 自动微分

- [ ] `requires_grad=True` 意味着什么？
  > 关键点：PyTorch 会追踪该张量参与的所有运算，构建计算图，以便后续求梯度

- [ ] `loss.backward()` 做了什么？它更新参数了吗？
  > 关键点：计算所有 requires_grad=True 的叶子节点的梯度，存到 .grad 里；**不更新参数**

- [ ] 为什么要调 `zero_grad()`？
  > 关键点：梯度默认会累加，不清零的话下一次 backward 会加到旧梯度上

- [ ] shape `(32, 3, 28, 28)` 每个维度是什么？
  > batch_size=32, channels=3, height=28, width=28

- [ ] `(4, 1) + (1, 3)` 的结果 shape？
  > `(4, 3)` — 两个维度都广播

- [ ] `view` 和 `reshape` 的区别？
  > view 要求内存连续，reshape 更安全（不连续时会自动复制）

- [ ] batch_size=64, 数据集 10000 张图，1 个 epoch 多少 iteration？
  > 10000 / 64 = 156.25 → 157 个 iteration（最后一个 batch 不满 64）

---

## 完成标准

**最低标准（必须达到）：**
- 环境全部就绪
- shell 练习 1–6 完成
- tensor_playground 的 Part 3-6 全部完成
- utils.py 的 5 个函数通过测试

**建议标准（更好）：**
- 以上全部 + Part 7 综合练习
- 概念自测能流畅回答
- 能在 tmux 里跑后台任务并用 tail -f 监控

**不在本周做（已移至 W2）：**
- model.py 的 SimpleCNN 实现
- dataset.py 的 get_dataloaders 实现
- train.py 的 smoke test

---

## 遇到困难时

1. **先查 PyTorch 官方文档**: https://pytorch.org/docs/stable/
2. **搜索具体报错信息**（复制粘贴错误消息去搜）
3. **用 print(x.shape) 调试形状问题**（90% 的 bug 都是形状不对）
4. **自己尝试 20-30 分钟后**再用 AI 辅助，AI 用来解释报错和审查逻辑
