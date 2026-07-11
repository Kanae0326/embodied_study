# 单元 1.2 · 训练最小闭环

> **篇 1 · 章 1（PyTorch 基础）· 单元 1.2。** 先看本页。
> **前置：** 单元 1.1 必须先出关——本单元会用到你在 `../project/utils.py` 里写好的 `set_seed / accuracy / save_checkpoint / count_parameters`。
> 本单元在章级 `../project_skeleton/` 的 `dataset.py / model.py / train.py` 三个脚手架上，把 FashionMNIST 的训练闭环亲手写出来。

---

## 单元目标
从「会写工具函数」升级到「能自己写出第一段完整的训练循环」：加载数据 → 定义模型 → 前向 → 反传 → 更新 → 评估，跑通一个能收敛的 FashionMNIST 分类器。

## 核心必做
- 用 torchvision 加载 FashionMNIST，切 train / val，确认 batch 形状与标签类型　→ 在 `project/dataset.py` 实现 `get_dataloaders`
- 自己写一个小 CNN 与 forward（输入 `(B,1,28,28)` → 输出 `(B,10)` logits）　→ 在 `project/model.py` 实现 `SimpleCNN`
- 实现 `train_one_epoch` 与 `evaluate`，跑通若干 epoch 并记录 loss / accuracy 日志　→ 在 `project/train.py` 实现
- 理论补强：训练五步循环、train/eval 模式、`no_grad`、过拟合信号　→ `study_guide.md`

## 出关条件
- **标准出关：** 能说清 `zero_grad → forward → loss → backward → optimizer.step` 的顺序、各自作用，以及本质规则（每次 backward 前清掉上一轮梯度，且不能清在 backward 与 step 之间）；`train.py` 能从命令行跑通、loss 下降、val accuracy 明显高于随机（10 类随机=10%，像样的小 CNN 通常能到 88–92%）；有第一版完整训练日志（逐 epoch 的 train/val loss 与 acc）。
- **弱算力出关：** 数据量 / epoch 数可缩小（只取一部分训练集、跑 2–3 个 epoch），只要闭环完整跑通、日志显示 loss 在下降即算过。

## 交付物
可运行的 `project/{dataset.py, model.py, train.py}` + 第一版训练日志。

## 可选加分
画 train/val 的 loss 与 accuracy 曲线；用单元 1.1 的 `save_checkpoint` 保存 best；加 `argparse` 让 lr/batch/epochs 可从命令行改。

---

## 学习路径（按进度推进，不计时）

> 先打通「形状」，再写训练循环。每步先自己写 20–30 分钟再问 AI；写完合上资料能默写出来才算懂。
> **外部资料的定位：卡壳时的查阅来源，不是前置必读**——`study_guide.md` 本身足够支撑写出全部三个文件。资料清单及"用到什么程度"见 `study_guide.md` 第十节。

1. **建立图景** → 读 `study_guide.md` 第一遍，看懂训练循环的整体数据流。
   - （可选）想看 `nn.Module` 的官方讲法：[Build Model 官方教程](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)，15 分钟
2. **数据** → 在 `project/dataset.py` 实现 `get_dataloaders`；用脚手架自带的 `if __name__ == "__main__"` 测试验证 batch 形状/类型。
   - 卡壳时查：[FashionMNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html) 与 [DataLoader](https://pytorch.org/docs/stable/data.html) 官方文档（只看参数签名和示例）
3. **模型** → 在 `project/model.py` 实现 `SimpleCNN`；用其 `__main__` 测试验证输出 `(B,10)` 与参数量；用 `print(x.shape)` 逐层确认形状。
   - 卡壳时查：[nn.Conv2d 文档](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)（页内有输出尺寸公式）；（可选）卷积原理没直觉，先看 `study_guide.md` 三，再不够看 [D2L 卷积神经网络章](https://d2l.ai/chapter_convolutional-neural-networks/index.html)
4. **训练循环** → 在 `project/train.py` 实现 `train_one_epoch / evaluate / main`；**先用很小的子集 + 1 个 epoch 跑通**，确认 loss 在动，再放大。
   - 卡壳时查：`study_guide.md` 第九节「常见 bug 清单」按症状定位；实在写不出来看 [Optimization 官方教程](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)（有完整循环示例，**看完关掉自己默写**）
5. **正式训练** → 跑完整若干 epoch，记录日志；（加分）画曲线 + 存 best。
   - 卡壳时查：[torch.optim 文档](https://pytorch.org/docs/stable/optim.html)（调 lr / 换优化器时）

## 验收
逐条对照 `checklist.md`。

---

## 与脚手架的关系（本单元先复制，再填空）
`../project_skeleton/{dataset,model,train}.py` 是**参考接口 + 自带测试**（函数体是 TODO）。本单元开始时，把这三个文件**复制**到你的 `project/`（覆盖单元 1.1 建的空占位），之后**只改复制过来的版本**；skeleton 原件保持原样，留作对照：

```bash
cd ~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/project/
cp ../project_skeleton/dataset.py ../project_skeleton/model.py ../project_skeleton/train.py .
```

每个文件底部自带 `if __name__ == "__main__":` 测试块；实现完成后在 `project/` 里直接 `python dataset.py`（或 `model.py` / `train.py`）验证。

## 常见 bug 速查（详见 `study_guide.md` 第九节）
忘了 `optimizer.zero_grad()`（梯度累加，训练发散）｜评估时忘了 `model.eval()` + `torch.no_grad()`｜`CrossEntropyLoss` 喂了 softmax 后的概率而非 raw logits｜标签 dtype 不是 `long`｜数据与模型不在同一 `device`｜用 `loss`（而非 `loss.item()`）累加导致显存/内存涨。

---

## 单元导航
← 上一单元：[`../U1.1_env_and_skeleton/index.md`](../U1.1_env_and_skeleton/index.md)（环境与骨架）
→ 下一单元：单元 1.3 工程化 1.0（config / checkpoint / 曲线图，待充实）
