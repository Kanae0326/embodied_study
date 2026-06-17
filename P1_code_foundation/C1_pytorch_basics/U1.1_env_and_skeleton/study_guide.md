# 单元 1.1 学习指南：PyTorch 张量与自动微分

## 本单元学习路径（v4 进度制 · 按完成度推进，不计时）

> 做完一步、对应自测通过，再进入下一步；不绑定天数/小时。
> 你有 VLA 微调经验，张量/autograd 这部分若能直接通过文末自测，可跳过讲解、直接做练习。

```
步骤 1  环境配置              --> environment_setup.md
步骤 2  shell 基础 + 练习 1-6 --> shell_basics.md
步骤 3  张量 Part 1-5         --> tensor_playground.py 前半
步骤 4  autograd Part 6-7     --> tensor_playground.py 后半
步骤 5  自建 project/ + utils --> ../project_skeleton/SETUP.md 指引下自建目录，写 utils.py 5 个函数
```

**本单元不做：** model.py / dataset.py / train.py 的实现 —— 按 v4 划分属于**单元 1.2**（训练最小闭环）。
单元 1.1 专注于"环境就绪 + shell 熟练 + tensor 练透 + utils 自己写出来"。

---

## 一、什么是张量 (Tensor)

张量是 PyTorch 中最基本的数据结构，本质上是**多维数组**。

```
标量 (scalar)     0 维张量    torch.tensor(3.14)
向量 (vector)     1 维张量    torch.tensor([1, 2, 3])          shape: (3,)
矩阵 (matrix)    2 维张量    torch.randn(3, 4)                shape: (3, 4)
3D 张量          3 维张量    torch.randn(2, 3, 4)             shape: (2, 3, 4)
4D 张量          4 维张量    torch.randn(8, 3, 32, 32)        shape: (8, 3, 32, 32)
                              ^batch  ^channels ^H   ^W
```

**关键属性：**

| 属性 | 含义 | 示例 |
|---|---|---|
| `.shape` / `.size()` | 各维度的大小 | `torch.randn(2,3).shape` → `torch.Size([2, 3])` |
| `.dtype` | 数据类型 | `torch.float32`（默认）、`torch.long`（分类标签） |
| `.device` | 所在设备 | `cpu` 或 `cuda:0` |
| `.ndim` | 维度数量 | `torch.randn(2,3,4).ndim` → `3` |
| `.requires_grad` | 是否需要计算梯度 | 模型参数默认为 `True` |

---

## 二、深度学习三个核心概念

### Batch（批次）

训练数据不会一次全部喂给模型，而是分批处理：

```
全部训练数据: 60000 张图片
batch_size = 64

每次训练拿 64 张图片组成一个 batch
一个 batch 的 shape: (64, 1, 28, 28)
                       ^B  ^C ^H  ^W

为什么分 batch？
  1. 显存/内存装不下全部数据
  2. 小 batch 的梯度有随机性，有助于跳出局部最优
  3. batch 太大收敛慢且泛化差，太小梯度噪声大
```

### Epoch（轮次）

```
1 个 epoch = 把全部训练数据过一遍

60000 张图 / batch_size 64 = 938 个 batch（iteration）
训练 10 个 epoch = 把数据集过 10 遍 = 跑了 9380 个 iteration
```

### Gradient（梯度）

梯度是 loss 相对于每个参数的偏导数，指出**参数应该朝哪个方向调整才能让 loss 变小**。

```
直觉：你在山上蒙着眼，想走到山谷（最小 loss）
      梯度 = 脚下最陡的方向
      梯度下降 = 每次朝最陡的反方向走一小步
```

---

## 三、一次训练步骤 (Training Step) 到底发生了什么

```
                 ┌─────────────┐
                 │  取一个 batch │
                 └──────┬──────┘
                        ▼
              ┌──────────────────┐
              │  model.forward()  │   前向传播：数据通过网络，得到预测
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  loss = criterion │   计算损失：预测 vs 真实标签的差距
              │   (output, label) │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  loss.backward()  │   反向传播：从 loss 回溯，算出每个参数的梯度
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ optimizer.step()  │   参数更新：按梯度方向走一步
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │optimizer.zero_grad│   清除梯度（否则梯度会累加）
              └──────────────────┘
```

**要点：**
- `forward` → `loss` → `backward` → `step` → `zero_grad`，这五步是不可打乱的固定顺序
- `backward()` 不更新参数，只计算梯度并存在 `param.grad` 里
- `step()` 才真正更新参数：`param = param - lr * param.grad`
- 如果不调 `zero_grad()`，梯度会在每次 `backward()` 时累加

---

## 四、PyTorch Autograd 机制

Autograd 是 PyTorch 的自动微分引擎。你不需要手动推导导数公式，PyTorch 会自动帮你算。

### 计算图

```python
x = torch.tensor(2.0, requires_grad=True)   # 叶子节点，需要梯度
y = x ** 2 + 3 * x + 1                       # y = x^2 + 3x + 1
y.backward()                                  # 反向传播
print(x.grad)                                 # dy/dx = 2x + 3 = 7.0
```

执行过程中 PyTorch 在幕后构建了一个**计算图**：

```
x ──→ [** 2] ──→ x^2 ──┐
                         ├──→ [+] ──→ [+1] ──→ y
x ──→ [* 3] ──→ 3x ──┘

backward() 从 y 开始，沿着这张图反向走，用链式法则算出 dy/dx
```

### 关键规则

1. **只有 `requires_grad=True` 的张量才会被追踪**
2. **`.backward()` 只能对标量调用**（loss 通常就是标量）
3. **梯度会累加**，所以每次需要 `zero_grad()`
4. **`.detach()` 可以切断计算图**（推理时不需要梯度）
5. **`with torch.no_grad():` 块内不构建计算图**（用于评估/推理）

---

## 五、常用张量操作速查

### 形状操作

| 操作 | 作用 | 示例 |
|---|---|---|
| `reshape(shape)` / `view(shape)` | 改变形状（元素不变） | `x.reshape(2, 6)` |
| `unsqueeze(dim)` | 在指定位置插入一个维度 | `(3,4)` → `unsqueeze(0)` → `(1,3,4)` |
| `squeeze(dim)` | 移除大小为 1 的维度 | `(1,3,4)` → `squeeze(0)` → `(3,4)` |
| `permute(dims)` | 调换维度顺序 | `(B,H,W,C)` → `permute(0,3,1,2)` → `(B,C,H,W)` |
| `flatten(start, end)` | 展平指定范围的维度 | `(B,C,H,W)` → `flatten(1)` → `(B, C*H*W)` |

### 广播 (Broadcasting)

当两个张量 shape 不完全相同时，PyTorch 会尝试**自动广播**使它们兼容：

```
规则（从最右侧维度开始对齐）：
  1. 两个维度相等 → OK
  2. 其中一个是 1  → 广播到另一个的大小
  3. 其中一个不存在 → 视为 1，然后广播

示例：
  (4, 3) + (   3) → (4, 3)     # 右侧维度相同
  (4, 3) + (4, 1) → (4, 3)     # 第二维度 1 → 广播到 3
  (4, 3) + (   1) → (4, 3)     # 右侧 1 → 广播到 3，左侧缺少 → 广播到 4
```

### 索引

```python
x = torch.randn(5, 4, 3)

x[0]            # 取第 0 个，shape (4, 3)
x[:, 1]         # 取所有行的第 1 列，shape (5, 3)
x[:, :, -1]     # 取最后一个通道，shape (5, 4)
x[x > 0]        # 布尔索引，返回 1D 张量
x[[0, 2, 4]]    # 花式索引，取第 0,2,4 个
```

---

## 六、推荐学习资源

### 必读（按优先级排序）

1. **PyTorch 官方 Tensor 教程**
   https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
   - 15 分钟快速入门张量操作

2. **PyTorch 官方 Autograd 教程**
   https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
   - 理解计算图和自动微分

3. **PyTorch 官方 Build Model 教程**
   https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
   - 理解 `nn.Module` 的基本结构

### 补充阅读（有余力时）

4. **Dive into Deep Learning - 2. Preliminaries**
   https://d2l.ai/chapter_preliminaries/index.html
   - 系统讲解张量运算和自动微分

5. **PyTorch Cheat Sheet**
   https://pytorch.org/tutorials/beginner/ptcheat.html
   - 常用 API 速查

### 视频（可选）

6. **3Blue1Brown - Neural Networks 系列**
   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   - 直觉性极强的梯度和反向传播讲解

---

## 七、本单元核心概念自测

学完本单元后，你应该能回答以下问题（不看笔记）：

1. 什么是张量？它和 numpy array 的主要区别是什么？
2. `requires_grad=True` 意味着什么？
3. `loss.backward()` 做了什么？它更新参数了吗？
4. 为什么每次训练步骤要调用 `zero_grad()`？
5. 一个 shape 为 `(32, 3, 28, 28)` 的张量，每个维度分别代表什么？
6. `(4, 1)` 和 `(1, 3)` 相加的结果 shape 是什么？为什么？
7. `view` 和 `reshape` 有什么区别？什么时候用 `reshape` 更安全？
8. batch_size = 64，数据集有 10000 张图，1 个 epoch 有多少个 iteration？
