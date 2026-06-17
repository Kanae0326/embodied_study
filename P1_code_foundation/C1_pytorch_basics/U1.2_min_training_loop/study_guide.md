# 单元 1.2 学习指南：训练最小闭环

> 本单元把单元 1.1 学的零件（tensor、autograd、5 个工具函数）组装成一台能跑的「训练机器」。
> 读法：先通读一遍建立整体图景，再边实现 `project/{dataset,model,train}.py` 边回看对应小节。**理解后合上文档自己默写，别直接抄。**

---

## 一、整体图景：一次训练在干什么

```
            ┌─────────────────────────────────────────────┐
            │                 main()                       │
            │  set_seed → dataloaders → model →            │
            │  criterion/optimizer → 循环若干 epoch        │
            └───────────────┬─────────────────────────────┘
                            │ 每个 epoch
              ┌─────────────┴──────────────┐
              ▼                            ▼
      train_one_epoch(...)          evaluate(...)
      model.train()                 model.eval() + no_grad()
      遍历 train_loader             遍历 val_loader
      ├ forward                     ├ forward
      ├ loss                        ├ loss
      ├ backward                    └ 累加 loss/acc（不反传、不更新）
      ├ optimizer.step
      └ optimizer.zero_grad
              │                            │
              └────────► 打印/记录 train/val 的 loss 与 acc ◄────┘
```

一句话：**训练集上「学」（更新参数），验证集上「考」（只看不学）**。两条曲线一起看，才知道模型是在进步还是在死记硬背。

---

## 二、数据：train / val 划分与 DataLoader

FashionMNIST：灰度 `28×28`、10 类、官方 60000 训练 + 10000 测试。

**为什么要切 train / val？**
- 训练集用来**更新参数**；验证集用来**判断泛化**（模型没见过的数据上表现如何）。
- 如果只看训练集指标，你无法发现「过拟合」——模型把训练样本背下来了，但换一批就不行。

**两种常见切法（任选其一，在 `get_dataloaders` 里实现）：**
1. 把 60000 训练集再切成 train / val（如 55000 / 5000），官方测试集留到最后；
2. 简化起见，直接用官方测试集当 val（本单元够用）。

**DataLoader 要点：**
```python
train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2)
```
- **训练集 `shuffle=True`**（每个 epoch 打乱，避免按顺序学出偏差）；**验证集 `shuffle=False`**（评估要可复现）。
- 一个 batch：`images` 形状 `(B, 1, 28, 28)`、`float32`、范围 `[0,1]`（`transforms.ToTensor()` 已归一化）；`labels` 形状 `(B,)`、**`int64`(long)**。
- 标签 dtype 必须是 `long`，否则 `CrossEntropyLoss` 报错——这是高频坑。

> 验证：脚手架 `dataset.py` 自带 `__main__` 会断言形状/dtype/范围。先让它过。

---

## 三、模型：一个最小 CNN

**为什么图像用 CNN 而不是纯 MLP？** 卷积核在整张图上滑动、**共享权重**，能用很少的参数捕捉局部图案（边、角、纹理），且对位置平移更鲁棒。

**硬约束（其余自由发挥）：** 输入 `(B,1,28,28)` → 输出 `(B,10)` 的 **logits**，最后一层**不要加 softmax**（`CrossEntropyLoss` 内部会做）。

**用形状推导来设计**（示例，不是标准答案，你的层数/通道自定）：
```
输入            (B, 1, 28, 28)
Conv(1→32,k3,pad1) + ReLU      (B, 32, 28, 28)
MaxPool(2)                     (B, 32, 14, 14)
Conv(32→64,k3,pad1) + ReLU     (B, 64, 14, 14)
MaxPool(2)                     (B, 64,  7,  7)
Flatten                        (B, 64*7*7) = (B, 3136)
Linear(3136→128) + ReLU        (B, 128)
Linear(128→10)                 (B, 10)   ← logits
```
- 写的时候用 `print(x.shape)` 在 forward 里逐层打印，确认每一步和你算的一致——**90% 的新手 bug 是形状对不上**。
- `nn.Module` 的写法：`__init__` 里**定义层**，`forward` 里**按顺序调用**。
- 用单元 1.1 的 `count_parameters(model)` 看看参数量（这个小 CNN 大概几十万到一百多万）。

> 验证：脚手架 `model.py` 的 `__main__` 会喂一个 `(4,1,28,28)` 假输入，断言输出是 `(4,10)`。

---

## 四、训练循环：五步在代码里的位置

单元 1.1 你已经知道五步顺序。现在把它放进一个遍历 batch 的循环里。**这是整个深度学习最核心的 20 行，务必能默写：**

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                          # 进入训练模式
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)   # 上 device

        outputs = model(images)            # 1) forward：得到 logits (B,10)
        loss = criterion(outputs, labels)  # 2) loss：logits vs 真标签

        optimizer.zero_grad()              # 清梯度（放 backward 前后都行，但必须有）
        loss.backward()                    # 3) backward：算每个参数的梯度
        optimizer.step()                   # 4) step：按梯度更新参数

        total_loss += loss.item() * images.size(0)   # 用 .item() 取标量！
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total
```

**每一步为什么在那儿：**
- `model.train()`：让 dropout / batchnorm 处于训练行为（即使你这次没用，养成习惯）。
- `.to(device)`：数据必须和模型在同一设备，否则报 `device mismatch`。
- `zero_grad()`：梯度默认**累加**，不清零会把上一个 batch 的梯度叠进来 → 训练乱掉。
- `loss.item()`：把单元素张量取成 Python float 再累加；直接 `+= loss` 会**留住计算图**，显存/内存一路涨。

---

## 五、评估：为什么和训练不一样

```python
@torch.no_grad()                           # 不建计算图，省显存、加速
def evaluate(model, loader, criterion, device):
    model.eval()                           # 进入评估模式
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total
```

- `model.eval()` + `torch.no_grad()` 是评估的**标配两件套**：前者切换 dropout/BN 行为，后者关闭梯度追踪。
- 评估**不调用** `backward()` / `optimizer.step()`——只看不学。
- 忘了 `eval()` 或 `no_grad()` 不会报错，但结果会偏、还更慢——属于「沉默的 bug」，要靠习惯防住。

---

## 六、损失与指标

- **`nn.CrossEntropyLoss`**：输入 **raw logits** `(B,10)` 和 **类别索引** `(B,)`（不是 one-hot、不是概率）。它内部 = `LogSoftmax + NLLLoss`，所以你模型最后**别加 softmax**。
- **accuracy**：`(logits.argmax(dim=1) == labels).float().mean()`——就是单元 1.1 你写的 `accuracy()`。
- 逐 epoch 打印四个数：`train_loss / train_acc / val_loss / val_acc`，这就是你的「第一版训练日志」。

---

## 七、main() 的整体流程

```python
def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_dataloaders(batch_size=64)
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")
        # （加分）va_acc > best_acc 时 save_checkpoint(...)
```

调试顺序：**先把 `num_epochs=1` + 一个很小的子集跑通**，确认 loss 在下降、形状没错，再放大正式训练。

---

## 八、过拟合：train 与 val 一起看

| 现象 | train acc | val acc | 解读 |
|---|---|---|---|
| 正常学习 | 上升 | 跟着上升 | 继续训 |
| 欠拟合 | 都很低、不太动 | 都很低 | 模型太弱 / lr 太小 / 训太短 |
| 过拟合 | 很高（接近 100%） | 先升后**停滞或下降** | 模型背训练集了；后续单元会学 早停 / 正则 |

本单元不要求解决过拟合，只要求**能看出来**——这是 val 集存在的意义。

---

## 九、常见 bug 清单（症状 → 原因 → 排查）

| 症状 | 很可能的原因 | 排查 |
|---|---|---|
| loss 不降 / 乱跳 / 变 NaN | 忘了 `zero_grad`；lr 太大 | 打印 grad、调小 lr、确认五步顺序 |
| `RuntimeError: expected scalar type Long` | 标签不是 `long` | `labels = labels.long()` 或检查 dataset |
| `shape mismatch` / Linear 维度错 | Flatten 后维度算错 | 在 forward 里 `print(x.shape)` 逐层看 |
| `Expected all tensors on same device` | 数据/模型不在同一 device | 模型 `.to(device)`，每个 batch 也 `.to(device)` |
| 显存/内存一路涨 | 用 `loss` 而非 `loss.item()` 累加 | 累加标量一律 `.item()` |
| val acc 比 train 还离谱地低/不稳 | 评估忘了 `model.eval()`/`no_grad()` | 补上评估两件套 |
| loss 用 softmax 后概率算 | 最后一层加了 softmax | 去掉 softmax，`CrossEntropyLoss` 吃 logits |

---

## 十、本单元自测（不看笔记回答）

1. `train_one_epoch` 里五步的顺序是什么？`zero_grad` 为什么不能省？
2. `model.train()` 和 `model.eval()` 有什么区别？评估为什么还要 `torch.no_grad()`？
3. `CrossEntropyLoss` 的两个输入分别是什么形状、什么 dtype？模型最后该不该加 softmax？
4. 累加统计量时为什么要用 `loss.item()` 而不是 `loss`？
5. 训练集 `shuffle=True`、验证集 `shuffle=False`，各自为什么？
6. 一个 `(B,1,28,28)` 输入，经过两次 `Conv(k3,pad1)+MaxPool(2)` 后空间尺寸变成多少？（答：`7×7`）
7. train acc 很高但 val acc 停滞，说明什么？这一单元要不要解决它？
8. 你怎么用单元 1.1 的 `accuracy()` 和 `count_parameters()`？

> 八题都能流畅回答 + `train.py` 能跑出下降的 loss 和合理的 val acc = 本单元出关。
