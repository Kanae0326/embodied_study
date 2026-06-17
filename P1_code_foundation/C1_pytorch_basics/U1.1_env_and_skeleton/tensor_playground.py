"""
单元 1.1 | 张量练习场 (Tensor Playground)
======================================

使用方法：
  python tensor_playground.py

练习结构：
  - Part 1-2: 完整示例（学习格式）
  - Part 3-7: TODO 练习（你来完成）

规则：
  - 先自己写 20-30 分钟，遇到报错先查 PyTorch 文档
  - 每个 TODO 都有提示和预期输出，对着验证
  - 写完后能不看代码口头解释每个操作的含义
"""

import torch

print("=" * 60)
print("单元 1.1 | Tensor Playground")
print("=" * 60)


# ============================================================
# Part 1: 创建与形状 (shape) —— 完整示例
# ============================================================
print("\n--- Part 1: 创建与形状 (示例) ---")

# 从 Python 列表创建
a = torch.tensor([1, 2, 3])
print(f"a = {a}")
print(f"  shape: {a.shape}, dtype: {a.dtype}, ndim: {a.ndim}")

# 创建随机张量
b = torch.randn(2, 3)  # 正态分布 N(0,1)
print(f"\nb = torch.randn(2, 3):")
print(f"  {b}")
print(f"  shape: {b.shape}")

# 创建全零 / 全一 / 指定值
c_zeros = torch.zeros(3, 4)
c_ones = torch.ones(3, 4)
c_full = torch.full((3, 4), fill_value=7.0)
print(f"\nzeros shape: {c_zeros.shape}")
print(f"ones shape:  {c_ones.shape}")
print(f"full shape:  {c_full.shape}, fill value: {c_full[0, 0].item()}")

# 模拟一个 batch 的图片: (batch_size, channels, height, width)
images = torch.randn(8, 3, 32, 32)
print(f"\nSimulated image batch shape: {images.shape}")
print(f"  batch_size=8, channels=3(RGB), height=32, width=32")


# ============================================================
# Part 2: reshape 与 view —— 完整示例
# ============================================================
print("\n--- Part 2: reshape / view (示例) ---")

x = torch.arange(12)  # [0, 1, 2, ..., 11]
print(f"x = {x}, shape = {x.shape}")

# reshape 成 3x4
x_3x4 = x.reshape(3, 4)
print(f"\nx.reshape(3, 4):")
print(f"  {x_3x4}")

# 用 -1 让 PyTorch 自动推断
x_auto = x.reshape(2, -1)  # 2 行，列数自动算出: 12/2=6
print(f"\nx.reshape(2, -1):")
print(f"  {x_auto}")
print(f"  shape: {x_auto.shape}")

# view 和 reshape 的区别:
# view 要求张量内存连续 (contiguous)，不连续会报错
# reshape 不管连续不连续都能用（可能会复制数据）
# 建议：不确定时用 reshape，更安全


# ============================================================
# Part 3: unsqueeze 与 squeeze —— TODO
# ============================================================
print("\n--- Part 3: unsqueeze / squeeze (TODO) ---")

t = torch.randn(3, 4)
print(f"t shape: {t.shape}")  # (3, 4)

# TODO 3.1: 用 unsqueeze 在 dim=0 插入一个维度
#   期望结果: shape 变为 (1, 3, 4)
#   提示: t.unsqueeze(?)
# t_unsq0 = ???
# print(f"unsqueeze(0): {t_unsq0.shape}")  # 应输出 torch.Size([1, 3, 4])

# TODO 3.2: 用 unsqueeze 在 dim=2 (最后)插入一个维度
#   期望结果: shape 变为 (3, 4, 1)
# t_unsq2 = ???
# print(f"unsqueeze(2): {t_unsq2.shape}")  # 应输出 torch.Size([3, 4, 1])

# TODO 3.3: 创建一个 shape 为 (1, 3, 1, 4) 的张量，然后用 squeeze 去掉所有大小为 1 的维度
#   期望结果: shape 变为 (3, 4)
#   提示: torch.randn(1, 3, 1, 4).squeeze()
# t_sq = ???
# print(f"squeeze all: {t_sq.shape}")  # 应输出 torch.Size([3, 4])

# TODO 3.4: 对同一个张量，只 squeeze dim=0
#   期望结果: shape 变为 (3, 1, 4)
# t_sq0 = ???
# print(f"squeeze(0): {t_sq0.shape}")  # 应输出 torch.Size([3, 1, 4])


# ============================================================
# Part 4: 广播 (Broadcasting) —— TODO
# ============================================================
print("\n--- Part 4: Broadcasting (TODO) ---")

# 广播规则回顾（从右向左对齐）：
#   维度相同 → OK
#   其中一个是 1 → 广播
#   其中一个缺失 → 视为 1，然后广播

# TODO 4.1: 创建 a (shape: 4, 3) 和 b (shape: 3)，做加法
#   期望：结果 shape 为 (4, 3)，b 被广播到每一行
# a = torch.randn(4, 3)
# b = torch.randn(3)
# c = ???  # a + b
# print(f"(4,3) + (3,) = {c.shape}")  # 应输出 torch.Size([4, 3])

# TODO 4.2: 创建 a (shape: 4, 1) 和 b (shape: 1, 3)，做加法
#   期望：结果 shape 为 (4, 3)
#   思考：发生了什么？a 的每行和 b 的每列是怎样组合的？
# a = torch.randn(4, 1)
# b = torch.randn(1, 3)
# c = ???
# print(f"(4,1) + (1,3) = {c.shape}")  # 应输出 torch.Size([4, 3])

# TODO 4.3: 给一个 batch 的图像每个通道加上不同的均值
#   images shape: (8, 3, 32, 32)  — 8张图, 3通道, 32x32
#   mean   shape: 需要你来决定
#   提示：mean 应该是什么 shape 才能正确广播？(想想哪些维度需要是 1)
# images = torch.randn(8, 3, 32, 32)
# channel_mean = torch.tensor([0.485, 0.456, 0.406])  # RGB 三通道均值
# # 需要把 channel_mean reshape 成什么形状？
# channel_mean_reshaped = ???  # 提示: reshape 成 (1, 3, 1, 1) 或用 unsqueeze
# result = images + channel_mean_reshaped
# print(f"Broadcast to images: {result.shape}")  # 应输出 torch.Size([8, 3, 32, 32])


# ============================================================
# Part 5: 索引 (Indexing) —— TODO
# ============================================================
print("\n--- Part 5: Indexing (TODO) ---")

x = torch.arange(20).reshape(4, 5).float()
print(f"x =\n{x}")
# x = tensor([[ 0.,  1.,  2.,  3.,  4.],
#              [ 5.,  6.,  7.,  8.,  9.],
#              [10., 11., 12., 13., 14.],
#              [15., 16., 17., 18., 19.]])

# TODO 5.1: 取出第 2 行（index=2）
#   期望输出: tensor([10., 11., 12., 13., 14.])
# row2 = ???
# print(f"row 2: {row2}")

# TODO 5.2: 取出所有行的第 3 列（index=3）
#   期望输出: tensor([ 3.,  8., 13., 18.])
# col3 = ???
# print(f"col 3: {col3}")

# TODO 5.3: 取出前 2 行、后 3 列组成的子矩阵
#   期望输出: tensor([[2., 3., 4.],
#                      [7., 8., 9.]])
# sub = ???
# print(f"sub matrix:\n{sub}")

# TODO 5.4: 用布尔索引取出所有 > 10 的元素
#   期望输出: tensor([11., 12., 13., 14., 15., 16., 17., 18., 19.])
# big = ???
# print(f"elements > 10: {big}")

# TODO 5.5: 用花式索引取出第 0, 2, 3 行
#   期望输出: shape (3, 5)
# rows = ???
# print(f"fancy indexing:\n{rows}")


# ============================================================
# Part 6: requires_grad 与 backward —— TODO
# ============================================================
print("\n--- Part 6: Autograd (TODO) ---")

# TODO 6.1: 手动验证梯度
#   创建 x = 3.0 (requires_grad=True)
#   计算 y = x^2 + 2x + 1
#   调用 backward()
#   打印 x.grad
#   手算验证: dy/dx = 2x + 2 = 2*3 + 2 = 8.0
#
# x = ???
# y = ???
# ???  # backward
# print(f"x.grad = {x.grad}")  # 应输出 8.0

# TODO 6.2: 向量到标量的梯度
#   创建 w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
#   计算 loss = (w ** 2).sum()   (即 1 + 4 + 9 = 14)
#   调用 backward()
#   打印 w.grad
#   手算验证: d(loss)/d(w_i) = 2 * w_i → [2.0, 4.0, 6.0]
#
# w = ???
# loss = ???
# ???  # backward
# print(f"w.grad = {w.grad}")  # 应输出 tensor([2., 4., 6.])

# TODO 6.3: 梯度累加实验
#   创建 x = torch.tensor(2.0, requires_grad=True)
#   第一次: y1 = x * 3, y1.backward()
#   不清零，第二次: y2 = x * 5, y2.backward()
#   打印 x.grad
#   思考: 为什么 grad 不是 5 而是 8？(因为 3 + 5 = 8，梯度累加了)
#
# x = ???
# y1 = ???
# ???  # backward
# print(f"After 1st backward: x.grad = {x.grad}")  # 应输出 3.0
# y2 = ???
# ???  # backward
# print(f"After 2nd backward (no zero_grad): x.grad = {x.grad}")  # 应输出 8.0
#
# # 现在清零再做一次
# ???  # zero_()
# y3 = x * 5
# y3.backward()
# print(f"After zero_grad + backward: x.grad = {x.grad}")  # 应输出 5.0


# ============================================================
# Part 7: 综合练习 —— TODO
# ============================================================
print("\n--- Part 7: 综合练习 (TODO) ---")

# TODO 7.1: 模拟一个 mini forward pass
#   - 输入 x: shape (4, 10) — 4 个样本，每个 10 维特征
#   - 权重 W: shape (10, 5) — 10 维输入到 5 维输出, requires_grad=True
#   - 偏置 b: shape (5,)    — 5 维, requires_grad=True
#   - 计算 output = x @ W + b  (矩阵乘法 + 广播加偏置)
#   - 计算 loss = output.sum()  (简化: 直接求和当 loss)
#   - 反向传播
#   - 打印 W.grad 和 b.grad 的 shape
#
# x = torch.randn(4, 10)
# W = ???
# b = ???
# output = ???
# loss = ???
# ???  # backward
# print(f"output shape: {output.shape}")  # 应输出 torch.Size([4, 5])
# print(f"W.grad shape: {W.grad.shape}")  # 应输出 torch.Size([10, 5])
# print(f"b.grad shape: {b.grad.shape}")  # 应输出 torch.Size([5])

# TODO 7.2: 用 torch.no_grad() 做推理
#   - 用上面的 W 和 b
#   - 创建新输入 x_test shape (2, 10)
#   - 在 torch.no_grad() 块中计算 output_test = x_test @ W + b
#   - 验证 output_test.requires_grad 应为 False
#
# x_test = torch.randn(2, 10)
# with torch.no_grad():
#     output_test = ???
# print(f"output_test requires_grad: {output_test.requires_grad}")  # 应输出 False


# ============================================================
# 完成确认
# ============================================================
print("\n" + "=" * 60)
print("完成所有练习后，取消注释并运行验证！")
print("如果所有输出都与预期一致，tensor_playground 通过！")
print("=" * 60)
