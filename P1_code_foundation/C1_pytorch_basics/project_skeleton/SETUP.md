# project_skeleton —— 使用方式

> 总计划中单元 1.1 的交付是**项目骨架 + tensor_playground**。
> 骨架应该**由你亲手搭起**，而不是直接改我给的文件。

---

## 为什么这么做

如果直接在 `project_skeleton/` 里填空，你永远不会练到下面这些能力：
- 从空文件夹开始建项目结构
- 自己想清楚 `import` 怎么写、模块怎么组织
- 第一次踩"文件放哪、相对导入、`if __name__ == '__main__'`"的坑

**这些能力是后续拆微调仓库、改仿真环境时真正需要的基本功。**

---

## 本单元你要做的

### 1. 建自己的项目目录

```bash
cd ~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/
mkdir -p project/checkpoints
cd project/
touch utils.py model.py dataset.py train.py
```

这 4 个文件单元 1.1 **只需要写 utils.py**。其他 3 个文件先建空的占位；单元 1.2 开始时，确认它们仍为空，再把 skeleton 的 `dataset.py / model.py / train.py` **复制过来覆盖占位**，然后填其中的 TODO（见下方分工表和第 3 节命令）。

### 2. 把 skeleton 当作参考，不要当作答案

`project_skeleton/` 里的 4 个文件是**参考接口**：
- 函数签名（输入输出类型）
- 测试代码（验证你实现是否正确）
- 一句话说明

它们**不包含完整实现**。你要在自己的 `project/utils.py` 里，**不看 skeleton 实现提示**，先自己写 20–30 分钟。卡住再看 skeleton 的提示（不是答案），再卡住才问 AI。

### 3. 用 skeleton 的测试验证自己的实现

```bash
# —— utils.py（单元 1.1）——
# 自己写实现，然后把 skeleton utils.py 底部的 if __name__ == "__main__": 一整段
# 复制到你的 project/utils.py 底部，再运行：
cd ~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/project/
python utils.py
# 看到 "All 5 tests PASSED!" 即通过。
# 注意：直接跑 ../project_skeleton/utils.py 没有意义——里面 5 个函数都是 pass，必然失败。

# —— dataset / model / train（单元 1.2）——
# 仅在三个目标文件仍为空占位时执行；-i 会在覆盖前询问：
cp -i ../project_skeleton/dataset.py ../project_skeleton/model.py ../project_skeleton/train.py .
# 空文件可确认输入 y；若已经写过代码则输入 n，之后手动合并，避免覆盖成果。
# 填完 TODO 后，测试块随文件自带（仍在 project/ 目录下执行）：
python dataset.py   # 实现完成后逐个验证；model.py / train.py 同理
```

---

## 文件的单元 1.1 / 单元 1.2 分工

| 文件 | 单元 1.1 做 | 单元 1.2 做 |
|---|---|---|
| `utils.py` | **全部 5 个函数**（自己写 + 复制测试块自验） | 不动 |
| `dataset.py` | 只建空文件占位 | 从 skeleton 复制后实现 get_dataloaders |
| `model.py` | 只建空文件占位 | 从 skeleton 复制后实现 SimpleCNN |
| `train.py` | 只建空文件占位 | 从 skeleton 复制后实现 train_one_epoch / evaluate / main |

---

## 自查清单

- [ ] 我有一个自己建的 `project/` 目录（不是 `project_skeleton/`）
- [ ] `project/utils.py` 里 5 个函数都是我自己写的
- [ ] `cd project/ && python utils.py` 输出 `All 5 tests PASSED!`
- [ ] `project/*.py` 已被 Git 跟踪并提交；只忽略 data/checkpoints/日志/缓存等生成产物
- [ ] 我没有直接复制 skeleton 里的实现（如果有，还能不看答案重写一遍吗？）

---

## 如果真的卡住了

顺序是：

1. 查 PyTorch 官方文档 (https://pytorch.org/docs/stable/)
2. 看 skeleton 的 docstring 里的一句话提示
3. 问 AI："我在写 set_seed 函数，想知道 PyTorch 里怎么固定 CUDA 随机种子。**不要给我完整代码**，告诉我用哪个 API。"
4. 实在不行才看 AI 给完整实现——但看完之后**关掉它，自己再默写一遍**。
