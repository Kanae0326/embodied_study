# project_skeleton —— 使用方式

> v4 计划中单元 1.1 的交付是**项目骨架 + tensor_playground**。
> 骨架应该**由你亲手搭起**，而不是直接改我给的文件。

---

## 为什么这么做

如果直接在 `project_skeleton/` 里填空，你永远不会练到下面这些能力：
- 从空文件夹开始建项目结构
- 自己想清楚 `import` 怎么写、模块怎么组织
- 第一次踩"文件放哪、相对导入、`if __name__ == '__main__'`"的坑

**这些能力是 M3 拆微调仓库、M9 改仿真环境时真正需要的基本功。**

---

## 本单元你要做的

### 1. 建自己的项目目录

```bash
cd ~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/
mkdir -p project/checkpoints
cd project/
touch utils.py model.py dataset.py train.py
```

这 4 个文件单元 1.1 **只需要写 utils.py**。其他 3 个文件先建空的占位，单元 1.2 再填。

### 2. 把 skeleton 当作参考，不要当作答案

`project_skeleton/` 里的 4 个文件是**参考接口**：
- 函数签名（输入输出类型）
- 测试代码（验证你实现是否正确）
- 一句话说明

它们**不包含完整实现**。你要在自己的 `project/utils.py` 里，**不看 skeleton 实现提示**，先自己写 20–30 分钟。卡住再看 skeleton 的提示（不是答案），再卡住才问 AI。

### 3. 用 skeleton 的测试验证自己的实现

```bash
# 你在 project/utils.py 写好了之后
cd ~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/project/
python ../project_skeleton/utils.py  # 这么跑没用，见下面的说明

# 直接跑 skeleton 的 utils.py 没用：里面 5 个函数都是 pass，测试必然失败；
# 所以推荐做法：把 skeleton utils.py 底部的 if __name__ == "__main__": 那一整段
# 复制到你自己的 project/utils.py 底部。然后：
cd project/
python utils.py
```

或者更干净的做法：把整个 `if __name__ == "__main__":` 块从 skeleton 复制到你自己的 utils.py 底部，就能直接 `python utils.py` 自验。

---

## 文件的单元 1.1 / 单元 1.2 分工

| 文件 | 单元 1.1 做 | 单元 1.2 做 |
|---|---|---|
| `utils.py` | **全部 5 个函数** | 不动 |
| `model.py` | 只建空文件 + 空 class 占位 | 实现 SimpleCNN |
| `dataset.py` | 只建空文件 + 函数签名占位 | 实现 get_dataloaders |
| `train.py` | 只建空文件 | 实现 train_one_epoch / evaluate / main + smoke test |

---

## 自查清单

- [ ] 我有一个自己建的 `project/` 目录（不是 `project_skeleton/`）
- [ ] `project/utils.py` 里 5 个函数都是我自己写的
- [ ] `cd project/ && python utils.py` 输出 `All 5 tests PASSED!`
- [ ] 我没有直接复制 skeleton 里的实现（如果有，还能不看答案重写一遍吗？）

---

## 如果真的卡住了

顺序是：

1. 查 PyTorch 官方文档 (https://pytorch.org/docs/stable/)
2. 看 skeleton 的 docstring 里的一句话提示
3. 问 AI："我在写 set_seed 函数，想知道 PyTorch 里怎么固定 CUDA 随机种子。**不要给我完整代码**，告诉我用哪个 API。"
4. 实在不行才看 AI 给完整实现——但看完之后**关掉它，自己再默写一遍**。
