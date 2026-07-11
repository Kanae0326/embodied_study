# 单元 1.1 · 环境与骨架

> **篇 1 · 章 1（PyTorch 基础）· 单元 1.1。** 本文件夹即本单元的全部资料，先看本页。
> 章级共享的 `../project_skeleton/`（FashionMNIST 项目骨架）贯穿单元 1.1–1.4：本单元写 `utils.py`，后续单元写 `model.py / dataset.py / train.py`。
> **学习环境：** 全套命令按 **Linux** 编写，WSL 2 或原生 Ubuntu 均可（配置见 `environment_setup.md` 第 0 节）。文中统一用 `~/code/embodied_study` 指代仓库根目录，你的克隆位置不同就替换成实际路径。

---

## 单元目标
从空文件夹独立搭一个最小训练项目骨架，并掌握最基础的环境、shell 与张量/自动微分——摆脱「只会看懂和跑脚本」的状态。

## 核心必做
- 配置 venv/conda、PyTorch、Git、VS Code　→ `environment_setup.md`
- 基本 shell：导航、管道、重定向、`tail -f`、tmux 跑后台　→ `shell_basics.md`（练习 1–6）
- 张量练习：shape / reshape / unsqueeze / broadcast / 索引 / requires_grad / backward　→ `tensor_playground.py`（Part 3–7）
- 自建 `project/`，实现 5 个工具函数 `set_seed / accuracy / save_checkpoint / load_checkpoint / count_parameters`　→ `../project_skeleton/SETUP.md` + `../project_skeleton/utils.py`

> 张量/autograd 的理论讲解见 `study_guide.md`。

## 出关条件
- **标准出关：** 能解释 tensor / batch / epoch / gradient；项目骨架可运行；5 个工具函数通过自测；能逐段说明 `python x.py > out.log 2>&1 &`，会用 tmux 跑后台并 `tail -f` 看日志。
- **弱算力出关：** 与标准一致——本单元 CPU 即可，无 GPU 也能全部完成。

## 交付物
项目骨架 + `tensor_playground.py`（练习完成）+ 通过测试的 `project/utils.py`。

## 可选加分
写一个最小 Dockerfile / 环境复现脚本。

---

## 学习路径（按进度推进，不计时）

> 做完一步、对应自测通过，再进入下一步；快慢由你。
> 你有 VLA 微调经验——张量/autograd 若能直接通过文末自测，可跳过 `study_guide.md` 直接做练习。
> **外部资料的定位：卡壳或自测不过时的查阅来源，不是前置必读**——能过自测就直接往前走。每条资料"何时用、用到什么程度"另见 `study_guide.md` 第六节。

1. **环境就绪** → `environment_setup.md`：跑通文末最终验证脚本，全部 OK。
   - 卡壳时查：[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)（版本/CUDA 对不上时以它为准）
2. **shell 熟练** → `shell_basics.md`：练习 1–6 全做完。
   - 卡壳时查：`man <命令>`；复杂管道命令贴进 [explainshell.com](https://explainshell.com) 逐段解析
3. **张量** → `tensor_playground.py` Part 1–5：TODO 全部取消注释并跑出预期 shape。理论先看 `study_guide.md` 一、五两节。
   - 卡壳时查：[PyTorch Tensors 官方教程](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)（跟敲约 15 分钟）；（可选）想系统补数学直觉看 [D2L Preliminaries 章](https://d2l.ai/chapter_preliminaries/index.html)
4. **自动微分** → `tensor_playground.py` Part 6–7。理论先看 `study_guide.md` 二、三、四节。
   - 卡壳时查：[PyTorch Autograd 官方教程](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)；（可选）想要几何直觉看 [3Blue1Brown 神经网络系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)第 3–4 集
5. **工具函数** → 按 `../project_skeleton/SETUP.md` 在章目录下自建 `project/`，自己写 `utils.py` 五个函数。
   - 卡壳时查：[PyTorch 官方文档](https://pytorch.org/docs/stable/)（按函数名查 `manual_seed` / `save` / `load` / `state_dict` 的用法，不要搜完整实现）

## 验收
逐条对照 `checklist.md`。

---

## utils.py 怎么自测（重要）

**不存在单独的 `tests/test_utils.py`。** 测试**内联在** `../project_skeleton/utils.py` 底部的 `if __name__ == "__main__":` 块里。做法：

1. 把那一整段测试块复制到你自己的 `project/utils.py`（在章目录 `C1_pytorch_basics/` 下）末尾；
2. `cd ~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/project/ && python utils.py`；
3. 看到 `All 5 tests PASSED!` 即通过。

> 别直接 `python ../project_skeleton/utils.py`——skeleton 里 5 个函数都是 `pass`，测试必然失败；要跑在**你自己实现**的版本上。

---

## 本章文件结构

```
C1_pytorch_basics/
├─ U1.1_env_and_skeleton/      ← 本单元
│  ├─ index.md（本页）
│  ├─ study_guide.md   environment_setup.md
│  ├─ shell_basics.md  checklist.md
│  └─ tensor_playground.py
├─ U1.2_min_training_loop/     ← 单元 1.2（训练最小闭环）
│  └─ index.md  study_guide.md  checklist.md
├─ project_skeleton/           ← 章级共享（参考接口，不含实现）
│  ├─ SETUP.md  utils.py       （单元 1.1）
│  └─ model.py  dataset.py  train.py   （单元 1.2）
└─ project/                    ← 你自己建的实现（.py 代码必须入库；data/checkpoints 等产物由 .gitignore 排除）
```

> 注：`project_skeleton/model.py · dataset.py · train.py` 属于**单元 1.2**；本单元（1.1）只建空占位、不实现。单元 1.2 的资料见 `../U1.2_min_training_loop/`。

---

## 单元导航
→ 下一单元：[`../U1.2_min_training_loop/index.md`](../U1.2_min_training_loop/index.md)（训练最小闭环）
