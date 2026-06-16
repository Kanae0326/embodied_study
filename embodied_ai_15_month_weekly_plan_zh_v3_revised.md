# 具身智能算法专家

## 15 个月学习计划（每周 8–10 小时版）— 修订版 v3

**适配背景：** AI 专业｜会读代码但独立写代码能力弱｜有 VLM/LLM/VLA 微调经验｜机器人学/控制/仿真薄弱

**版本：** 2026 年 3 月 修订版 | 基于 v2 全面补充 2025–2026 前沿技术栈

---

### 修订说明（相比 v2 的主要变化）

| 变更 | 说明 |
|---|---|
| ROS 2 从 3 个月压缩到 2 个月 | 算法岗不需要成为 ROS 专家，省出时间给学习算法 |
| 新增 Diffusion Policy / Flow Matching | 当下机器人策略学习的主流范式，不可缺 |
| LeRobot 升级为主力工具链 | Hugging Face 生态已成为开源机器人学习标准平台 |
| 新增 GPU 并行仿真（Isaac Lab / Genesis） | 只会 MuJoCo 已不够，GPU 并行训练是行业标配 |
| 新增数据工程与遥操作 | 采集、格式化、规模化是算法岗核心技能 |
| 新增 Sim-to-Real 专题 | 仿真到真机的迁移是落地关键 |
| 新增世界模型与人形机器人概览 | 2025–2026 两大前沿方向 |
| 新增 ACT 架构 | Action Chunking Transformer 是 ALOHA/LeRobot 核心策略 |
| 新增多任务 / 语言条件策略学习 | VLA 时代的必备能力 |
| 新增标准 Benchmark 体系 | 从 M10 起就在标准 benchmark 上训练评估 |

---

### 使用方式

每周优先完成"核心任务"，把"本周交付"真正做出来；只有交付落地了，阅读和理解才算转化成能力。

### 建议固定 4 个学习时段

2 小时 × 4 次，优先放在精力好的时段；每次结束前都要把代码、日志或笔记收尾到"下次能继续"的状态。

### 压缩 / 延长规则

如果你连续 6 周都能稳定完成交付，可把 M13–M15 压缩到 12 个月版本；如果某阶段频繁掉队，就允许延长 1 个月，但不要跳过前置能力。

---

## 这份计划的核心任务

> 不是"多跑几个大模型脚本"，而是把你训练成能独立搭起具身智能闭环的人。与 v2 不同的是，v3 同时要求你掌握 2025–2026 年的主流技术栈（Diffusion Policy、LeRobot、GPU 仿真、VLA 系统），而不仅仅是经典技术。

---

### 如何使用这份计划

| 执行原则 | 具体要求 |
|---|---|
| 每周必须有产出 | 最少产出 1 个可验证成果：脚本、实验表、报告、图、视频之一；没有产出就不算完成 |
| AI 是审查员，不是代写员 | 先自己写 20–30 分钟，再用 AI 解释报错、审查逻辑、补充理解；不要整包生成再假装学会 |
| 月度末要复盘 | 固定回答 4 个问题：学到了什么、卡在何处、交付了什么、下月最大的风险是什么 |
| 掉队先补齐 | 若某周完成度低于 60%，下一周先补核心交付，再推进新内容 |
| 项目先于论文 | 每个阶段都以"能跑起来、能解释、能复现"为先，论文阅读用来提升理解和方向感 |

---

### 每周固定节奏（8–10 小时）

| 编码实现 | 理论补强 | 调试 / 复盘 | 文档 / Git |
|---|---|---|---|
| 4–5 小时：写代码、改脚本、跑实验 | 2 小时：读笔记、补原理、画图理解 | 1.5–2 小时：排 bug、看日志、写复盘 | 0.5–1 小时：整理 README、提交 Git、清理目录 |

---

## 15 个月路线总览

| 学习周期 | 15 个月 / 60 周 | 时间预算 | 每周 8–10 小时 |
|---|---|---|---|
| 月度节奏 | 每月 4 周；每周有交付 | 成果形式 | 代码仓库 / 实验记录 / 报告 / 视频 |
| 主线目标 | 代码自立 → 机器人闭环 → 仿真学习 → VLM/VLA 系统 | 适用原则 | 先能独立实现，再追前沿模型 |

---

| 阶段 | 月份 | 核心能力 | 阶段交付 |
|---|---|---|---|
| 阶段 1 | M1–M3 | Python / PyTorch 自立；训练脚本重写；VLM/VLA 微调脚本拆解 | 最小训练模板 + 微调脚本方法论 |
| 阶段 2 | M4–M6 | 数学、空间表示、运动学、控制 | 运动学工具 + 控制实验对比 |
| 阶段 3 | M7–M8 | ROS 2、TF/URDF、MoveIt 2 基础 _(压缩)_ | MoveIt 规划 demo + ROS 调试能力 |
| 阶段 4 | M9–M12 | 仿真生态、IL/BC/Diffusion Policy、RL、Sim-to-Real、数据工程 _(大幅扩展)_ | 仿真模板 + Diffusion Policy baseline + 数据管线 |
| 阶段 5 | M13–M15 | 感知前端、VLM/VLA 闭环、世界模型、Benchmark 与作品集 _(增强)_ | 可展示系统 + 报告 + 视频 |

---

## 阶段 1：代码自立与深度学习工程（M1–M3）

> 本阶段与 v2 基本一致，这是你最需要先补齐的能力。

---

### 第 1 个月｜Python / PyTorch 项目自立

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 从空文件夹独立搭建一个最小深度学习训练项目，摆脱"只能看懂和跑脚本"的状态 | 一个可训练、可验证、可保存 best checkpoint 的 FashionMNIST 小项目；配 README、曲线图和 1 页训练流程说明 | 不要追求高指标；本月唯一目标是把训练代码自己写出来 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 环境与骨架 | • 配置 Python 虚拟环境、PyTorch、Git、VS Code；建立固定目录结构（train.py / model.py / dataset.py / utils.py）<br>• 完成张量练习：shape、reshape、unsqueeze、broadcast、索引、requires_grad、loss.backward()<br>• 实现 5 个小工具函数：set_seed、accuracy、save_checkpoint、load_checkpoint、count_parameters | 交付：项目骨架 + tensor_playground 脚本<br>验收：能解释 tensor / batch / epoch / gradient；能独立运行并保存文件 |
| W2 | 训练最小闭环 | • 用 torchvision 加载 FashionMNIST，切 train / val，确认 batch 形状和标签类型<br>• 自己写一个小 CNN 和 forward；完成 train_one_epoch 与 evaluate<br>• 跑通 5–10 个 epoch，记录最基础 loss / accuracy 日志 | 交付：第一次完整训练日志<br>验收：能说清 forward、loss、backward、optimizer.step 的先后关系 |
| W3 | 工程化 1.0 | • 加入 argparse 或 yaml 配置；支持 learning rate、batch size、epochs、save path<br>• 补齐 checkpoint 保存逻辑：latest 与 best；输出训练曲线图<br>• 整理 README：环境、运行命令、目录结构、常见报错 | 交付：best.pth + loss/acc 曲线图<br>验收：从命令行改参数即可复现实验；自己能加载 best 模型做验证 |
| W4 | 最小实验能力 | • 做 3 组小实验：不同 learning rate、不同 batch size、dropout 或 weight decay 二选一<br>• 写 1 页总结：一次 train step 到底发生了什么；为什么会过拟合<br>• 复盘本月最常见的 5 个 bug，并写出对应排查步骤 | 交付：小实验表 + 1 页总结<br>验收：能从空目录独立重建这个小项目；至少独立修掉 5 个 bug |

---

### 第 2 个月｜训练脚本理解与改写

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 从会写基础 train loop，升级到能读懂、缩写、重构一个中等复杂度训练脚本 | 一个更规范的训练模板仓库；外加一份你亲手画出的"训练脚本地图" | 不要把框架包装当理解；任何现成脚本都必须能拆成数据、模型、损失、优化四层 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 工程规范 | • 为上月项目加入 config、日志目录、固定随机种子、结果输出目录、实验命名规则<br>• 学习基本调试方法：print shape、assert、查看 grad、检查 NaN、定位数据问题<br>• 建立实验记录模板：实验目标、改动点、结果、解释、下一步 | 交付：可复用训练模板 1.0<br>验收：新建任务时，30 分钟内能搭起同样结构；定位问题不再只靠猜 |
| W2 | 模块级理解 | • 把一个简单 CNN 项目拆成 data / model / train / eval / utils 五层，理清相互调用关系<br>• 补齐核心概念：scheduler、weight decay、early stopping、gradient clipping、mixed precision（先理解不必全用）<br>• 挑一个现成开源训练脚本，手动画出数据流和函数调用图 | 交付：训练脚本地图（1 张图）<br>验收：能看懂主函数入口；知道每个模块为什么存在而不是只会运行 |
| W3 | 最小 Transformer | • 用 PyTorch 自己实现 embedding、self-attention、MLP、residual、layer norm 的最小版本<br>• 做一次前向 shape 跟踪：输入 token → attention → 输出 logits<br>• 写一份 1–2 页说明：attention 为什么能工作；与 CNN 的差别是什么 | 交付：mini_transformer.py<br>验收：能独立解释 Q、K、V、attention score、残差连接和 layer norm 的作用 |
| W4 | 脚本改写练习 | • 把一个现成训练脚本缩写成最小可运行版，删除多余框架包装和不懂的辅助模块<br>• 亲手改一个功能点：loss、数据增强、评估指标三选一<br>• 写复盘：你删掉了什么、保留了什么、为什么 | 交付：最小化重写脚本<br>验收：不用 AI 逐行解释，也能讲清主干流程；能独立改一个功能点并验证结果 |

---

### 第 3 个月｜VLM / LLM / VLA 脚本拆解与微调理解

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 把你已有的微调经验从"会跑脚本"升级成"理解数据流、参数更新和显存策略" | 一份微调脚本解剖文档；一次你自己改过并验证过的微调实验 | 不要追大模型和大数据；只做你能完整讲清楚的数据流和训练逻辑 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 脚本解剖 | • 选择一个你已经跑过的 LoRA / PEFT 微调脚本，标出入口文件、数据整理、模型加载、可训练参数设置、训练循环<br>• 追踪一次 batch：原始样本 → tokenizer / processor → collator → model.forward → loss<br>• 写注释版代码地图，只保留真正关键的 20% 文件 | 交付：微调脚本结构图<br>验收：能讲清每个 batch 的数据长什么样；知道哪些参数在更新，哪些被冻结 |
| W2 | 参数高效微调原理 | • 补齐 LoRA、rank、target modules、adapter merge、梯度累积、mixed precision 的基本概念<br>• 做一次参数统计：全量微调 vs LoRA 可训练参数数目、显存占用预估<br>• 检查训练日志：学习率、loss、grad norm、显存峰值分别说明什么 | 交付：PEFT 原理笔记<br>验收：能解释为什么 LoRA 能省参数；能判断日志里哪些信号是不正常的 |
| W3 | 自己动手改一处 | • 在现有微调脚本中主动改一个关键点：数据整理器、loss 计算、评估逻辑或保存逻辑<br>• 设计一个小对照实验，确认改动是否生效<br>• 记录改动前后差异和你自己的判断依据 | 交付：一次可复现实验记录<br>验收：你能解释"我改的不是表面参数，而是训练逻辑中的一环" |
| W4 | 形成方法论 | • 写《我如何阅读一个大模型微调仓库》1–2 页方法笔记<br>• 整理一份常见故障清单：显存爆、loss 不降、数据格式错、评估不一致、保存失败<br>• 把前 3 个月的所有代码仓库做一次清理：目录、README、requirements、运行脚本 | 交付：方法笔记 + 代码整理版仓库<br>验收：之后再遇到新仓库，你有一套固定的拆解顺序而不是靠试运气 |

---

## 阶段 2：数学、运动学与控制（M4–M6）

> 本阶段与 v2 基本一致，是具身智能的必要地基。

---

### 第 4 个月｜数学补强与空间表示

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 建立具身智能所需的线性代数、优化和三维空间表达直觉 | 一个 2D/3D 坐标变换可视化 notebook；一份空间表示速查表 | 不要追求推导完美；先把"能用、能画、能解释"建立起来 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 线性代数回炉 | • 复习矩阵乘法、特征值、SVD、投影、正交基；只抓和视觉/机器人相关的核心概念<br>• 用 numpy 写几个最小例子：二维旋转、坐标变换、最小二乘拟合<br>• 把自己容易忘的公式写成 1 页速查表 | 交付：线代速查表<br>验收：看到矩阵表达式时能解释其几何意义 |
| W2 | 概率与优化 | • 补齐高斯分布、贝叶斯视角、最大似然、Adam / SGD / momentum 的区别<br>• 用图示理解学习率、收敛、振荡、梯度爆炸/消失<br>• 写 2 个小实验：不同学习率和不同优化器对收敛曲线的影响 | 交付：优化实验图<br>验收：能解释为什么训练会抖动、发散或收敛慢 |
| W3 | 三维空间语言 | • 学习欧拉角、旋转矩阵、四元数、齐次变换、位姿 pose 的定义<br>• 用 Python 实现旋转与平移组合；验证先旋转后平移和先平移后旋转的差别<br>• 理解坐标系之间的变换链：world → base → camera → end-effector | 交付：pose 变换脚本<br>验收：看到一个 4×4 变换矩阵时能说出它表示什么 |
| W4 | 可视化闭环 | • 做一个 notebook：输入位姿参数，画出坐标系和末端姿态变化<br>• 整理《具身智能必备数学最小包》：哪些真正要熟，哪些可后补<br>• 做本月复盘：自己最薄弱的是概率、优化还是空间表示 | 交付：坐标变换可视化 notebook<br>验收：能自己写 pose compose / inverse；后续学习运动学时不再被坐标系劝退 |

---

### 第 5 个月｜机器人运动学

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 理解 link、joint、末端执行器、工作空间、正逆运动学和 Jacobian | 一个交互式机械臂运动学小程序；配说明文档和测试样例 | 本月先做"会算、会画、会解释"，暂不深究复杂动力学 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 机器人语言入门 | • 理解机械臂的 link / joint / DOF / base / end-effector / workspace<br>• 阅读一个简单 2-link 与 6-DOF 机械臂示意，区分关节空间和笛卡尔空间<br>• 建立术语表，后续文档和代码统一用词 | 交付：机器人术语表<br>验收：能看懂一个机械臂示意图并指出基座、关节、连杆与末端 |
| W2 | 正运动学 | • 用二维两连杆开始，自己写 forward kinematics；输入关节角，输出末端位置<br>• 逐步扩展到 3D 简化机械臂，理解变换连乘<br>• 画可达工作空间，区分可达与不可达点 | 交付：forward kinematics 脚本<br>验收：给一组关节角，你能算并画出末端位置 |
| W3 | 逆运动学与 Jacobian | • 实现二维机械臂解析或数值逆解；理解多解、无解和奇异位形<br>• 学习 Jacobian 的几何直觉：关节速度如何映射到末端速度<br>• 做一个数值实验：靠近奇异位形时会发生什么 | 交付：inverse kinematics + Jacobian notebook<br>验收：知道 IK 不一定有唯一解；知道奇异位形为何会带来控制问题 |
| W4 | 封装成工具 | • 做一个交互式小程序：点选目标点，机械臂给出 IK 结果并画轨迹<br>• 加入不可达点和多解提示；写清楚你的处理逻辑<br>• 复盘：运动学里最容易混淆的 5 个概念 | 交付：运动学小程序<br>验收：后面学 MoveIt 和机械臂规划时，你已经有自己的直觉 |

---

### 第 6 个月｜控制基础

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 建立状态、误差、反馈、稳定性和经典控制器的直觉 | PID / LQR 小实验对比报告；至少一个可复现控制器 demo | 控制不是背公式；重点是理解为什么会稳、抖、发散、超调 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 反馈控制直觉 | • 补齐状态、控制输入、目标、误差、反馈、闭环系统、稳定性等基本词汇<br>• 用最简单的一维系统或小车模型说明"没有反馈"和"有反馈"的区别<br>• 做曲线图：误差随时间如何变化 | 交付：反馈控制概念图<br>验收：能用自己的话解释为什么反馈控制比开环更稳 |
| W2 | PID 动手 | • 自己写 PID 控制器，先在一维位置跟踪上测试，再调 P、I、D 的大小<br>• 观察欠调、超调、振荡和稳态误差分别长什么样<br>• 整理 PID 调参顺序和常见坑 | 交付：PID demo + 调参笔记<br>验收：看到响应曲线时，能判断问题更像是 P、I 还是 D 没调好 |
| W3 | 状态空间与 LQR | • 理解状态空间表示；用一个小系统写出 x_{t+1} = Ax_t + Bu_t 的形式<br>• 学习 LQR 的目标：在状态偏差和控制代价之间做平衡<br>• 在经典控制环境中做 PID vs LQR 的简单对比 | 交付：LQR 对比实验<br>验收：知道 PID 和 LQR 的使用语境差异 |
| W4 | 与学习策略对照 | • 在 Gymnasium 的简单控制环境上跑通一个经典控制器；再读一个 RL baseline<br>• 写 1 页比较：经典控制 vs 学习策略各擅长什么<br>• 记录控制视角会如何影响你以后理解机器人策略 | 交付：控制对比小报告<br>验收：后面遇到 RL / policy 时，你能分清策略学习和控制执行分别负责什么 |

---

## 阶段 3：ROS 2 与机器人软件栈（M7–M8）

> **v3 变化：从 3 个月压缩到 2 个月。** 算法岗需要理解 ROS 2 生态但不需要成为 ROS 系统工程师。省出的 4 周分配给仿真平台和学习算法。

---

### 第 7 个月｜ROS 2 入门 + TF / URDF / RViz

_（合并原 v2 的 M7 和 M8 为一个月）_

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 理解 ROS 2 节点通信、坐标系管理和机器人模型描述，建立调试能力 | 一个带 TF 树、URDF 模型和 RViz 可视化的最小机器人项目 | 不要追求精通每个 ROS 2 子系统；目标是"能用、能调、能读懂" |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 环境与节点通信 | • 准备 Ubuntu + ROS 2 环境；熟悉 package、workspace、colcon build<br>• 做一个 talker / listener 最小样例；练习 ros2 topic list / echo / hz<br>• 理解 node、topic、publisher、subscriber；把消息流画成图 | 交付：ROS 2 安装笔记 + topic demo<br>验收：能独立新建 workspace、build、运行并调试 topic 通信 |
| W2 | Service / Action / Launch | • 学习 service 和 action 的使用场景差异；分别做一个最小 demo<br>• 练习 launch 文件和参数管理；尝试录制 rosbag<br>• 记录 service vs topic vs action 各自适用场景 | 交付：service + action + launch demo<br>验收：知道什么时候该用 topic，什么时候该用 service 或 action |
| W3 | TF 树与 URDF | • 理解 frame、父子关系、静态变换和动态变换；画 world / base / camera / ee 树<br>• 手写一个简化 URDF（2-link 或小车）；在 RViz 中查看<br>• 把第 4 个月的坐标变换直觉与 TF 连接起来 | 交付：TF 关系图 + 最小 URDF 模型<br>验收：看到变换错位时，能从 frame 定义和父子链路入手排查 |
| W4 | RViz 整合项目 | • 做一个带 URDF、TF、RViz 的小整合项目<br>• 学习 RViz 显示项：TF、Marker、RobotModel、Image 的基本用途<br>• 建立调试顺序：节点正常吗、topic 正常吗、TF 正常吗、可视化正常吗<br>• 写 README：启动顺序、消息说明、调试命令 | 交付：ROS 2 可视化项目<br>验收：为下个月进入 MoveIt 机械臂规划做好模型与坐标系准备 |

---

### 第 8 个月｜MoveIt 2 基础 + 仿真平台全景

_（v3 变化：MoveIt 2 压缩到前 2 周；后 2 周新增仿真平台全景）_

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 理解运动规划闭环 + 了解当下仿真生态的格局和选型依据 | 一个 MoveIt 规划 demo + 一份仿真平台对比分析 | MoveIt 重在理解概念而非精通配置；仿真平台重在全景视野 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | MoveIt 2 上手 | • 安装并跑通 MoveIt 2 官方 demo；看懂 planning scene、planning group、end-effector<br>• 在 RViz 中做几次 pose goal / joint goal 规划<br>• 理解 SRDF、kinematics 配置、planning pipeline 基本结构<br>• 向 planning scene 加入桌面、障碍物 | 交付：MoveIt 2 demo + 带障碍物场景<br>验收：能分清 joint goal 和 pose goal；知道规划失败的常见原因 |
| W2 | Pick-and-Place 与闭环 | • 理解抓取链路：目标位姿 → 接近 → 闭合 → 抬起 → 放置<br>• 做一个简化 pick-and-place demo；录屏<br>• 总结 MoveIt 在具身智能中的定位：经典规划 vs 学习策略各负责什么 | 交付：机械臂 pick-and-place demo<br>验收：能把运动学、规划、碰撞和执行链从头讲到尾 |
| W3 | 仿真平台全景 | **[v3 新增]**<br>• 调研当前主流仿真平台：MuJoCo、Isaac Lab（NVIDIA）、Genesis、SAPIEN/ManiSkill<br>• 理解 GPU 并行仿真的意义：为什么现代 RL/IL 训练都需要上千并行环境<br>• 对比各平台特点：物理精度、速度、可微分性、渲染、生态 | 交付：仿真平台对比表<br>验收：能解释为什么不同场景选不同仿真器；知道 Isaac Lab 已取代 IsaacGymEnvs |
| W4 | Isaac Lab 初体验 | **[v3 新增]**<br>• 安装 Isaac Lab 或 Genesis；跑通一个最小机械臂示例<br>• 理解 GPU 并行环境的接口：vec_env、parallel rollout、batch observation<br>• 对比同一任务在 MuJoCo 和 Isaac Lab/Genesis 下的速度差异<br>• 写笔记：哪个平台更适合你后续的 IL/RL 实验 | 交付：GPU 仿真初体验记录<br>验收：后续做 IL/RL 时不再只限于单环境 MuJoCo |

---

## 阶段 4：仿真与机器人学习（M9–M12）

> **v3 重大变化：从 3 个月扩展到 4 个月。** 新增 Diffusion Policy、Flow Matching、ACT、Sim-to-Real、数据工程。这是你与其他候选人拉开差距的关键阶段。

---

### 第 9 个月｜仿真深入（以 MuJoCo 为主）

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 建立"任务—状态—动作—奖励—动力学—回合"这套仿真语言 | 一个你亲手改过 observation / reward / reset 的仿真任务模板 | 别同时学多个仿真器；本月在 MuJoCo 上深入，但保持对 Isaac Lab 的了解 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 仿真基本概念 | • 安装并跑通 MuJoCo 基础示例；看懂模型文件、关节、执行器、传感器的角色<br>• 梳理 observation、action、reward、done、reset 的含义<br>• 选择一个简单任务（reach / push / balance）作为本月主线 | 交付：仿真环境笔记<br>验收：不再把环境当黑盒；知道每一步交互接口返回什么 |
| W2 | 改任务定义 | • 修改 observation 或 reward 中的一项，验证行为变化<br>• 阅读并修改 reset 逻辑：初始位置、目标位置、随机化范围<br>• 保存几段 rollout，观察不同策略下系统表现 | 交付：自定义环境改动版<br>验收：知道 reward 设计会如何影响策略方向 |
| W3 | 渲染、日志与 Domain Randomization | • 学会录制视频、输出关节/状态曲线、保存实验日志<br>• 加入 domain randomization：目标位置、初始姿态、摩擦、外观随机化<br>• 记录"我改了什么 → 表现怎么变"的实验因果链 | 交付：视频 + 实验日志<br>验收：以后进入 RL/IL 时，能判断问题出在环境、奖励还是策略 |
| W4 | 仿真模板 + Benchmark 初识 | • 整理最小可复用仿真任务模板：环境创建、rollout、日志、可视化<br>• **[v3 新增]** 浏览 ManiSkill / LIBERO / RLBench 的任务设计，理解标准 benchmark 的结构<br>• 写 README，说明 observation/action/reward 设计 | 交付：仿真任务模板 1.0<br>验收：别人能基于你的环境模板继续做 IL/RL 训练 |

---

### 第 10 个月｜模仿学习：BC → Diffusion Policy → ACT

_（v3 重大增强：从纯 BC 扩展到覆盖 Diffusion Policy、ACT 和 LeRobot）_

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 掌握三种主流模仿学习范式（BC、Diffusion Policy、ACT），用 LeRobot 作为统一工具链 | Diffusion Policy baseline + ACT 对比实验 + LeRobot 使用报告 | 先在一个任务上打通全流程，不要同时尝试多个环境 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | BC 基础 + LeRobot 上手 | **[v3 更新]**<br>• 理解 trajectory、observation-action pair、expert policy、compounding error<br>• **安装 LeRobot，跑通一个内置教程**（推荐 pusht 或 LIBERO 子任务）<br>• 理解 LeRobot 的数据集格式（LeRobotDataset v3）和 policy 接口<br>• 训练一个最小 BC baseline，记录 loss 和成功率 | 交付：LeRobot BC baseline<br>验收：能解释 LeRobot 的 dataset → policy → train → eval 流程 |
| W2 | Diffusion Policy | **[v3 新增]**<br>• 学习 Diffusion Policy 的核心思想：将策略表示为条件去噪扩散过程<br>• 理解为什么 Diffusion 天然处理多模态动作分布（比 MSE loss 的 BC 更好）<br>• 在 LeRobot 中训练一个 Diffusion Policy；与 W1 的 BC 做对比<br>• 画 rollout，观察两者在多模态场景中的行为差异 | 交付：Diffusion Policy 实验 + BC 对比<br>验收：能解释 diffusion 为什么比纯 BC 更适合机器人操控 |
| W3 | ACT (Action Chunking Transformer) | **[v3 新增]**<br>• 学习 ACT 的核心概念：action chunking（一次预测多步动作）+ CVAE 结构<br>• 理解 ACT 为什么能减少 compounding error<br>• 在 LeRobot 中训练一个 ACT policy；与 BC、Diffusion Policy 做三方对比<br>• 写实验对比表：成功率、轨迹平滑度、训练稳定性 | 交付：ACT 实验 + 三方对比表<br>验收：能分清 BC / Diffusion Policy / ACT 各自的优势和适用场景 |
| W4 | Flow Matching + 形成模板 | **[v3 新增]**<br>• 学习 Flow Matching 的基本原理：与 Diffusion 的区别（学速度场 vs 学噪声）<br>• 理解 π0 为什么选择 Flow Matching（更简单、更快、更稳定）<br>• 整理 IL 训练模板：dataset / policy / train / eval / rollout / metrics<br>• 写 1–2 页总结：四种策略（BC / Diffusion / ACT / Flow Matching）的适用边界 | 交付：IL 模板仓库 + 策略对比总结<br>验收：之后做 VLA 策略时，你能准确选择合适的 policy head |

---

### 第 11 个月｜强化学习 + Sim-to-Real

_（v3 变化：压缩 RL 到前 2 周，后 2 周新增 Sim-to-Real 专题）_

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 理解策略优化和奖励设计；掌握仿真到真机迁移的核心方法 | RL baseline + BC/RL/Diffusion 对比报告 + Sim-to-Real 方法笔记 | RL 先把一个算法跑通跑稳；Sim-to-Real 重在建立方法框架 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | RL 基础与训练 | • 补齐 MDP、policy、value、advantage、exploration 等概念<br>• 用现成实现跑通 PPO 或 SAC baseline；重点看训练循环与日志<br>• 记录 episode return、成功率、长度变化 | 交付：RL baseline 结果<br>验收：训练失败时，能优先排查奖励、观测、动作范围 |
| W2 | 奖励设计 + 方法对照 | • 对 reward 做 1–2 次有意识修改；比较训练稳定性和行为差异<br>• 记录 reward hacking 和奇怪行为<br>• 写 BC vs RL vs Diffusion Policy 的综合对比：数据成本、训练难度、探索、鲁棒性<br>• 整理 RL 最小模板 | 交付：方法对照报告<br>验收：你能为一个新任务选择合适的学习范式并说明理由 |
| W3 | Sim-to-Real 方法论 | **[v3 新增]**<br>• 系统学习 Sim-to-Real 的核心挑战：physics gap、visual gap、dynamics gap<br>• 理解 Domain Randomization 的系统化方法（不只是随机化几个参数）<br>• 学习 System Identification 的基本概念：测量真实系统参数来校准仿真<br>• 了解新兴方法：可微分仿真优化、生成模型视觉域适应 | 交付：Sim-to-Real 方法论笔记<br>验收：能画出 sim-to-real 的 gap 分析框架；知道 DR 和 SysID 各自适用场景 |
| W4 | Sim-to-Real 实践 | **[v3 新增]**<br>• 在你的仿真任务上设计一套 Domain Randomization 方案<br>• 评估不同 DR 强度对策略鲁棒性的影响<br>• 了解 real-world deployment 的安全约束：力限制、速度限制、碰撞检测<br>• 写复盘：如果你有真机，你的迁移方案是什么 | 交付：DR 实验 + 迁移方案设计<br>验收：面试时能回答"你的仿真策略如何部署到真机"这类问题 |

---

### 第 12 个月｜数据工程与多任务策略学习

_（v3 完全新增的一个月）_

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 掌握机器人学习的数据全链路：采集、格式化、规模化；建立多任务/语言条件策略能力 | 数据管线文档 + 多任务策略实验 | 数据工程不性感但极其关键；这是很多人忽略的竞争力 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 数据采集与遥操作 | **[v3 新增]**<br>• 了解主流遥操作系统：ALOHA（双臂低成本）、UMI（通用操控接口）、VR 遥操、Gello<br>• 理解数据采集的工程挑战：同步、延迟、标注、质量筛选<br>• 如果有条件，在仿真中模拟一个简单遥操作数据采集流程<br>• 研究 Open X-Embodiment（22 种机器人、100 万+ episode）的数据组织方式 | 交付：遥操作系统调研 + 数据采集笔记<br>验收：能解释从遥操作采集到 policy 训练的完整数据链路 |
| W2 | 数据格式与数据集管理 | **[v3 新增]**<br>• 学习 LeRobot Dataset v3 格式：episode 组织、多模态存储（图像、状态、动作）、元数据<br>• 了解其他常见格式：RLDS、HDF5<br>• 浏览 DROID 数据集（76K 轨迹、564 场景）的设计<br>• 把你之前的仿真数据转换成 LeRobot 格式，上传到 HuggingFace Hub | 交付：数据格式化管线<br>验收：能把任意仿真 rollout 转成标准格式供 LeRobot 训练使用 |
| W3 | 多任务与语言条件策略 | **[v3 新增]**<br>• 理解多任务训练的挑战：任务间干扰、数据平衡、条件化方式<br>• 在 LIBERO 或 ManiSkill 上做一个语言条件策略训练实验<br>• 理解 language embedding 如何注入 policy 网络<br>• 对比单任务 vs 多任务 vs 语言条件策略的性能差异 | 交付：语言条件策略实验<br>验收：能解释为什么 VLA 需要语言条件化；知道多任务训练的主要技巧 |
| W4 | Benchmark 体系与评估 | **[v3 新增]**<br>• 系统了解主流 benchmark：LIBERO、ManiSkill、RLBench、CALVIN、SimplerEnv<br>• 理解评估协议：成功率、泛化性（新场景/新物体/新指令）、样本效率<br>• 选择 1–2 个 benchmark 跑一次完整评估<br>• 写总结：不同 benchmark 测试的核心能力是什么 | 交付：Benchmark 评估报告<br>验收：之后你的实验结果可以放在标准 benchmark 下讨论 |

---

## 阶段 5：系统整合与前沿（M13–M15）

> v3 增强了 VLA 深度、新增世界模型和人形机器人概览。

---

### 第 13 个月｜感知前端：检测 / 分割 / 位姿 / 点云

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 建立从视觉输入到可执行目标位姿的前端能力 | 一个从 RGB 或 RGB-D 到目标位置/位姿的最小前端模块 | 不要贪多；核心是让感知结果能接到后面的规划或策略中 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 相机模型 | • 理解相机内参、外参、投影、深度图、像素坐标到三维坐标的关系<br>• 做一个小练习：已知深度和内参，把图像点恢复到相机坐标系<br>• 连接前面的 TF / pose 直觉，理解 camera frame 到 world frame | 交付：相机几何 notebook<br>验收：知道"检测到一个框"还不等于"知道机器人该抓哪儿" |
| W2 | 检测与分割 | • 复习目标检测、实例分割任务形式；在一个小数据集或开源脚本上跑通推理<br>• 把输出结果转成你后续可用的数据结构：类别、置信度、框、mask<br>• 整理检测前端与机器人任务之间的接口定义 | 交付：检测/分割推理模块<br>验收：能把感知结果保存成结构化对象 |
| W3 | 位姿估计与点云 | • 学习 6D pose、点云、表面法向、抓取点候选的基本概念<br>• 用深度图或仿真相机做一个最小三维目标定位示例<br>• **[v3 新增]** 了解触觉感知的概念：力/触觉信号如何辅助操控决策<br>• 记录感知误差会如何传导到规划或执行 | 交付：目标位姿估计最小示例<br>验收：能说清 2D 检测、3D 定位和可执行抓取位姿之间的差别 |
| W4 | 前端整合 | • 把检测/定位结果输出给一个下游模块：规划器、控制器或策略网络<br>• 完成一次端到端测试：看见目标 → 生成目标点/位姿 → 下游执行<br>• 写 1 页总结：哪些误差最影响结果 | 交付：感知前端 1.0<br>验收：具身智能系统里，你已经拥有"感知入口" |

---

### 第 14 个月｜VLM / VLA 系统整合（深度版）

_（v3 大幅增强：增加 VLA 架构全景、实际训练/微调、多场景评估）_

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 发挥你的 VLM/VLA 微调优势，构建完整的语言驱动机器人闭环系统 | 一个"语言指令 → 目标理解 → 策略执行"的可演示系统 | 先让系统能用，再追求泛化；异常处理和回退机制很重要 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | VLA 架构全景 | **[v3 新增]**<br>• 系统梳理当前主要 VLA 模型：π0/π0.5（Physical Intelligence）、OpenVLA（Stanford）、GR00T N1（NVIDIA）、SmolVLA（HuggingFace）、Gemini Robotics（Google）<br>• 对比架构差异：backbone 选择、action head（autoregressive vs diffusion vs flow matching）、训练数据来源<br>• 理解 dual-system 架构（System 1 快速动作 + System 2 慢速推理）的设计思想<br>• 选择一个开源 VLA（推荐 π0 或 SmolVLA）作为本月主线 | 交付：VLA 架构对比图<br>验收：能说清主流 VLA 的核心设计差异和各自优劣 |
| W2 | VLA 微调与语言 Grounding | • 用 LeRobot / OpenPI 对选定的 VLA 做一次 fine-tuning（在你的仿真任务上）<br>• 把语言指令映射到目标类别、空间关系或操作约束<br>• 在简单场景中实现对象选择：颜色、类别、位置等<br>• 记录 grounding 失败的类型和原因 | 交付：VLA 微调实验 + 语言 grounding 模块<br>验收：你不再只是"跑脚本微调"，而是理解数据流和训练逻辑后主动调整 |
| W3 | 端到端闭环 | • 把 VLA 连接到感知前端和下游执行<br>• 完成最小闭环：语言 → VLA → 动作执行 → 环境反馈<br>• 在 2–3 个场景变体中测试（不同物体位置、不同指令表述）<br>• 记录系统延迟、失败点与接口不匹配问题 | 交付：VLA 闭环 demo<br>验收：至少能在多个场景变体中完成任务 |
| W4 | 系统稳态化 | • 写 1–2 页架构说明：高层语义、感知、规划、控制各自负责什么<br>• 补充异常处理：目标缺失、识别错误、规划失败时如何回退<br>• **[v3 新增]** 对比你的系统与模块化方案（VLM 任务分解 + 经典规划执行）的优劣<br>• 整理演示视频和结果表 | 交付：可展示的多模态机器人 demo<br>验收：你已经从"只会微调模型"走到"会把模型接进机器人系统" |

---

### 第 15 个月｜前沿方向 + Benchmark + 作品集

_（v3 增强：新增世界模型和人形机器人概览）_

| 本月目标 | 本月交付 | 风险提醒 |
|---|---|---|
| 像研究者和求职者一样评估系统、了解前沿、形成可展示成果 | 一套最终作品集：代码仓库、技术报告、演示视频、实验复盘 | 不要等作品完美才整理；本月重点是收敛、评估、表达和打包 |

| 周次 | 本周主线 | 核心任务 | 本周交付 / 验收 |
|---|---|---|---|
| W1 | 世界模型 + 人形机器人概览 | **[v3 完全新增]**<br>• 了解世界模型的概念：预测环境状态变化，用于规划和数据生成<br>• 浏览 NVIDIA Cosmos、Genie 3、LAC-WM 等代表性工作<br>• 了解人形机器人的基本框架：whole-body control、双足步态、上半身操控<br>• 了解 GR00T N1、Helix、Tesla Optimus 的技术路线<br>• 写 1 页总结：这两个方向与你的技能栈如何对接 | 交付：前沿方向调研报告<br>验收：面试时能讨论世界模型和人形机器人的发展趋势 |
| W2 | Benchmark 对照实验 | • 选择一个 benchmark 路径：LIBERO 子任务、SimplerEnv 或自定义任务套件<br>• 至少跑一个 baseline 和一个你自己的改进版；保留相同评估设置<br>• 画结果表、失败分类图或关键案例图<br>• 开始撰写技术报告：问题、系统结构、实验、失败分析 | 交付：结果表 + 报告初稿<br>验收：能说清你的改进到底有没有价值 |
| W3 | 作品集包装 | • 整理主仓库：环境说明、安装、运行、结果复现、目录、模型权重说明<br>• 录制演示视频；准备 5–8 页项目讲解 slide<br>• 把 15 个月的关键产出归档成一个 portfolio 文件夹<br>• **[v3 新增]** 准备技术栈关键词清单：Diffusion Policy / Flow Matching / ACT / LeRobot / Isaac Lab / VLA / Sim-to-Real | 交付：作品集 1.0<br>验收：陌生人打开仓库能理解项目做了什么 |
| W4 | 论文复现与面试准备 | • 选择一篇与你主线接近的 VLA / 机器人学习论文，复现一个核心实验<br>• 做一次自我答辩：研究动机、方法、实验设计、失败点、下一步<br>• **[v3 新增]** 准备面试高频问题：<br>  — Diffusion Policy vs BC vs RL 的优劣<br>  — Sim-to-Real 如何设计<br>  — VLA 架构选型依据<br>  — 数据采集与规模化策略<br>• 写最终复盘：我的强项、短板、接下来 6 个月如何继续深化 | 交付：最终报告 + 自我答辩提纲<br>验收：你已经具备申请具身智能算法岗/研究岗的第一版作品证明 |

---

## 里程碑与作品集清单

| 时间点 | 你应该已经具备的能力 | 可展示证据 |
|---|---|---|
| M2 结束 | 能从空项目写出训练脚本，独立保存/加载模型并做最小实验 | train.py、best.pth、曲线图、README |
| M3 结束 | 能拆解一个微调仓库，改动训练逻辑而不是只改参数 | 微调脚本结构图、改动实验记录 |
| M6 结束 | 理解 pose、IK、Jacobian、PID/LQR，对控制和运动学有直觉 | 运动学工具、控制对比报告 |
| M8 结束 | 能在 ROS 2 + MoveIt 2 中完成基础规划；了解 GPU 仿真生态 | MoveIt demo + 仿真平台对比表 |
| M10 结束 | **掌握 BC / Diffusion Policy / ACT 三种策略；熟练使用 LeRobot** | Diffusion Policy baseline + 三方对比表 |
| M12 结束 | 掌握 RL 基础、Sim-to-Real 方法、数据工程和多任务策略 | BC/RL/Diffusion 对比报告 + 数据管线 + Benchmark 评估 |
| M15 结束 | 能做一个语言驱动的具身系统 demo，了解前沿方向，以研究/求职方式打包 | 代码仓库、技术报告、演示视频、答辩提纲 |

---

## 推荐资源（按阶段，v3 更新）

| 阶段 | 建议资源 |
|---|---|
| 阶段 1｜代码与深度学习工程 | PyTorch 官方 Tutorials、Dive into Deep Learning、你自己已跑过的微调仓库 |
| 阶段 2｜数学 / 运动学 / 控制 | Modern Robotics、线代与优化速查资料、经典控制入门材料 |
| 阶段 3｜ROS 2 / MoveIt 2 / 仿真全景 | ROS 2 官方 Tutorials、MoveIt 2 Quickstart、**Isaac Lab 文档**、**Genesis 文档** |
| 阶段 4｜仿真 / IL / RL / 数据 | MuJoCo 文档、**LeRobot 官方教程与 HuggingFace Robotics Course**、**Diffusion Policy 项目页**、Gymnasium、**OpenPI (π0 开源仓库)**、**Open X-Embodiment**、**LIBERO / ManiSkill / SimplerEnv** |
| 阶段 5｜感知 / VLA / 前沿 | 目标检测与位姿估计教程、**VLA Survey (arXiv:2505.04769)**、**ICLR 2026 VLA 综述博客**、**Awesome-VLA (GitHub)**、NVIDIA Cosmos 文档 |

---

## v3 新增关键论文/项目推荐

| 主题 | 推荐资源 |
|---|---|
| Diffusion Policy | Diffusion Policy (Chi et al., RSS 2023 / IJRR 2024) — diffusion-policy.cs.columbia.edu |
| Flow Matching | π0 (Physical Intelligence, 2024) — arxiv.org/abs/2410.24164 |
| ACT | Learning Fine-Grained Bimanual Manipulation (Zhao et al., RSS 2023) |
| LeRobot | github.com/huggingface/lerobot + HuggingFace Robotics Course |
| OpenVLA | OpenVLA (Kim et al., 2024) — openvla.github.io |
| Isaac Lab | isaac-sim.github.io/IsaacLab |
| Genesis | genesis-world.readthedocs.io |
| Open X-Embodiment | robotics-transformer-x.github.io |
| DROID | droid-dataset.github.io |
| World Models | NVIDIA Cosmos、Genie 3 |
| Sim-to-Real | Domain Randomization 综述 + DROPO |
| VLA Survey | arxiv.org/abs/2505.04769 |
| Benchmark | LIBERO、ManiSkill、RLBench、CALVIN、SimplerEnv |

---

## 最后提醒

对你来说，真正决定进步速度的不是再多看多少前沿模型，而是能否把每一周的最小交付稳稳做出来。连续 60 周的小交付，会比零散刷知识点更快把你送到具身智能算法岗的门口。

**v3 的额外提醒：** 2025–2026 年具身智能正经历"ChatGPT 时刻"。掌握 Diffusion Policy、LeRobot、VLA 微调和 GPU 仿真不再是加分项，而是算法岗的基本要求。这份修订版确保你的技能栈与行业节奏同步，而不是只停留在经典技术上。
