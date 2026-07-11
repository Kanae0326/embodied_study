# 单元 1.1 环境配置手册

## 0. 学习环境说明（先读）

- 本套资料的所有命令按 **Linux** 编写：**WSL 2（Ubuntu）或原生 Ubuntu 均可**，对本课程没有区别。**不要**在 Windows 原生 PowerShell / CMD 里照敲。
- Windows 机器上装 WSL：管理员 PowerShell 运行 `wsl --install -d Ubuntu`，之后所有操作都在 Ubuntu 终端里进行（官方指引：https://learn.microsoft.com/windows/wsl/install ）。
- 路径约定：文档统一用 `~/code/embodied_study` 指代仓库根目录。你的克隆位置不同（例如 WSL 里通过 `/mnt/d/...` 访问 Windows 盘），就把命令里的这段路径替换成实际路径。**WSL 下建议把仓库 clone 到 Linux 文件系统（如 `~/code/`）而不是 `/mnt/` 下，文件 IO 快得多。**
- **换机器学习时：** 环境不随仓库走，每台新机器都要重做本文档第 1–3 节，并跑通第 7 节的最终验证；仓库本身推到远端（GitHub 等私有仓库），新机器 `git clone` 即可。

## 1. 创建 Python 虚拟环境

推荐使用 Conda（新机器上没有就先装 Miniconda：https://docs.anaconda.com/miniconda/ ）：

```bash
# 创建环境
conda create -n embodied python=3.11 -y

# 激活环境
conda activate embodied

# 确认 Python 版本
python --version
# 应该输出: Python 3.11.x
```

如果你更习惯 venv：

```bash
python3 -m venv ~/envs/embodied
source ~/envs/embodied/bin/activate
```

## 2. 安装 PyTorch

### 有 NVIDIA GPU (推荐)

先检查驱动支持的 CUDA 版本上限：
```bash
nvidia-smi
# 看右上角 CUDA Version，如 12.x
```

然后用 **PyTorch 官方安装选择器**生成安装命令（选 Linux / pip / 对应 CUDA）：
https://pytorch.org/get-started/locally/

生成的命令形如 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX`。**cuXXX 以选择器给出的为准，不要照抄教程或旧文档里的版本号**——旧 CUDA 渠道（如 cu124）已停留在旧版 PyTorch，会无意中装到过时版本。

### 仅 CPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 验证安装

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(3, 4)
print('Tensor test OK, shape:', x.shape)
"
```

你应该看到类似输出：
```
PyTorch version: 2.x.x
CUDA available: True
Tensor test OK, shape: torch.Size([3, 4])
```

## 3. 安装其他必要包

```bash
pip install matplotlib numpy pyyaml tqdm
```

## 4. Git 初始化

> 已经在别的机器建好仓库、推到了远端？那本节跳过，直接 `git clone <你的远端地址> ~/code/embodied_study`。

```bash
cd ~/code/embodied_study

# 如果还没有初始化 git
git init

# 创建 .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/

# 环境
.env
venv/
*.egg

# PyTorch
*.pth
*.pt
checkpoints/

# 数据集（不入库，太大）
data/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# 日志
logs/
runs/
wandb/
EOF

git add .gitignore
git commit -m "Initial commit: add .gitignore"
```

## 5. VS Code 推荐配置

### 必装扩展

| 扩展 | 作用 |
|---|---|
| **Python** (Microsoft) | Python 语言支持 |
| **Pylance** (Microsoft) | 代码补全和类型检查 |
| **GitLens** | Git 增强 |

### 可选扩展

| 扩展 | 作用 |
|---|---|
| **autoDocstring** | 自动生成 docstring |
| **Error Lens** | 在行内直接显示错误 |
| **Jupyter** | Notebook 支持 |

### 设置 Python 解释器

1. `Ctrl+Shift+P` → 输入 "Python: Select Interpreter"
2. 选择你创建的 `embodied` 环境

## 6. 项目目录结构

```bash
# 进入本章（章 1）工作目录（如果你用这份资料，目录已经创建好了）
cd ~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/

# 目录结构如下：
# C1_pytorch_basics/                     # 章 1（在 P1_code_foundation/ 下）
# ├── U1.1_env_and_skeleton/             # 单元 1.1（本单元）
# │   ├── index.md                       # 单元入口（先看这个）
# │   ├── study_guide.md                 # 张量/autograd 学习指南
# │   ├── environment_setup.md           # 本文件
# │   ├── shell_basics.md                # shell 基础
# │   ├── checklist.md                   # 验收 / 出关
# │   └── tensor_playground.py           # 张量练习
# └── project_skeleton/                  # 章级共享 · FashionMNIST 项目骨架
#     ├── SETUP.md                       # 怎么用骨架
#     ├── utils.py                       # 单元 1.1 写
#     └── model.py / dataset.py / train.py   # 单元 1.2 写
# （你自己建的 project/ 在章目录下；其中 .py 代码必须入库，data/checkpoints 等产物由 .gitignore 排除）
```

## 7. 最终验证

跑以下命令，全部通过说明环境就绪：

```bash
conda activate embodied

python -c "import torch; print('torch OK')"
python -c "import torchvision; print('torchvision OK')"
python -c "import matplotlib; print('matplotlib OK')"
python -c "import numpy; print('numpy OK')"
python -c "import yaml; print('pyyaml OK')"

# 测试 GPU（如有）
python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(100, 100, device='cuda')
    print('GPU compute OK, device:', x.device)
else:
    print('No GPU, will use CPU (本单元 CPU 足够)')
"

git status  # 确认 git 工作正常
```

全部显示 OK 后，环境配置完成，开始做 `tensor_playground.py`。
