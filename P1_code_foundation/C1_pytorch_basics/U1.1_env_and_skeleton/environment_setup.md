# 单元 1.1 环境配置手册

## 1. 创建 Python 虚拟环境

推荐使用 Conda（你已有 miniconda3）：

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

先检查你的 CUDA 版本：
```bash
nvidia-smi
# 看右上角 CUDA Version，如 12.x
```

安装 PyTorch（以 CUDA 12.4 为例）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

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
# （你自己建的 project/ 在章目录下，不入库）
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
    print('No GPU, will use CPU (also OK for M1)')
"

git status  # 确认 git 工作正常
```

全部显示 OK 后，环境配置完成，开始做 `tensor_playground.py`。
