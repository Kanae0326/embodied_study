# 单元 1.1 | Shell 基础（工程生存技能）

> 本文档是单元 1.1 的一部分。后续做 ROS 2 colcon build、训练日志重定向、GPU 监控、tmux 挂训练任务时，shell 熟练度会直接决定你卡多久。
> 目标不是背完所有选项，而是**遇到任何场景都能立刻想到该用哪个命令 + 会查 man**。

---

## 学习节奏建议

- 首次通读：40 分钟
- 动手练习：60 分钟
- 以后遇到不熟的命令，回来补练习 3–5 就够用

---

## 一、导航与查看

| 命令 | 作用 | 常用写法 |
|---|---|---|
| `pwd` | 打印当前目录 | `pwd` |
| `ls` | 列目录 | `ls -lah`（长列表 + 隐藏 + 人类可读大小） |
| `cd` | 切换目录 | `cd -`（回上一次目录）、`cd ~`（home） |
| `tree` | 树状列目录 | `tree -L 2`（只看 2 层） |
| `cat` | 输出文件内容 | 小文件用，大文件别用 |
| `less` | 分页看文件 | `less file.log`，按 `q` 退出，`/` 搜索 |
| `head` / `tail` | 头/尾 N 行 | `head -n 20`、`tail -n 50` |
| `tail -f` | 持续追踪文件新增行 | **看训练日志必备** |
| `wc` | 行/词/字符计数 | `wc -l file.py`（只看行数） |

---

## 二、文件操作

| 命令 | 作用 | 注意 |
|---|---|---|
| `mkdir -p` | 递归建目录 | `-p` 遇到已存在不报错 |
| `touch` | 新建空文件或更新时间戳 | |
| `cp` | 复制 | `cp -r` 递归复制目录 |
| `mv` | 移动 / 重命名 | 同目录下就是重命名 |
| `rm` | 删除 | `rm -r` 删目录；**rm -rf 之前多看一眼 pwd** |
| `ln -s` | 软链接 | 数据集太大不想重复下载时常用 |

---

## 三、查找

```bash
# 按文件名找
find . -name "*.py"
find . -type d -name "checkpoints"

# 按内容找（推荐用 ripgrep，如未装就用 grep -rn）
grep -rn "def train_one_epoch" .
grep -rn "batch_size" --include="*.py" .

# 在 PATH 里找命令的位置
which python
which nvidia-smi
```

**记忆点：** `find` 找文件名，`grep` 找文件内容，`which` 找命令。

---

## 四、管道、重定向、组合

这是 shell 的灵魂。**不会管道 = 不会 shell**。

```bash
# 管道：前一个命令的 stdout 当作下一个命令的 stdin
ls | wc -l                          # 数当前目录文件数
ps aux | grep python                # 找所有 python 进程

# 重定向
python train.py > train.log         # stdout 写到 train.log（覆盖）
python train.py >> train.log        # stdout 追加
python train.py 2> err.log          # 只重定向 stderr
python train.py > all.log 2>&1      # stdout + stderr 都写到 all.log
python train.py &> all.log          # 简写（bash 专用）
python train.py > /dev/null 2>&1    # 全部丢掉（只想让它跑不看输出）

# tee：同时写文件和屏幕（训练时非常常用）
python train.py 2>&1 | tee train.log
```

---

## 五、权限、进程、环境

```bash
# 权限
chmod +x script.sh          # 加可执行权限
chmod 644 file.txt          # 数字模式（rw-r--r--）

# 进程
ps aux | grep python        # 找进程
top / htop                  # 实时查看（htop 更好看，需要安装）
kill <PID>                  # 正常结束
kill -9 <PID>               # 强制杀
Ctrl+C                      # 终止前台进程
Ctrl+Z + bg                 # 挂后台
jobs                        # 看后台任务
fg %1                       # 把 job 1 拉回前台

# 环境变量
echo $PATH
export CUDA_VISIBLE_DEVICES=0
env | grep CUDA             # 看所有 CUDA 相关环境变量
source ~/.bashrc            # 重新加载配置
```

---

## 六、磁盘与打包

```bash
df -h                       # 看各分区剩余（数据盘满了会训到一半崩）
du -sh *                    # 看当前目录下每个子项大小
du -sh ~/.cache             # 经常是显存/硬盘杀手

tar czvf out.tar.gz dir/    # 打包 + gzip 压缩
tar xzvf out.tar.gz         # 解压
```

---

## 七、远程与会话（后面训练必用）

```bash
# SSH 与文件传输
ssh user@host
scp local_file user@host:/remote/path/
rsync -avz local/ user@host:/remote/   # 增量同步，比 scp 好

# tmux —— 长训练任务的救命工具
tmux new -s train           # 新建一个叫 train 的 session
# （在 tmux 里跑 python train.py）
# Ctrl+b 然后 d：detach 离开（训练继续跑）
tmux ls                     # 看有哪些 session
tmux attach -t train        # 回到 train session

# tmux 基本按键（都是 Ctrl+b 前缀）
# c  新开窗口     n/p  切换窗口
# %  左右分屏     "  上下分屏
# 方向键  切换 pane
# d  detach     x  关 pane
```

**一旦训练超过 30 分钟，一定要在 tmux / screen / nohup 里跑，否则 SSH 掉线就白训。**

---

## 八、GPU 与 ML 场景专用

```bash
nvidia-smi                          # 瞬时看一眼 GPU
watch -n 1 nvidia-smi               # 每秒刷新一次
nvidia-smi dmon -s u                # 利用率监控

# 只用某张卡
CUDA_VISIBLE_DEVICES=0 python train.py

# 日志追踪
tail -f logs/train.log              # 监训练进度

# 找显存泄漏的僵尸进程
nvidia-smi | grep python
```

---

## 九、动手练习（必做）

**要求：每题先自己动手 5 分钟，卡住再查 man 或 AI。全程不要直接让 AI 帮你写完整命令。**

### 练习 1：建个目录结构

在 `~/tmp/shell_exercise/` 下创建：
```
shell_exercise/
├── data/
│   └── raw/
├── logs/
└── README.md（空文件）
```
完成后用 `tree` 验证。

### 练习 2：找文件 + 统计行数

在 `~/code/embodied_study/` 下，用**一条命令**：
- 找出所有 `.py` 文件
- 统计它们的总行数

提示：`find ... | xargs wc -l`

### 练习 3：重定向 + tee

写一个一行 Python 命令，输出一段字符串到 stdout 和一段报错到 stderr：
```bash
python -c "import sys; print('ok'); print('bad', file=sys.stderr)"
```
然后分别完成：
- 只把 stdout 写进 `out.log`，stderr 照常显示
- stdout 和 stderr 都写进 `all.log`
- 用 `tee` 让 stdout 同时写进文件 **并** 显示到屏幕

### 练习 4：tmux 实战

1. `tmux new -s test`
2. 里面跑 `for i in $(seq 1 30); do echo $i; sleep 1; done`
3. 按 `Ctrl+b` 然后 `d` detach
4. 关掉当前终端窗口，**重新**打开一个终端
5. `tmux attach -t test` 回去看，任务是否还在跑？

### 练习 5：grep 查代码

在 `~/code/embodied_study/P1_code_foundation/C1_pytorch_basics/` 下：
- 找出所有用到 `torch.no_grad` 的行，连同文件名和行号
- 找出所有 `TODO` 注释

### 练习 6：模拟真实训练场景

```bash
cd ~/tmp/shell_exercise
tmux new -s fake_train
# 在 tmux 里跑：
python -c "
import time
for i in range(20):
    print(f'epoch {i} loss={1.0/(i+1):.3f}', flush=True)
    time.sleep(2)
" 2>&1 | tee logs/train.log
```

然后：
1. detach 出来
2. 在外面用 `tail -f logs/train.log` 监看
3. 看完按 `Ctrl+C` 退出 tail（**不会**杀掉训练）
4. attach 回 tmux 确认训练还在跑

---

## 十、验收

- [ ] 能说出 `find` / `grep` / `which` 的区别
- [ ] 能写出 `python x.py > out.log 2>&1 &` 的每一部分含义
- [ ] 知道 `tmux detach` vs `Ctrl+C` 的本质差异
- [ ] 会用 `tail -f` 看训练日志
- [ ] 会用 `nvidia-smi` 看 GPU 状态
- [ ] 练习 1–6 全部完成

---

## 十一、遇到不会的命令怎么办

1. `man <命令>` —— 系统自带手册
2. `<命令> --help` —— 简版帮助
3. `tldr <命令>` —— 简洁例子（需装 tldr）
4. 查 ExplainShell.com —— 解析复杂管道命令
5. **最后**才是问 AI

原则：先看官方 man / --help，再问 AI。man 读不懂时问 AI 解释 man，不是让 AI 直接给答案。
