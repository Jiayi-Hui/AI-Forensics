# 图像 DeepFake 检测系统 — 新组员完整使用教程

> 适用读者：有基本 Python 基础、但未接触过本项目的新组员。  
> 阅读本文档后，你应能在不询问任何人的情况下独立运行整个项目。
第一次运行该项目时可以自行阅读此文档，也可以引导AI阅读此文档后操作。

---

## 0. 项目简介

本项目是一套基于深度学习的 **AI 生成图像检测系统**，使用 EfficientNet-B7 作为骨干网络，能够判断一张图像是真实照片还是 AI 生成的假图。

项目的核心问题是**跨域泛化**：模型在 StyleGAN 生成的人脸图像上训练，能否正确检测出 Stable Diffusion（CIFAKE 数据集）生成的图像？实验设计了三个递进策略来解决这一问题。

最终结果（策略B 混合训练）在 CIFAKE 数据集上达到 **REAL 准确率 98%、FAKE 准确率 100%、总体 99%**，同时保持 StyleGAN 域总体准确率 **94%**，实现了两个域的兼顾。

---

## 1. 环境要求

### Python 版本

Python **3.10 或 3.11**（推荐 3.11）。  
Python 3.12 尚未全面验证，不建议使用。

---

### 第一步：确认你的运行环境

不同的机器需要安装不同版本的 PyTorch，**装错版本是最常见的问题来源**。  
请先确认自己属于哪种情况：

#### 情况一：没有 NVIDIA GPU（纯 CPU）

可以运行本项目，但训练速度很慢。**推理脚本（不含训练）可以正常使用**，速度可接受。

| 脚本 | CPU 预计时长 |
|------|------------|
| `train_model.py` | 约 3～6 小时 |
| `finetune_cifake.py` | 约 4～8 小时 |
| `train_mixed.py` | 约 8～16 小时 |
| `batch_predict.py` / `batch_predict_cifake.py` | 约 2～5 分钟（可接受） |
| `eval_all_strategies.py` | 约 5～15 分钟（可接受） |

> **建议**：如果没有 GPU，使用已有的训练好的权重（见第 5 节），跳过所有训练步骤，直接运行推理和评估。

#### 情况二：有 NVIDIA GPU，但不是 RTX 5070

先查自己的 CUDA 版本（**在系统终端，非 Python 中运行**）：

```bash
nvidia-smi
```

输出中找 `CUDA Version: XX.X` 这一行，根据版本号选择 PyTorch 安装命令（见下方兼容性速查表）。

#### 情况三：RTX 5070 / 5080 / 5090（Blackwell 架构）

这是本项目原始开发环境。稳定版 PyTorch 无法识别 Blackwell GPU，必须安装 nightly 版本（见下方安装方案 C）。

#### 情况四：腾讯云 / 阿里云 / Google Colab 等云 GPU

云环境通常预装了 CUDA，同样先运行 `nvidia-smi` 查看版本，再对照下方速查表安装。  
Colab 额外注意事项见本节末尾。

---

### GPU / PyTorch 兼容性速查表

| 你的 GPU | 架构 | CUDA 版本 | 推荐 PyTorch 安装方案 |
|----------|------|-----------|----------------------|
| RTX 5070 / 5080 / 5090 | Blackwell | 12.8 | **方案 C**（nightly cu128） |
| RTX 4060 / 4070 / 4080 / 4090 | Ada Lovelace | 12.1 ~ 12.6 | **方案 B**（稳定版 cu121） |
| RTX 3060 / 3070 / 3080 / 3090 | Ampere | 11.8 ~ 12.1 | **方案 A**（稳定版 cu118） |
| Tesla T4 / V100（腾讯云标准型） | Volta/Turing | 11.x | **方案 A**（稳定版 cu118） |
| A100 / H100（腾讯云高性能型） | Ampere/Hopper | 12.x | **方案 B**（稳定版 cu121） |
| 无 GPU / MacBook / AMD GPU | — | — | **方案 D**（CPU-only） |

> **如何确认架构**：不确定 GPU 型号时，运行 `nvidia-smi` 查看 GPU 名称，再对照上表。

---

### 安装依赖

#### 其他依赖（所有方案通用，先安装）

```bash
pip install timm==1.0.25 pandas pillow matplotlib numpy opencv-python grad-cam scikit-learn kaggle
```

#### 方案 A：稳定版 PyTorch，CUDA 11.8（RTX 30 系 / Tesla T4 / V100）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 方案 B：稳定版 PyTorch，CUDA 12.1（RTX 40 系 / A100）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 方案 C：nightly 版 PyTorch，CUDA 12.8（RTX 50 系 Blackwell，本项目原始环境）

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 方案 D：CPU-only（无 GPU 或 macOS / AMD GPU）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> **macOS Apple Silicon（M1/M2/M3）用户**：可以使用 MPS 加速（Metal Performance Shaders），安装标准版 `pip install torch torchvision torchaudio` 即可，PyTorch 会自动检测 MPS。但速度远不如 NVIDIA GPU，训练脚本速度约为 CUDA 的 1/5～1/3。

---

### 验证安装是否正确

```bash
python test_GPU.py
```

| 你期望看到的输出 | 说明 |
|----------------|------|
| `True` + GPU 名称 | GPU 可用，正常 |
| `False` + `无GPU` | 将使用 CPU，训练速度慢 |
| 报错 `ModuleNotFoundError` | PyTorch 未安装，重新执行上方安装命令 |

**GPU 可用时的正确输出示例：**
```
2.12.0.dev20260320+cu128
True
NVIDIA GeForce RTX 5070
```

**CPU 模式的正常输出示例：**
```
2.3.0+cpu
False
无GPU
```

---

### 使用本项目已有虚拟环境（仅限 Windows 本机，RTX 5070 环境）

项目根目录下的 `.venv/` 是原始开发环境（PyTorch nightly cu128），**仅适用于与原机器相同的 RTX 50 系 + CUDA 12.8 配置**。

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Windows Git Bash / CMD
.venv\Scripts\activate
```

若你的环境不同，请新建虚拟环境并按上方方案安装：

```bash
python -m venv venv_new
# Windows
venv_new\Scripts\activate
# Linux / macOS / 云服务器
source venv_new/bin/activate

pip install timm==1.0.25 pandas pillow matplotlib numpy opencv-python grad-cam scikit-learn kaggle
# 然后根据你的 GPU 选择上方方案 A/B/C/D 安装 PyTorch
```

---

### 腾讯云 GPU 实例专项说明

腾讯云常见 GPU 实例（GN 系列）及对应方案：

| 腾讯云实例 | GPU 型号 | 推荐方案 |
|-----------|---------|---------|
| GN7（入门） | Tesla T4 | 方案 A（cu118） |
| GN10X（标准） | Tesla V100 | 方案 A（cu118） |
| GN16（高性能） | NVIDIA A100 | 方案 B（cu121） |

**腾讯云配置步骤：**

```bash
# 1. 登录实例后，先检查 CUDA 版本
nvidia-smi

# 2. 创建虚拟环境（推荐，避免污染系统 Python）
python3 -m venv ~/deepfake_env
source ~/deepfake_env/bin/activate

# 3. 安装依赖（以 T4/V100 为例，cu118）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm==1.0.25 pandas pillow matplotlib numpy opencv-python grad-cam scikit-learn kaggle

# 4. 上传项目文件（在本地执行）
scp -r ./8307_GroupProject ubuntu@<实例IP>:~/

# 5. 上传权重文件（约 750 MB，三个 .pth 文件）
scp weights/*.pth ubuntu@<实例IP>:~/8307_GroupProject/weights/

# 6. 验证
cd ~/8307_GroupProject
python test_GPU.py
```

> **数据集上传**：两个数据集压缩包合计约 4 GB，建议通过腾讯云对象存储（COS）中转，或直接在实例上用 `kaggle` 命令下载（需配置 Kaggle API Key）。

---

### Google Colab 专项说明

Colab 免费版通常提供 T4（cu118），Pro 版可能有 A100。

```python
# Colab 中第一个 Cell 运行：检查 GPU
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

```bash
# 安装依赖（Colab 已预装 PyTorch，通常只需装额外包）
!pip install timm==1.0.25 grad-cam kaggle
```

```python
# Colab 中挂载 Google Drive（存放权重和数据）
from google.colab import drive
drive.mount('/content/drive')

# 将项目文件上传到 Google Drive，然后：
import os
os.chdir('/content/drive/MyDrive/8307_GroupProject')
```

> **Colab 注意事项**：Colab 运行时断开后环境会重置，每次需重新安装依赖。建议将权重文件存在 Google Drive 中以避免重复下载。

---

## 2. 数据集准备

本项目需要两个数据集（**`data/` 目录下已包含压缩包**，解压即可）：

### 数据集 A：140K Real and Fake Faces（StyleGAN 人脸）

- **来源**：Kaggle 数据集 `xhlulu/140k-real-and-fake-faces`
- **已有压缩包**：`data/140k-real-and-fake-faces.zip`（约 3.8 GB）
- **作用**：训练基础 Baseline 模型

```bash
# Kaggle 下载命令（若压缩包丢失）
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/

# 解压
cd data
unzip 140k-real-and-fake-faces.zip -d real_vs_fake
```

解压后目录结构：

```
data/
└── real_vs_fake/
    └── real-vs-fake/
        ├── train/
        │   ├── fake/    ← StyleGAN 生成人脸
        │   └── real/    ← 真实人脸照片
        ├── valid/
        │   ├── fake/
        │   └── real/
        └── test/
            ├── fake/
            └── real/
```

### 数据集 B：CIFAKE（Stable Diffusion 生成图像）

- **来源**：Kaggle 数据集 `birdy654/cifake-real-and-ai-generated-synthetic-images`
- **已有压缩包**：`data/cifake-real-and-ai-generated-synthetic-images.zip`（约 105 MB）
- **作用**：跨域测试目标域，以及策略A/B 的训练数据

```bash
# Kaggle 下载命令（若压缩包丢失）
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/

# 解压（直接解压到 data/ 目录）
cd data
unzip cifake-real-and-ai-generated-synthetic-images.zip
```

解压后 `data/` 下应出现：

```
data/
├── train/
│   ├── FAKE/    ← Stable Diffusion 生成图像（注意：大写）
│   └── REAL/    ← 真实照片
└── test/
    ├── FAKE/
    └── REAL/
```

### 数据就绪检查

运行以下命令验证数据是否准备正确：

```bash
python -c "
import os
checks = [
    'data/real_vs_fake/real-vs-fake/train/fake',
    'data/real_vs_fake/real-vs-fake/train/real',
    'data/real_vs_fake/real-vs-fake/valid/fake',
    'data/real_vs_fake/real-vs-fake/valid/real',
    'data/train/FAKE',
    'data/train/REAL',
    'data/test/FAKE',
    'data/test/REAL',
]
for p in checks:
    count = len(os.listdir(p)) if os.path.exists(p) else -1
    status = '✅' if count > 0 else '❌'
    print(f'{status} {p} ({count} 个文件)')
"
```

---

## 3. 项目文件说明

| 文件名 | 作用 | 输入 | 输出 |
|--------|------|------|------|
| `train_model.py` | 训练 Baseline 模型（在 StyleGAN 人脸数据上） | `data/real_vs_fake/real-vs-fake/train/` 和 `valid/` | `weights/efficientnet_b7_deepfake.pth` |
| `batch_predict.py` | 用 Baseline 权重对 StyleGAN 测试图像推理 | `test_images/real/` 和 `test_images/fake/` | `results/predictions.csv` |
| `prepare_cifake.py` | 从 CIFAKE 测试集随机采样 50+50 张图像 | `data/test/REAL/` 和 `data/test/FAKE/` | `test_images_cifake/real/` 和 `test_images_cifake/fake/` |
| `batch_predict_cifake.py` | 用 Baseline 权重对 CIFAKE 采样图像推理（跨域零样本测试） | `test_images_cifake/` + `weights/efficientnet_b7_deepfake.pth` | `results/predictions_cifake.csv` |
| `compare_results.py` | 对比 StyleGAN 域与 CIFAKE 域的推理结果，生成分布图 | `results/predictions.csv` + `results/predictions_cifake.csv` | `results/compare_distribution.png`、`compare_boxplot.png` |
| `finetune_cifake.py` | 策略A：将 Baseline 权重在 CIFAKE 训练集上微调（冻结前层） | `data/train/REAL/FAKE/` + `weights/efficientnet_b7_deepfake.pth` | `weights/efficientnet_b7_cifake_finetuned.pth` |
| `train_mixed.py` | 策略B：从头混合 StyleGAN + CIFAKE 两个域联合训练 | `data/real_vs_fake/` + `data/train/` + `data/test/` | `weights/efficientnet_b7_mixed.pth` |
| `eval_all_strategies.py` | 对三个权重在 CIFAKE 测试集上统一评估，生成对比图表 | `weights/` 下三个权重 + `test_images_cifake/` | `results/predictions_cifake_strategyA/B.csv`、三张对比图 |
| `gradcam_vis.py` | 生成 Grad-CAM 热力图，可视化模型关注区域 | `test_images/` + `weights/efficientnet_b7_deepfake.pth` | `heatmaps/REAL/` 和 `heatmaps/FAKE/` |
| `test_GPU.py` | 验证 GPU 环境是否可用 | — | 打印 PyTorch 版本和 GPU 名称 |
| `app.py` | Streamlit 应用入口（Web 界面，非实验必须） | — | 启动 Web 服务 |

---

## 4. 运行流程

> 所有命令均在项目根目录下运行，且已激活虚拟环境。

**跨平台路径说明**：`train_model.py`、`batch_predict.py`、`gradcam_vis.py` 这三个脚本内部硬编码了 Windows 绝对路径（`F:\HYH_LocalFile\...`）。如果你在 Linux / 云服务器 / macOS 上运行，需要在运行前修改这些脚本中的路径。其他脚本（`batch_predict_cifake.py`、`finetune_cifake.py`、`train_mixed.py`、`eval_all_strategies.py`、`compare_results.py`、`prepare_cifake.py`）均使用 `Path(__file__).resolve().parent` 动态解析，**无需修改，可直接跨平台运行**。

**修改方法**（以 `train_model.py` 为例）：

```python
# 原来（第 9~11 行）：
TRAIN_DIR = r"F:\HYH_LocalFile\8307_GroupProject\data\real_vs_fake\real-vs-fake\train"
VALID_DIR = r"F:\HYH_LocalFile\8307_GroupProject\data\real_vs_fake\real-vs-fake\valid"
SAVE_PATH = r"F:\HYH_LocalFile\8307_GroupProject\weights\efficientnet_b7_deepfake.pth"

# 改为（适用于所有平台）：
from pathlib import Path
_root = Path(__file__).resolve().parent
TRAIN_DIR = str(_root / "data" / "real_vs_fake" / "real-vs-fake" / "train")
VALID_DIR = str(_root / "data" / "real_vs_fake" / "real-vs-fake" / "valid")
SAVE_PATH = str(_root / "weights" / "efficientnet_b7_deepfake.pth")
```

---

### 步骤 1：训练基础模型

```bash
python train_model.py
```

**预期控制台输出：**
```
类别映射: {'fake': 0, 'real': 1}
训练集: 2000 张 | 验证集: 400 张
使用设备: cuda
Epoch 1/5 | Loss: 0.4231 | Train Acc: 0.7935 | Val Acc: 0.8250
  💾 最佳模型已保存 (Val Acc: 0.8250)
Epoch 2/5 | Loss: 0.3102 | Train Acc: 0.8670 | Val Acc: 0.8575
  💾 最佳模型已保存 (Val Acc: 0.8575)
...
✅ 训练完成！最佳验证准确率: 0.87xx
权重保存在: ...\weights\efficientnet_b7_deepfake.pth
```

**预期运行时长**：

| 环境 | 时长 |
|------|------|
| RTX 5070 / 4090 (GPU) | 约 10～20 分钟 |
| RTX 3080 / T4 (GPU) | 约 20～40 分钟 |
| A100 (GPU) | 约 5～10 分钟 |
| CPU（无 GPU） | 约 3～6 小时，不推荐 |

**成功标志**：`weights/efficientnet_b7_deepfake.pth` 文件生成（约 245 MB）

---

### 步骤 2：StyleGAN 基础推理（Baseline 测试）

```bash
python batch_predict.py
```

**预期输出：**
```
✅ 已加载训练好的 Deepfake 检测权重
📁 REAL 文件夹 → 50 张图像
📁 FAKE 文件夹 → 50 张图像
...
  REAL: 78.0%  (39/50 正确)
  FAKE: 84.0%  (42/50 正确)
  总体准确率: 81.0%
✅ 结果已保存至 results/predictions.csv
```

**成功标志**：`results/predictions.csv` 生成

---

### 步骤 3：准备 CIFAKE 测试图像

```bash
python prepare_cifake.py
```

**预期输出：**
```
=== CIFAKE 测试集采样 ===
源目录 REAL: ...\data\test\REAL
源目录 FAKE: ...\data\test\FAKE
REAL 可用图像: 10000
FAKE 可用图像: 10000
=== 采样完成 ===
REAL 已复制: 50 张
FAKE 已复制: 50 张
test_images_cifake/real: 50 张
test_images_cifake/fake: 50 张
```

**成功标志**：`test_images_cifake/real/` 和 `test_images_cifake/fake/` 各含 50 张图像

---

### 步骤 4：CIFAKE 跨域推理（零样本测试）

```bash
python batch_predict_cifake.py
```

**预期输出：**
```
使用设备: cuda
✅ 已加载训练好的 Deepfake 检测权重
📁 REAL 文件夹 → 50 张图像
📁 FAKE 文件夹 → 50 张图像
...
  REAL: 28.0%  (14/50 正确)
  FAKE: 58.0%  (29/50 正确)
  总体准确率: 43.0%
✅ 结果已保存至 results/predictions_cifake.csv
```

> 注意：此步骤准确率很低（约 43%）是预期结果，说明跨域泛化问题存在，后续策略将解决此问题。

**成功标志**：`results/predictions_cifake.csv` 生成

---

### 步骤 5：跨域对比分析

```bash
python compare_results.py
```

**预期输出：**
```
| 数据来源       | 生成方式           | REAL 准确率 | FAKE 准确率 | 总体准确率 |
|----------------|--------------------|-------------|-------------|------------|
| 140K 数据集    | StyleGAN（训练域） |    78.0%    |    84.0%    |   81.0%    |
| CIFAKE 数据集  | Stable Diffusion   |    28.0%    |    58.0%    |   43.0%    |
```

**成功标志**：`results/compare_distribution.png` 和 `results/compare_boxplot.png` 生成

---

### 步骤 6：策略A — 专项微调

在 Baseline 权重基础上，冻结前层，仅在 CIFAKE 训练集上微调后层。

```bash
python finetune_cifake.py
```

**预期输出：**
```
训练设备: cuda
REAL 总数: 50000
FAKE 总数: 50000
训练子集: REAL 3000 + FAKE 3000 = 6000
验证子集: REAL 500 + FAKE 500 = 1000
可训练参数: 17xxx/66xxx
[Epoch 1/5] Train Loss: 0.3521 | Train Acc: 85.00% | Val Acc: 87.50% [saved]
...
训练完成，最佳 Val Acc: 92.xx%
权重已保存: ...\weights\efficientnet_b7_cifake_finetuned.pth
```

**预期运行时长**：

| 环境 | 时长 |
|------|------|
| RTX 5070 / 4090 (GPU) | 约 15～25 分钟 |
| RTX 3080 / T4 (GPU) | 约 30～50 分钟 |
| CPU（无 GPU） | 约 4～8 小时，不推荐 |

**成功标志**：`weights/efficientnet_b7_cifake_finetuned.pth` 生成

---

### 步骤 7：策略B — 混合训练（推荐最优策略）

从 Baseline 权重出发，混合 StyleGAN + CIFAKE 两个域联合训练 8 个 epoch，全参数更新。

```bash
python train_mixed.py
```

**预期输出：**
```
训练设备: cuda
训练集构成: StyleGAN REAL 2000, StyleGAN FAKE 2000, CIFAKE REAL 2000, CIFAKE FAKE 2000
验证集构成: StyleGAN REAL 400, StyleGAN FAKE 400, CIFAKE REAL 400, CIFAKE FAKE 400
[Epoch 1/8] Loss: 0.2831 | Mixed Val: 88.12% | StyleGAN Val: 86.25% | CIFAKE Val: 90.00%
...
训练完成，最佳 Mixed Val: 96.xx%
权重已保存: ...\weights\efficientnet_b7_mixed.pth
```

**预期运行时长**：

| 环境 | 时长 |
|------|------|
| RTX 5070 / 4090 (GPU) | 约 30～50 分钟 |
| RTX 3080 / T4 (GPU) | 约 60～90 分钟 |
| A100 (GPU) | 约 15～25 分钟 |
| CPU（无 GPU） | 约 8～16 小时，不推荐 |

**成功标志**：`weights/efficientnet_b7_mixed.pth` 生成

---

### 步骤 8：三策略统一评估

对 Baseline、策略A、策略B 三个权重在 CIFAKE 测试集上统一评估，并附加 StyleGAN 遗忘检查。

```bash
python eval_all_strategies.py
```

**预期输出：**
```
评估设备: cuda
╔══════════════════════════════════════════════════════╗
║          CIFAKE 跨域测试 — 策略对比汇总              ║
╠══════════════╦════════════╦════════════╦════════════╣
║ 策略         ║ REAL 准确率 ║ FAKE 准确率 ║ 总体准确率 ║
╠══════════════╬════════════╬════════════╬════════════╣
║ Baseline     ║  28.0%    ║  58.0%    ║  43.0%    ║
║ 策略A（微调） ║  92.0%    ║  84.0%    ║  88.0%    ║
║ 策略B（混合） ║  98.0%    ║ 100.0%    ║  99.0%    ║
╚══════════════╩════════════╩════════════╩════════════╝

StyleGAN 测试集附加评估（遗忘检查）：
  Baseline StyleGAN 总体准确率: 81.0%
  策略A StyleGAN 总体准确率: 48.0%
  策略B StyleGAN 总体准确率: 94.0%
```

**成功标志**：`results/` 下生成三张对比图（PNG）和两个策略的 CSV 明细文件

---

### 快速复现路径（仅复现策略B 最优结果）

如果 `weights/` 下已有三个权重文件，只需：

```bash
# 1. 准备 CIFAKE 测试图像（如果 test_images_cifake/ 已存在可跳过）
python prepare_cifake.py

# 2. 运行统一评估
python eval_all_strategies.py
```

如果权重不存在，最短复现路径（仅训练策略B）：

```bash
# 1. 训练基础模型
python train_model.py

# 2. 混合训练（策略B）
python train_mixed.py

# 3. 准备 CIFAKE 测试图像
python prepare_cifake.py

# 4. 评估
python eval_all_strategies.py
```

---

## 5. 如果已有训练好的权重

如果 `weights/` 目录下已有以下文件，可以**完全跳过所有训练步骤**，直接从推理开始：

| 权重文件 | 大小 | 对应策略 |
|----------|------|---------|
| `weights/efficientnet_b7_deepfake.pth` | ~245 MB | Baseline（StyleGAN 训练域） |
| `weights/efficientnet_b7_cifake_finetuned.pth` | ~245 MB | 策略A（CIFAKE 专项微调） |
| `weights/efficientnet_b7_mixed.pth` | ~245 MB | 策略B（混合训练，推荐） |

**跳过训练后从哪一步开始：**

- 如果只想看跨域对比分析 → 从**步骤 3**（prepare_cifake.py）开始
- 如果只想看三策略评估对比 → 从**步骤 3**开始，然后直接跳到**步骤 8**（eval_all_strategies.py）
- 如果想生成 Grad-CAM 热力图 → 直接运行 `python gradcam_vis.py`

---

## 6. 输出文件说明

| 文件名 | 内容说明 |
|--------|---------|
| `results/predictions.csv` | Baseline 对 StyleGAN 测试图像的逐张推理结果（含 fake_prob、pred_label、correct） |
| `results/predictions_cifake.csv` | Baseline 对 CIFAKE 测试图像的逐张推理结果（跨域零样本） |
| `results/predictions_cifake_strategyA.csv` | 策略A 对 CIFAKE 测试图像的逐张推理结果 |
| `results/predictions_cifake_strategyB.csv` | 策略B 对 CIFAKE 测试图像的逐张推理结果 |
| `results/compare_distribution.png` | Fake 概率分布直方图：StyleGAN 域 vs CIFAKE 域对比 |
| `results/compare_boxplot.png` | Fake 概率箱线图：StyleGAN-REAL / StyleGAN-FAKE / SD-REAL / SD-FAKE 四组 |
| `results/strategy_comparison_accuracy.png` | 三策略准确率柱状图（REAL/FAKE/总体） |
| `results/strategy_comparison_separation.png` | 三策略 FAKE 组均值 vs REAL 组均值折线图（分离度趋势） |
| `results/strategy_comparison_boxplot.png` | 三策略 Fake 概率箱线图对比 |
| `heatmaps/REAL/` | Grad-CAM 热力图（真实图像，模型关注区域可视化） |
| `heatmaps/FAKE/` | Grad-CAM 热力图（假图像，模型关注区域可视化） |

---

## 7. 常见问题排查

### 7.1 GPU / CUDA 相关

---

**Q：运行时显示"使用 CPU 训练"，但我有 NVIDIA GPU**

A：这是 PyTorch 版本与 CUDA 不匹配的问题。按以下顺序诊断：

```bash
# 步骤1：查看系统 CUDA 版本
nvidia-smi

# 步骤2：查看 PyTorch 识别到的 CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

**常见原因和修复：**

| 现象 | 原因 | 修复方法 |
|------|------|---------|
| `cuda.is_available()` 为 False，GPU 是 RTX 5070 | 稳定版 PyTorch 不支持 Blackwell 架构 | 安装方案 C（nightly cu128） |
| `cuda.is_available()` 为 False，其他 GPU | PyTorch 版本与 CUDA 不匹配 | 对照速查表重装 PyTorch |
| `cuda.is_available()` 为 False，`nvidia-smi` 也无法运行 | 未安装 NVIDIA 驱动 | 前往 NVIDIA 官网下载并安装驱动 |
| `cuda.is_available()` 为 True，但脚本仍用 CPU | 未激活虚拟环境 | 先激活虚拟环境再运行脚本 |

检查当前 Python 是否在虚拟环境中：
```bash
# Linux / macOS / 云服务器
which python

# Windows
where python
```
路径中应包含 `.venv` 或你创建的虚拟环境名称。

---

**Q：CUDA out of memory（显存不足 / OOM）**

A：将 `BATCH_SIZE` 从 16 降低，各脚本修改位置：

| 脚本 | 位置 | 当前值 | 建议改为 |
|------|------|--------|---------|
| `train_model.py` | 第 39 行 `batch_size=16` | 16 | 8（6GB 显存）/ 4（4GB 显存） |
| `finetune_cifake.py` | 第 10 行 `BATCH_SIZE = 16` | 16 | 同上 |
| `train_mixed.py` | 第 9 行 `BATCH_SIZE = 16` | 16 | 同上 |

**不同显存的推荐配置：**

| 显卡显存 | 推荐 BATCH_SIZE |
|---------|----------------|
| 16 GB 以上 | 16（默认） |
| 8 GB（如 RTX 3070/4060） | 8 |
| 6 GB（如 RTX 3060） | 4 ～ 6 |
| 4 GB 及以下 | 2 ～ 4（速度较慢） |

---

**Q：腾讯云 / 云服务器上 `nvidia-smi` 可以运行，但 Python 里 CUDA 不可用**

A：这通常是因为云实例的 CUDA 驱动版本与安装的 PyTorch 不匹配，或使用了系统 Python 而非虚拟环境。

```bash
# 查看云实例上的 CUDA 版本
nvidia-smi | grep "CUDA Version"

# 查看 nvcc 编译器版本（有时与驱动版本不同）
nvcc --version 2>/dev/null || echo "nvcc 未安装（正常，只需要驱动版本）"

# 根据驱动 CUDA 版本重新安装 PyTorch
# CUDA 11.x → 方案A（cu118）
# CUDA 12.0-12.3 → 方案B（cu121）
# CUDA 12.4+ → 方案B（cu121 或 cu124）
```

---

**Q：macOS M1/M2/M3 运行时报错 `MPS backend out of memory` 或速度很慢**

A：macOS 使用 Apple MPS 后端，不是 CUDA。

```python
# macOS 上检查 MPS 是否可用
python -c "import torch; print(torch.backends.mps.is_available())"
```

若为 True，PyTorch 会自动使用 MPS。训练脚本里的 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` **不会**使用 MPS，如果需要在 M 系列 Mac 上加速，将该行改为：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

---

### 7.2 路径与文件相关

---

**Q：在 Linux / 云服务器上运行 `train_model.py` 报错 `No such file or directory: F:\HYH_LocalFile\...`**

A：`train_model.py`、`batch_predict.py`、`gradcam_vis.py` 三个脚本硬编码了 Windows 绝对路径。在非 Windows 环境运行前，需要修改这些路径。

`train_model.py` 修改（第 9～11 行）：
```python
# 删除原来三行，替换为：
from pathlib import Path
_root = Path(__file__).resolve().parent
TRAIN_DIR = str(_root / "data" / "real_vs_fake" / "real-vs-fake" / "train")
VALID_DIR = str(_root / "data" / "real_vs_fake" / "real-vs-fake" / "valid")
SAVE_PATH = str(_root / "weights" / "efficientnet_b7_deepfake.pth")
```

`batch_predict.py` 修改（第 9 行、第 53 行、第 61 行）：
```python
# 第 9 行
from pathlib import Path
WEIGHT_PATH = Path(__file__).resolve().parent / "weights" / "efficientnet_b7_deepfake.pth"

# 第 53 行
base = Path(__file__).resolve().parent / "test_images"

# 第 61 行（保存结果）
df.to_csv(Path(__file__).resolve().parent / "results" / "predictions.csv", index=False)
```

`gradcam_vis.py` 修改（第 12 行、第 31 行、第 62 行）：同理，将硬编码路径改为 `Path(__file__).resolve().parent / ...`。

---

**Q：找不到数据集路径（FileNotFoundError：data/train/FAKE 不存在）**

A：检查数据集解压结构。最常见的问题是解压后多了一层目录。

```bash
python -c "
from pathlib import Path
paths = [
    'data/real_vs_fake/real-vs-fake/train/fake',
    'data/real_vs_fake/real-vs-fake/train/real',
    'data/real_vs_fake/real-vs-fake/valid/fake',
    'data/real_vs_fake/real-vs-fake/valid/real',
    'data/train/FAKE',
    'data/train/REAL',
    'data/test/FAKE',
    'data/test/REAL',
]
ok = True
for p in paths:
    exists = Path(p).exists()
    ok = ok and exists
    mark = 'OK' if exists else 'MISSING'
    print(f'[{mark}] {p}')
print()
print('全部就绪' if ok else '存在缺失，请检查解压步骤')
"
```

注意：StyleGAN 数据集子目录为**小写**（`fake/`、`real/`），CIFAKE 子目录为**大写**（`FAKE/`、`REAL/`）。

---

**Q：`prepare_cifake.py` 报错"测试目录不存在"**

A：脚本依赖 `data/test/` 目录。确认 CIFAKE 数据集已正确解压，且 `data/test/` 下存在 `REAL/` 和 `FAKE/` 两个子目录。

---

**Q：在 Colab / 腾讯云上权重文件不存在**

A：权重文件（`weights/*.pth`，共约 750 MB）不在代码仓库里，需要从原始机器传输过去。有三种方案：

1. **手动上传**：使用 `scp`（云服务器）或 Colab 的文件上传功能
2. **Google Drive 中转**：将三个 `.pth` 文件上传到 Google Drive，再在 Colab/云服务器上下载
3. **重新训练**：按第 4 节步骤 1、6、7 重新训练（需要 GPU 且耗时较长）

---

### 7.3 依赖包相关

---

**Q：ModuleNotFoundError: No module named 'timm'**

```bash
pip install timm==1.0.25
```

---

**Q：ModuleNotFoundError: No module named 'grad_cam' 或 'pytorch_grad_cam'**

```bash
pip install grad-cam==1.5.5
```

---

**Q：ImportError: cannot import name 'XXX' from 'timm'（timm 版本不对）**

A：本项目使用 timm 1.0.25，新版接口有变化。

```bash
pip install timm==1.0.25 --force-reinstall
```

---

**Q：`matplotlib` 显示报错 `cannot connect to X server` （云服务器 / 无头环境）**

A：云服务器通常没有图形界面。本项目的绘图脚本已使用 `matplotlib.use("Agg")` 切换到无头后端（不需要显示器），只输出 PNG 文件，不会弹出窗口。

若仍报错，在脚本开头手动添加：
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

---

**Q：`UnicodeDecodeError` 或中文乱码（Windows CMD 环境）**

A：部分脚本已包含 `sys.stdout.reconfigure(encoding="utf-8")`，但 Windows CMD 默认编码为 GBK。解决方案：

```powershell
# 方法1：改用 PowerShell（默认 UTF-8）
# 方法2：在 CMD 中设置编码
chcp 65001
python train_model.py
```

---

## 8. 核心实验结论

以下是三个策略的最终对比结果：

| 策略 | CIFAKE REAL 准确率 | CIFAKE FAKE 准确率 | CIFAKE 总体准确率 | StyleGAN 总体准确率 | 分离度 |
|------|-------------------|-------------------|------------------|--------------------|----|
| Baseline（无跨域适配） | 28% | 58% | 43% | 81% | −0.15 |
| 策略A（CIFAKE 专项微调） | 92% | 84% | 88% | 48% | +0.60 |
| 策略B（混合训练，推荐） | **98%** | **100%** | **99%** | **94%** | **+0.92** |

**结论**：
- Baseline 在 CIFAKE 上仅有 43% 准确率，接近随机猜测，跨域泛化失败
- 策略A 大幅提升 CIFAKE 准确率至 88%，但代价是 StyleGAN 准确率从 81% 下降至 48%（灾难性遗忘）
- 策略B 通过两域联合训练，CIFAKE 准确率达 99%，同时 StyleGAN 准确率仍有 94%，实现最佳平衡

---

## 9. 如何将本模块集成到多模态总体项目

本节面向需要将图像检测模块与文本检测、视频检测模块合并为统一系统的组员。

### 9.1 本模块在总体项目中的定位

**本模块的职责边界：**
- **输入**：单张图像（JPG / PNG）
- **输出**：`fake_prob`（0 到 1 之间的浮点数）和 `pred_label`（REAL / FAKE）
- **不负责**：文本处理、视频抽帧、多模态融合、最终判断

**调用关系示意：**

```
文本检测模块 ──┐
图像检测模块 ──┼──→ 融合层 ──→ 最终输出（是否为 AI 生成内容）
视频检测模块 ──┘
```

### 9.2 对接接口：如何调用本模块的推理功能

**推荐封装方式**：在项目根目录新建 `image_detector.py`，暴露一个 `predict(image_path)` 函数。

```python
# image_detector.py —— 供其他模块调用的推理接口

from pathlib import Path
import torch
import timm
from torchvision import transforms
from PIL import Image

# 使用策略B权重（两域兼顾，推荐用于生产集成）
WEIGHT_PATH = Path(__file__).parent / "weights" / "efficientnet_b7_mixed.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 模型在模块级别加载一次，避免每次调用都重新加载（约需数秒）
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = timm.create_model("tf_efficientnet_b7", pretrained=False, num_classes=1)
_model.load_state_dict(torch.load(WEIGHT_PATH, map_location=_device))
_model = _model.to(_device)
_model.eval()


def predict(image_path: str) -> dict:
    """
    输入图像路径，返回检测结果。
    返回格式：
    {
        "fake_prob": float,    # 0.0 ~ 1.0，越高越像假图
        "pred_label": str,     # "FAKE" 或 "REAL"
        "confidence": float    # max(fake_prob, 1-fake_prob)，置信度
    }
    """
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        real_prob = torch.sigmoid(_model(tensor)).item()
    fake_prob = 1.0 - real_prob
    return {
        "fake_prob": round(fake_prob, 4),
        "pred_label": "FAKE" if fake_prob > 0.5 else "REAL",
        "confidence": round(max(fake_prob, 1 - fake_prob), 4),
    }
```

**调用示例：**

```python
from image_detector import predict

result = predict("path/to/image.jpg")
print(result["fake_prob"])    # 0.97
print(result["pred_label"])   # "FAKE"
print(result["confidence"])   # 0.97
```

**应使用哪个权重**：`weights/efficientnet_b7_mixed.pth`（策略B，同时兼顾 StyleGAN 和 Stable Diffusion 两个域，推荐用于集成部署）

**重要**：模型权重文件约 **245 MB**，加载时间约 3～5 秒。上面的写法将模型作为模块级变量，在 `import image_detector` 时加载一次，后续所有 `predict()` 调用共享同一个模型实例，不会重复加载。

### 9.3 视频检测模块的对接方式

视频本质上是帧序列，图像模块只负责处理单帧。

**对接思路：**

1. 视频抽帧由视频模块负责，图像模块接收单张帧图像
2. 视频模块对每一帧调用 `predict()`，收集所有帧的 `fake_prob`

```python
# 视频模块中的伪代码
from image_detector import predict

frame_probs = []
for frame_path in extracted_frames:
    result = predict(frame_path)
    frame_probs.append(result["fake_prob"])

# 选择汇总策略（见下）
fake_prob_video = sum(frame_probs) / len(frame_probs)  # 均值策略
```

**帧级汇总策略（三选一）：**

| 策略 | 计算方式 | 适用场景 |
|------|---------|---------|
| 均值 | `mean(所有帧的 fake_prob)` | 生成内容均匀分布的视频 |
| 最大值 | `max(所有帧的 fake_prob)` | 只要有一帧造假就判定为假的严格场景 |
| 滑动窗口 | 对连续 N 帧取均值后再取最大值 | 过滤单帧噪声，更稳健 |

**抽帧频率建议**：每秒抽 1～3 帧，在速度和覆盖率之间取平衡。具体由视频模块决定，图像模块不参与。

### 9.4 与文本检测模块的结果融合

多模态融合由专门的融合层负责。以下方案供负责融合层的组员参考：

**方式一：规则融合（最简单，推荐作为 baseline）**
```python
final_score = w1 * text_fake_prob + w2 * image_fake_prob + w3 * video_fake_prob
# w1 + w2 + w3 = 1，初始可设为均等权重（各 1/3）
```

**方式二：取最大值**
```python
final_score = max(text_fake_prob, image_fake_prob, video_fake_prob)
# 任一模态判定为假则整体判定为假，适合高召回场景
```

**方式三：置信度加权**
```python
scores = []
if image_confidence > 0.8:
    scores.append(image_fake_prob)
if text_confidence > 0.8:
    scores.append(text_fake_prob)
# 只有高置信度的模态参与最终计算
final_score = sum(scores) / len(scores) if scores else 0.5
```

**量纲一致性说明**：本模块输出的 `fake_prob` 是经过 sigmoid 归一化的概率值（0～1）。若文本模块输出的是 perplexity（困惑度）或 log-prob（对数概率），**需要由融合层做归一化处理**，这不是本模块的责任。

### 9.5 集成时需要注意的事项

1. **模型加载耗时**：EfficientNet-B7 权重文件约 **245 MB**，首次加载需要 3～5 秒。建议在服务启动时预加载（`import image_detector` 即触发加载），不要在每次请求时重新加载。

2. **GPU 资源共享**：若文本、图像、视频模块都需要 GPU，建议明确分配显存或使用队列串行推理，避免同时加载多个大模型导致显存溢出（OOM）。

3. **输入格式要求**：本模块只接受 RGB 图像。若上游传入 RGBA（带透明通道）或灰度图，需要在调用前转换：
   ```python
   from PIL import Image
   img = Image.open(path).convert("RGB")  # 统一转为 RGB
   ```

4. **批量推理优化**：如果需要同时处理多张图像（如视频抽帧场景），建议传入 batch 而不是逐张调用，可将 `batch_size` 设为 8～16，速度显著快于逐张推理。

5. **跨域局限性**：本模块在 StyleGAN 和 Stable Diffusion 两类图像上准确率为 94～99%，但对其他生成方式（如 Midjourney、DALL-E 3、Sora 生成的视频帧）尚未测试。集成后建议在总体项目中标注此局限性。

### 9.6 建议的集成测试方案

集成完成后，按以下顺序验证图像模块工作正常：

**1. 单元测试**（验证输出格式）

```python
from image_detector import predict
import os

for label, folder in [("REAL", "test_images/real"), ("FAKE", "test_images/fake")]:
    files = os.listdir(folder)[:5]
    for f in files:
        result = predict(os.path.join(folder, f))
        assert 0.0 <= result["fake_prob"] <= 1.0, "fake_prob 超出范围"
        assert result["pred_label"] in ("FAKE", "REAL"), "pred_label 格式错误"
        print(f"[{label}] {f}: fake_prob={result['fake_prob']}, pred={result['pred_label']}")
```

**2. 回归测试**（验证准确率未退化）

```bash
# 确认策略B的准确率仍为 ~99%
python eval_all_strategies.py
```

若数字有明显变化，说明集成过程中改动了推理逻辑，需要排查。

**3. 联调测试**（验证与其他模块的协作）

准备一份多模态测试样本（同一内容的文字描述 + 对应图像），分别输入各模块，检查融合层输出是否符合预期。

---

*文档最后更新：2026-04-23*  
*对应代码版本：参见 git log（最新 commit：`3c5c3e4`）*  
*环境兼容性覆盖：Windows / Linux / macOS、RTX 50/40/30 系、Tesla T4/V100/A100、腾讯云 GN 系列、Google Colab、CPU-only*
