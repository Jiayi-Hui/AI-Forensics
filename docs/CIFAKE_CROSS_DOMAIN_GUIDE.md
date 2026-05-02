# CIFAKE 跨域泛化测试实验操作文档

## 1. 文档目的

本文档用于指导新成员在本项目中完整复现并汇报 CIFAKE 跨域泛化实验，包含：

- Baseline 跨域评估（StyleGAN 训练权重直接测 CIFAKE）
- 策略 A：CIFAKE 专项 Fine-tune
- 策略 B：StyleGAN + CIFAKE 混合域训练
- 策略 C：统一评估三套权重并生成对比图

文档同时包含常见环境问题（尤其是 GPU 解释器不匹配）排查流程。

---

## 2. 项目与数据约定

项目根目录：

`F:\HYH_LocalFile\8307_GroupProject`

关键数据目录：

- `data/real_vs_fake/real-vs-fake/train/{real,fake}`（StyleGAN 训练域）
- `data/real_vs_fake/real-vs-fake/valid/{real,fake}`（StyleGAN 验证域）
- `data/train/{REAL,FAKE}`（CIFAKE 训练）
- `data/test/{REAL,FAKE}`（CIFAKE 测试）
- `test_images_cifake/{real,fake}`（CIFAKE 小样本测试集，50+50）
- `test_images/{real,fake}`（StyleGAN 小样本测试集，50+50）

关键权重与结果：

- Baseline 权重：`weights/efficientnet_b7_deepfake.pth`
- 策略 A 权重：`weights/efficientnet_b7_cifake_finetuned.pth`
- 策略 B 权重：`weights/efficientnet_b7_mixed.pth`
- Baseline CIFAKE 结果：`results/predictions_cifake.csv`

---

## 3. 代码脚本说明（本实验相关）

- `prepare_cifake.py`：从 CIFAKE test 集随机抽样 50 REAL + 50 FAKE 到 `test_images_cifake/`
- `finetune_cifake.py`：策略 A（低学习率 + 部分冻结）
- `train_mixed.py`：策略 B（混合域训练，全参数）
- `eval_all_strategies.py`：策略 C（统一评估 + 生成图表）

---

## 4. 先决条件与环境要求

推荐硬件：

- NVIDIA GPU（本实验使用 RTX 5070）

推荐 Python 环境：

- 需保证 `torch.cuda.is_available()` 返回 `True`
- 推荐 `torch` 为 CUDA 版本（例如 `2.x.x+cuXXX` 或 `2.x.x.dev...+cuXXX`）

常用依赖：

- `torch`, `torchvision`, `timm`, `pandas`, `matplotlib`, `Pillow`

---

## 5. GPU 环境诊断（必须先做）

### 5.1 快速检测

在项目根目录执行：

```powershell
python test_GPU.py
```

若输出 `False` 或显示“无GPU”，不要直接开始训练。

### 5.2 定位正确解释器

1. 查看系统解释器：

```powershell
where.exe python
```

2. 递归查找候选：

```powershell
where.exe /R C:\Users python.exe
```

3. 对每个候选执行：

```powershell
<python路径> -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

4. 选择满足 `cuda=True` 的解释器，记为 `PYTHON_PATH`。

### 5.3 本项目已验证可用解释器（示例）

```text
C:\Users\CodexSandboxOffline\.codex\.sandbox\cwd\d03cc8e159098882\.venv\Scripts\python.exe
```

对应验证输出示例：

```text
torch 版本: 2.12.0.dev20260320+cu128
CUDA 可用: True
GPU 名称: NVIDIA GeForce RTX 5070
timm 可用: True
```

后续所有训练/评估命令都应使用该 `PYTHON_PATH`。

---

## 6. 实验流程总览

按顺序执行：

1. （可选）准备 CIFAKE 50+50 测试子集
2. 策略 A：`finetune_cifake.py`
3. 策略 B：`train_mixed.py`
4. 策略 C：`eval_all_strategies.py`

---

## 7. 详细操作步骤

## 7.1 步骤 0：进入项目目录

```powershell
cd F:\HYH_LocalFile\8307_GroupProject
```

## 7.2 步骤 1（可选）：重新准备 CIFAKE 测试子集

如果 `test_images_cifake/` 已存在且数量正确，可跳过。

```powershell
<PYTHON_PATH> prepare_cifake.py
```

预期输出要点：

- REAL 可用图像约 10000
- FAKE 可用图像约 10000
- `test_images_cifake/real` = 50
- `test_images_cifake/fake` = 50

## 7.3 步骤 2：策略 A（CIFAKE 专项微调）

```powershell
<PYTHON_PATH> finetune_cifake.py
```

脚本做了什么：

- 从 `data/train/REAL` 和 `data/train/FAKE` 各取 3000 训练 + 500 验证（seed=42）
- 使用 `weights/efficientnet_b7_deepfake.pth` 初始化
- 冻结 `conv_stem/bn1/blocks.0~4`，训练 `blocks.5~` + 分类头
- 训练 5 epoch，保存最优验证准确率到：
  `weights/efficientnet_b7_cifake_finetuned.pth`

日志检查点：

- `训练设备: cuda`
- 每轮输出：
  `[Epoch N/5] Train Loss ... | Train Acc ... | Val Acc ... [saved/skipped]`

## 7.4 步骤 3：策略 B（混合域训练）

```powershell
<PYTHON_PATH> train_mixed.py
```

脚本做了什么：

- StyleGAN train 各采样 2000（real/fake）
- CIFAKE train 各采样 2000（REAL/FAKE）
- 混合训练总计 8000
- 混合验证集 1600（StyleGAN valid 各400 + CIFAKE test 各400）
- 全参数训练 8 epoch，保存最优 mixed val 到：
  `weights/efficientnet_b7_mixed.pth`

日志检查点：

- 打印 class_to_idx 核对标签方向：
  - StyleGAN: `{'fake': 0, 'real': 1}`
  - CIFAKE: `{'FAKE': 0, 'REAL': 1}`
- 每轮输出：
  `[Epoch N/8] Loss ... | Mixed Val ... | StyleGAN Val ... | CIFAKE Val ...`

## 7.5 步骤 4：策略 C（统一评估与作图）

```powershell
<PYTHON_PATH> eval_all_strategies.py
```

脚本做了什么：

- 在同一 CIFAKE 50+50 测试集上评估三套权重：
  1. Baseline：`efficientnet_b7_deepfake.pth`
  2. 策略 A：`efficientnet_b7_cifake_finetuned.pth`
  3. 策略 B：`efficientnet_b7_mixed.pth`
- 输出策略对比表、分布分离度结论
- 附加评估 `test_images/`（StyleGAN）用于遗忘检查
- 生成结果：
  - `results/predictions_cifake_strategyA.csv`
  - `results/predictions_cifake_strategyB.csv`
  - `results/strategy_comparison_accuracy.png`
  - `results/strategy_comparison_separation.png`
  - `results/strategy_comparison_boxplot.png`

---

## 8. 本项目一次实测结果（可作为参考基线）

在当前仓库一次实际运行中得到：

### 8.1 CIFAKE 跨域结果

- Baseline：REAL 28.0%，FAKE 58.0%，总体 43.0%
- 策略 A：REAL 92.0%，FAKE 84.0%，总体 88.0%
- 策略 B：REAL 98.0%，FAKE 100.0%，总体 99.0%

### 8.2 分离度（FAKE均值 - REAL均值）

- Baseline：-0.1503（反转）
- 策略 A：+0.5968（分离良好）
- 策略 B：+0.9229（分离良好）

### 8.3 StyleGAN 附加评估（遗忘检查）

- Baseline：81.0%
- 策略 A：48.0%（遗忘明显）
- 策略 B：94.0%（兼顾两域）

结论：在该次实验下，策略 B（混合训练）是最佳折中与最佳性能方案。

---

## 9. 汇报建议模板（可直接复制）

### 9.1 实验目的

验证 StyleGAN 训练模型在 Stable Diffusion（CIFAKE）域的泛化能力，并通过微调与混合训练改善跨域性能。

### 9.2 关键发现

1. Baseline 在 CIFAKE 上出现概率分布反转，泛化失败（43%）。
2. 策略 A 能显著修复跨域表现（88%），但对原域遗忘明显。
3. 策略 B 同时提升 CIFAKE 与 StyleGAN，综合表现最佳（CIFAKE 99%，StyleGAN 94%）。

### 9.3 结论

建议将混合域训练（策略 B）作为后续默认训练方案；策略 A 适合作为快速域适配手段，但需配套遗忘抑制机制。

---

## 10. 常见问题与排错

## 10.1 训练跑在 CPU

现象：

- `torch.cuda.is_available()` 为 `False`
- 日志显示 `训练设备: cpu`

处理：

- 重新执行第 5 节“GPU 环境诊断”，锁定正确 `PYTHON_PATH`
- 用 `PYTHON_PATH` 启动脚本，不要用裸 `python`

## 10.2 matplotlib 中文字体警告

现象：

- `UserWarning: Glyph ... missing from font(s) DejaVu Sans`

影响：

- 不影响数值结果，只影响图中中文字符显示

处理（可选）：

- 安装并配置中文字体（如 SimHei / Microsoft YaHei），或改图中文字为英文

## 10.3 权重文件不存在

检查：

- `weights/efficientnet_b7_deepfake.pth` 是否存在
- 策略 A/B 是否已训练完成并生成对应权重

---

## 11. 一键命令清单（按顺序）

```powershell
cd F:\HYH_LocalFile\8307_GroupProject

# 0) 检查 GPU
<PYTHON_PATH> test_GPU.py

# 1) （可选）准备 CIFAKE 50+50
<PYTHON_PATH> prepare_cifake.py

# 2) 策略A
<PYTHON_PATH> finetune_cifake.py

# 3) 策略B
<PYTHON_PATH> train_mixed.py

# 4) 统一评估与作图
<PYTHON_PATH> eval_all_strategies.py
```

---

## 12. 输出物核对清单

- `weights/efficientnet_b7_cifake_finetuned.pth`
- `weights/efficientnet_b7_mixed.pth`
- `results/predictions_cifake_strategyA.csv`
- `results/predictions_cifake_strategyB.csv`
- `results/strategy_comparison_accuracy.png`
- `results/strategy_comparison_separation.png`
- `results/strategy_comparison_boxplot.png`

全部存在即表示本实验流程复现完成。

