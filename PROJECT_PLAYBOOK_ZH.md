# STAT8307 项目实施与交接手册（整合版）

## 1. 项目目标与边界

### 1.1 课程主目标

- 完成 AI 生成图像检测的可复现实验流程
- 以 Community-Forensics 的 ViT 路线为主线，验证 zero-shot 泛化能力
- 输出可交付结果：模型配置、指标、报告

### 1.2 当前仓库已完成范围

- 零样本推理复现（官方 checkpoint）
- 本地 parquet 评测链路
- 轻量创新：阈值自适应 + 温度校准
- 最终判别配置落盘（可部署）

### 1.3 暂未纳入当前代码范围

- EfficientNet-B7 + SE/CBAM 的完整训练实验
- Grad-CAM / Attention-map 可视化流水线
- 大规模消融与鲁棒性系统评测

---

## 2. 环境要求与迁移要点

### 2.1 推荐资源

- CUDA GPU：必需（建议 24GB+ 显存）
- 存储：建议 >= 500GB（最好 1TB）
- 内存：建议 >= 32GB

### 2.2 网络与镜像策略

- 若 Hugging Face 直连不稳定，使用：
  - `HF_ENDPOINT=https://hf-mirror.com`
- 统一缓存目录：
  - `HF_HOME=/root/AI-generated-image-detection/.hf_cache`

### 2.3 机器切换后最小检查

```bash
python3 - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
PY
df -h /
```

若 `cuda_available=False` 或磁盘不足，不建议直接启动实验。

---

## 3. 数据与划分策略（执行版）

### 3.1 数据来源

- 评测数据：`OwensLab/CommunityForensics-Eval`（CompEval）
- 模型：`OwensLab/commfor-model-224`

### 3.2 当前执行口径

- 优先跑通 zero-shot 推理闭环
- 允许先在“已缓存分片子集”完成完整方法验证
- 若课程要求“全量口径”，再补全全部分片后同流程复跑

### 3.3 创新评估划分

- 对同一批推理输出做 `val/test = 20%/80%`
- 固定随机种子：`42`
- 在 `val` 上拟合温度与阈值，在 `test` 上做最终汇报

---

## 4. 标准执行流程（可直接照跑）

### 4.1 环境准备

1. 准备官方仓库 `Community-Forensics`
2. 安装依赖（保持当前可用 PyTorch 环境）
3. 设置环境变量：`HF_HOME`、`HF_ENDPOINT`、`COMMFOR_REPO`

### 4.2 基线评测

- 官方链路（优先）：
  - `pipelines/run_zero_shot_eval_224.sh`
- 网络不稳定时：
  - `pipelines/local_eval.py`（本地 parquet）

### 4.3 创新模块执行

- 运行：`pipelines/adaptive_threshold_calibrated_eval.py`
- 输出：
  - `results/final_decision_config.json`
  - `results/adaptive_calibration_report.json`

### 4.4 报告输出

- 英文总报告：`RESULTS.md`
- 中文总报告：`REPORT_ZH.md`

---

## 5. 交付物清单（Definition of Done）

本项目至少应交付以下内容：

1. 可运行脚本与可复现命令
2. Zero-shot 指标（Accuracy / Precision / Recall / F1 / ROC-AUC / AP）
3. 最终模型判别配置（checkpoint + threshold + temperature）
4. 结果报告（中英文至少一份）
5. 关键运行信息（样本数、batch size、耗时、镜像是否使用）

---

## 6. 当前最终方案（2026-04 更新）

### 6.1 最终判别模型

- 模型：`OwensLab/commfor-model-224`
- 决策规则：`sigmoid(logit / temperature) >= threshold`
- 参数：
  - `temperature = 2.1098015308380127`
  - `threshold = 0.24`

### 6.2 创新价值

- 相比固定阈值，F1 与 Accuracy 有提升
- ECE 与 Brier 下降，概率输出更可靠、更适合取证报告解释

---

## 7. 后续扩展建议（按优先级）

1. 全量分片复跑，形成严格可比口径结果
2. 按生成器家族做分组阈值（family-specific threshold）
3. 补充鲁棒性测试（压缩、缩放、噪声、色偏）
4. 增加可解释性可视化（Attention map / Grad-CAM）
5. 逐步扩展到 EfficientNet-attention 路线做对照实验
