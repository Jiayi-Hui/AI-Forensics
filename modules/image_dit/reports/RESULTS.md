# Experiment Results (Final, with Innovation)

Date: 2026-04-27  
Status: **Completed (baseline + adaptive threshold + calibration)**  
Model: `OwensLab/commfor-model-224` (ViT-Small, 224x224, patch 16)  
GPU: 1 x NVIDIA A100 40GB  
Eval source: `OwensLab/CommunityForensics-Eval` / `CompEval` (local cached parquet subset)  
Samples used in this run: **21,275**

---

## 1. Completion Summary

This round has delivered:

1. Official zero-shot model inference (`OwensLab/commfor-model-224`) on available cached eval data.
2. Innovation module: **adaptive threshold + temperature calibration**.
3. Final deployable decision config (threshold and temperature):
   - `results/final_decision_config.json`
   - `results/adaptive_calibration_report.json`

---

## 2. Baseline Result (Zero-Shot, threshold=0.5)

Runtime script: `pipelines/local_eval.py`

| Metric | Value |
|---|---|
| Accuracy | 0.8784 |
| Precision | 0.9565 |
| Recall | 0.8313 |
| F1 | 0.8895 |
| ROC-AUC | 0.9613 |
| Avg Precision (AP) | 0.9753 |
| Loss | 0.3589 |
| Examples | 21,275 |
| Elapsed | 549.6s |
| Batch size | 32 |

---

## 3. Innovation Result: Adaptive Threshold + Calibration

Runtime script: `pipelines/adaptive_threshold_calibrated_eval.py`  
Validation/Test split: `val_ratio=0.2`, `seed=42`

### 3.1 Test metrics comparison

| Setting | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | Brier | ECE(10) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (`t=0.5`) | 0.8771 | 0.9550 | 0.8304 | 0.8884 | 0.9608 | 0.9750 | 0.0965 | 0.0779 |
| Threshold only (`t=0.08`) | 0.8842 | 0.8873 | 0.9203 | 0.9035 | 0.9608 | 0.9750 | 0.0965 | 0.0779 |
| Threshold + calibration | **0.8844** | 0.8879 | 0.9198 | **0.9036** | 0.9608 | 0.9750 | **0.0862** | **0.0636** |

### 3.2 Key gains vs baseline (`t=0.5`)

- F1: `0.8884 -> 0.9036` (**+0.0152**)
- Accuracy: `0.8771 -> 0.8844` (**+0.0073**)
- ECE: `0.0779 -> 0.0636` (**better calibration**)
- Brier: `0.0965 -> 0.0862` (**better probability quality**)

---

## 4. Final Decision Model (Deliverable)

The final detector is still the official pretrained model:

- Backbone checkpoint: `OwensLab/commfor-model-224`
- Decision rule: `sigmoid(logit / temperature) >= threshold`
- Final parameters:
  - `temperature = 2.1098015308380127`
  - `threshold = 0.24`

Machine-readable artifact:

- `results/final_decision_config.json`

---

## 5. Notes and Scope

- This completion is based on currently cached eval parquet files (21,275 samples).
- Therefore, this is a **completed and valid project deliverable** for the current environment.
- If strict full-set reproducibility is required, rerun the same pipeline after all eval shards are fully cached.

---

## 6. Relevant Paths

- Project root: `/root/AI-generated-image-detection`
- Official repo path: `/root/Community-Forensics`
- HF cache root: `/root/AI-generated-image-detection/.hf_cache`
- Final config: `/root/AI-generated-image-detection/results/final_decision_config.json`
- Full report JSON: `/root/AI-generated-image-detection/results/adaptive_calibration_report.json`
