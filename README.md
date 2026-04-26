# AI-generated-image-detection

Community-Forensics zero-shot detection workspace for STAT8307, with a completed
"small innovation" track: adaptive threshold + temperature calibration.

## 1) Repository purpose

- Reproduce zero-shot inference using official checkpoint:
  `OwensLab/commfor-model-224`
- Run local parquet evaluation for unstable-network environments
- Produce final deployable decision config:
  `sigmoid(logit / temperature) >= threshold`

## 2) Source-of-truth documents

- Execution entry (this file): `README.md`
- Full Chinese implementation + handover playbook: `PROJECT_PLAYBOOK_ZH.md`
- Final experiment report (English): `RESULTS.md`
- Final experiment report (Chinese): `REPORT_ZH.md`

## 3) Environment setup

```bash
cd /root
if [ ! -d Community-Forensics ]; then
  curl -L -o Community-Forensics-main.zip \
    https://codeload.github.com/JeongsooP/Community-Forensics/zip/refs/heads/main
  unzip -q Community-Forensics-main.zip
  mv Community-Forensics-main Community-Forensics
fi

python3 -m pip install torchmetrics datasets timm wandb opencv-python \
  pandas huggingface-hub transformers scikit-learn pyarrow
```

## 4) Recommended runtime env vars

```bash
export HF_HOME=/root/AI-generated-image-detection/.hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export COMMFOR_REPO=/root/Community-Forensics
```

## 5) Run commands

### 5.1 Official zero-shot eval (full HF pipeline)

```bash
cd /root/AI-generated-image-detection
BATCH_SIZE_PER_GPU=32 ./pipelines/run_zero_shot_eval_224.sh
```

### 5.2 Local zero-shot eval (cached parquet, more stable)

```bash
cd /root/AI-generated-image-detection
EVAL_DATA_DIR=/root/AI-generated-image-detection/.hf_cache/hub/datasets--OwensLab--CommunityForensics-Eval/snapshots/7d4a74a88d2cac93b513c0853bf92c260eaceea0/data \
python3 pipelines/local_eval.py
```

### 5.3 Innovation run (adaptive threshold + calibration)

```bash
cd /root/AI-generated-image-detection
RESULT_DIR=/root/AI-generated-image-detection/results \
python3 pipelines/adaptive_threshold_calibrated_eval.py
```

## 6) Final artifacts

- Decision config: `results/final_decision_config.json`
- Full metrics report: `results/adaptive_calibration_report.json`
- Human-readable summaries: `RESULTS.md`, `REPORT_ZH.md`

## 7) Demo (Gradio)

Run a minimal web demo that uses the finalized decision config:

```bash
cd /root/AI-generated-image-detection
export COMMFOR_REPO=/root/Community-Forensics
python3 demo_app.py
```

Then open:

- `http://127.0.0.1:7860`

Demo outputs:

- Predicted label (`AI-generated` / `Real`)
- Calibrated probability
- Threshold and confidence
- JSON details (logit / temperature / decision rule)

## 8) Practical notes

- If download is unstable, keep `HF_ENDPOINT=https://hf-mirror.com`.
- If OOM occurs, reduce `BATCH_SIZE_PER_GPU` first.
- If you need strict full-set reproducibility, ensure all eval shards are cached,
  then rerun sections 5.1 and 5.3.
