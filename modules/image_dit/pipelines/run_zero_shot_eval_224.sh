#!/usr/bin/env bash
set -euo pipefail

# Zero-shot inference with official Community-Forensics 224 model.
# Prerequisite: clone https://github.com/JeongsooP/Community-Forensics
# and install dependencies in that repository.

COMMFOR_REPO="${COMMFOR_REPO:-/path/to/Community-Forensics}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_CPUS_PER_GPU="${NUM_CPUS_PER_GPU:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
MASTER_PORT="${MASTER_PORT:-23949}"
HF_MODEL_REPO="${HF_MODEL_REPO:-OwensLab/commfor-model-224}"
HF_TEST_REPO="${HF_TEST_REPO:-OwensLab/CommunityForensics-Eval}"
HF_TEST_SPLIT="${HF_TEST_SPLIT:-CompEval}"

if [[ ! -d "$COMMFOR_REPO" ]]; then
  echo "COMMFOR_REPO does not exist: $COMMFOR_REPO"
  echo "Set COMMFOR_REPO to the cloned Community-Forensics path."
  exit 1
fi

cd "$COMMFOR_REPO"

export OMP_NUM_THREADS="$NUM_CPUS_PER_GPU"
export MASTER_ADDR=localhost
export MASTER_PORT

torchrun --nproc_per_node="$NUM_GPUS" --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d eval.py \
  --gpus "$NUM_GPUS" \
  --cpus-per-gpu "$NUM_CPUS_PER_GPU" \
  --batch_size "$BATCH_SIZE_PER_GPU" \
  --hf_model_repo "$HF_MODEL_REPO" \
  --huggingface_test_repo "$HF_TEST_REPO" \
  --hf_split_test "$HF_TEST_SPLIT" \
  --verbose 2
