#!/usr/bin/env bash
#SBATCH --job-name=thoughtcomm-adapter
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G a100:2
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 7-0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@60

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR}"
cd "${REPO_DIR}"
mkdir -p logs

# Keep the runtime's caches and any tokenizer metadata off the home quota.
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/xdg_cache}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Set these in the submission environment; never put credentials in this file.
: "${THOUGHTCOMM_MODEL_PATH:?Set THOUGHTCOMM_MODEL_PATH to a local cached Qwen3 snapshot}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Slurm assigns the allocated GPU(s) and sets CUDA_VISIBLE_DEVICES.
# Full-context runs use two A100s; set THOUGHTCOMM_SINGLE_GPU=1 only for a one-card smoke.
export THOUGHTCOMM_SINGLE_GPU="${THOUGHTCOMM_SINGLE_GPU-}"

export THOUGHTCOMM_MODEL_DTYPE="float32"
export THOUGHTCOMM_GRAD_CLIP="1"
export THOUGHTCOMM_PREFIX_INIT_SCALE="0.005"
export THOUGHTCOMM_ADAPTER_LR="1e-5"
export THOUGHTCOMM_STABLE_SAMPLING=1
export THOUGHTCOMM_STABLE_LOSS=1
export THOUGHTCOMM_EXTRA_SITE="${THOUGHTCOMM_EXTRA_SITE:-}"
export THOUGHTCOMM_EPOCH="${THOUGHTCOMM_EPOCH:-1}"
export THOUGHTCOMM_OUTPUT_DIR="${THOUGHTCOMM_OUTPUT_DIR:-artifacts/full_qwen3_exp1/adapter_cluster_a100/epoch${THOUGHTCOMM_EPOCH}}"
export THOUGHTCOMM_CONTEXTS_PATH="${THOUGHTCOMM_CONTEXTS_PATH:-artifacts/full_qwen3_exp1/merged/contexts.jsonl}"
export THOUGHTCOMM_STATES_PATH="${THOUGHTCOMM_STATES_PATH:-artifacts/full_qwen3_exp1/merged/states.pt}"
export THOUGHTCOMM_AE_CKPT="${THOUGHTCOMM_AE_CKPT:-artifacts/full_qwen3_exp1/ae/ae.pt}"
export THOUGHTCOMM_MASK_PATH="${THOUGHTCOMM_MASK_PATH:-artifacts/full_qwen3_exp1/structure/mask.pt}"
export THOUGHTCOMM_ADAPTER_CKPT="${THOUGHTCOMM_ADAPTER_CKPT:-}"
export THOUGHTCOMM_OVERRIDES="${THOUGHTCOMM_OVERRIDES:-model.attn_implementation=sdpa adapter.epochs=1 adapter.grad_accum=1}"

exec "${PYTHON_BIN}" scripts/run_adapter_2gpu_runtime.py
