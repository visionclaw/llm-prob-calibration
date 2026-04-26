#!/bin/bash
#SBATCH --job-name=llm-prob-calibration
#SBATCH --partition=research
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/workspace/bryan/llm-prob-calibration/logs/%j.out
#SBATCH --error=/mnt/workspace/bryan/llm-prob-calibration/logs/%j.err
#SBATCH --time=04:00:00

set -euo pipefail

echo "=== LLM Probability Calibration Experiment ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo ""

# --- Configuration ---
PROJECT_DIR="/home/vision/llm-prob-calibration"
RESULTS_DIR="/mnt/workspace/bryan/llm-prob-calibration/results"
LOGS_DIR="/mnt/workspace/bryan/llm-prob-calibration/logs"
# Available models: Qwen/Qwen3-8B, google/gemma-3-12b-it
# Override: MODEL=google/gemma-3-12b-it sbatch slurm/run_experiment.sbatch
MODEL="${MODEL:-Qwen/Qwen3-8B}"

export HF_HOME="/mnt/workspace/.hf"
export UV_CACHE_DIR="/tmp/uv-cache-${USER}"
export PATH="$HOME/.local/bin:$PATH"

# --- Setup ---
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

cd "${PROJECT_DIR}"

# Create/activate venv
if [ ! -d ".venv" ]; then
    echo "[+] Creating virtual environment..."
    uv venv .venv
fi
source .venv/bin/activate

echo "[+] Installing dependencies..."
uv pip install -r requirements.txt

# --- Pre-download model ---
echo "[+] Ensuring model is cached: ${MODEL}"
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
token_path = '/mnt/workspace/.hf/token'
if os.path.exists(token_path):
    login(token=open(token_path).read().strip())
print('Checking tokenizer...')
AutoTokenizer.from_pretrained('${MODEL}', trust_remote_code=True)
print('Checking model...')
AutoModelForCausalLM.from_pretrained('${MODEL}', trust_remote_code=True, torch_dtype='auto')
print('Model cached.')
"

# --- Run experiment ---
echo ""
echo "[+] Running experiment..."
python src/run_experiment.py \
    --model "${MODEL}" \
    --output "${RESULTS_DIR}/$(echo ${MODEL} | tr '/' '-')" \
    --resume \
    --dtype bfloat16

# --- Run analysis ---
echo ""
echo "[+] Running analysis..."
FIGURES_DIR="${RESULTS_DIR}/figures"
mkdir -p "${FIGURES_DIR}"

python src/analyze.py \
    --results "${RESULTS_DIR}" \
    --output "${FIGURES_DIR}" \
    --no-show

echo ""
echo "=== Experiment Complete ==="
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "Figures: ${FIGURES_DIR}"
