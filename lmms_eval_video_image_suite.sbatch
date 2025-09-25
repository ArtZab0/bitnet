#!/bin/bash -l

#SBATCH --job-name=bitnet-video-image-suite
#SBATCH --output=bitnet-video-image-suite-%j.out
#SBATCH --error=bitnet-video-image-suite-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

echo "========== SLURM ENV =========="
date
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "PWD=$(pwd)"

echo "\n========== MODULES =========="
if command -v module >/dev/null 2>&1; then
  module purge || true
  module --ignore-cache avail 2>&1 | head -n 50 | cat || true
  module load CUDA/12.8.0 || module load CUDA || module load cuda || true
  module list 2>&1 | cat || true
else
  echo "Environment Modules not available; skipping module load"
fi

echo "\n========== CONDA =========="
if command -v conda >/dev/null 2>&1; then
  :
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$(/bin/bash -lc 'conda info --base' 2>/dev/null)/etc/profile.d/conda.sh" ]; then
  base_dir="$(/bin/bash -lc 'conda info --base' 2>/devnull)"
  [ -n "$base_dir" ] && source "$base_dir/etc/profile.d/conda.sh" || true
fi

ENV_PATH=/home/axz402/conda_envs/lmms-eval
PY_VER=3.10

if command -v conda >/dev/null 2>&1; then
  if [ ! -d "$ENV_PATH" ]; then
    echo "Creating conda env at $ENV_PATH"
    conda create -y -p "$ENV_PATH" python="$PY_VER"
  fi
  conda activate "$ENV_PATH"
else
  echo "Conda not found. Please install/initialize conda."
  exit 1
fi

ENV_MARKER="$ENV_PATH/.lmms_eval_env_ready"

echo "which python: $(which python || echo not-found)"
echo "python --version: $(python --version 2>&1 || echo not-found)"
echo "pip --version: $(pip --version 2>&1 || echo not-found)"

if [ ! -f "$ENV_MARKER" ]; then
  echo "\n========== UPGRADE PIP/SETUPTOOLS/WHEEL =========="
  python -m pip install --upgrade pip setuptools wheel

  echo "\n========== INSTALL CUDA PYTORCH STACK =========="
  python - <<'PY'
import subprocess, sys
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])
for pkg in [
    "torch", "torchvision", "torchaudio", "triton",
    "nvidia-cuda-runtime-cu12", "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-cupti-cu12",
    "nvidia-cublas-cu12", "nvidia-cufft-cu12", "nvidia-curand-cu12",
    "nvidia-cusolver-cu12", "nvidia-cusparse-cu12", "nvidia-nvtx-cu12", "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12", "nvidia-cudnn-cu12"
]:
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
pip("install", "--extra-index-url", "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1+cu124", "torchvision==0.20.1+cu124", "torchaudio==2.5.1+cu124")
PY
else
  echo "Using cached environment; verifying CUDA PyTorch stack"
  python - <<'PY'
import subprocess, sys
def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])
import torch
v = getattr(torch, "__version__", "")
if not v.startswith("2.5.1") or "+cu124" not in v:
    print(f"Torch version {v} not compatible; will install 2.5.1+cu124")
    pip_install("install", "--upgrade", "--extra-index-url", "https://download.pytorch.org/whl/cu124",
               "torch==2.5.1+cu124", "torchvision==0.20.1+cu124", "torchaudio==2.5.1+cu124")
PY
fi

echo "\n========== SOURCE: lmms-eval =========="
cd /home/axz402/bitnet-hpc
if [ -d lmms-eval/.git ]; then
  echo "Using local lmms-eval source; skipping git reset to preserve local edits"
else
  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
fi

if [ ! -f "$ENV_MARKER" ]; then
  echo "\n========== INSTALL BASE DEPENDENCIES =========="
  cd /home/axz402/bitnet-hpc/lmms-eval
  python -m pip install -U "pip<25"
  python -m pip install \
    transformers==4.56.2 \
    accelerate>=0.29.1 \
    datasets>=2.19.0 \
    evaluate==0.4.0 \
    einops \
    ftfy \
    opencv-python-headless \
    av \
    hf_transfer \
    nltk \
    pillow \
    pydantic \
    pandas \
    scikit-learn \
    sacrebleu \
    tqdm \
    sentencepiece
else
  echo "Using cached environment; skipping base deps"
fi

if [ ! -f "$ENV_MARKER" ]; then
  echo "\n========== INSTALL lmms-eval (no torch pin) =========="
  python -m pip install -e . --no-deps || python -m pip install --no-deps git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
else
  echo "Using cached environment; skipping lmms-eval install"
fi

echo "\n========== COMMON ENV VARS =========="
export HF_HOME="${HF_HOME:-/home/axz402/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONPATH="/home/axz402/bitnet-hpc/lmms-eval:${PYTHONPATH:-}"
export AUTOGPTQ_DISABLE_TRITON_TUNING=1
export BITNET_DISABLE_TRITON=${BITNET_DISABLE_TRITON:-1}
echo "HF_HOME=$HF_HOME"
echo "PYTHONPATH=$PYTHONPATH"

echo "\n========== VERIFY MODEL REGISTRY (BITNET) =========="
python - <<'PY'
import sys
sys.path.insert(0, '/home/axz402/bitnet-hpc/lmms-eval')
from lmms_eval.models import AVAILABLE_SIMPLE_MODELS
print('has_bitnet_in_available_models:', 'bitnet' in AVAILABLE_SIMPLE_MODELS)
try:
    from lmms_eval.models.simple.bitnet import BitNet
    print('BitNet import OK')
except Exception as e:
    print('BitNet import failed:', e)
PY

echo "\n========== CUDA SANITY CHECK =========="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda.is_available", torch.cuda.is_available())
print("torch.version.cuda", getattr(torch.version, 'cuda', None))
if torch.cuda.is_available():
    print("device_count", torch.cuda.device_count())
    try:
        print("device_name0", torch.cuda.get_device_name(0))
    except Exception as e:
        print("device query error", e)
PY

echo "\n========== PREPARE OPTIONAL RUNTIME DEPS =========="
python - <<'PY'
import importlib.util, subprocess, sys
def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])
# Ensure core pins
pip_install("install", "tenacity==8.3.0", "protobuf==3.20.*", "numpy==1.26.4", "opencv-python-headless==4.8.1.78")
# Optional: AutoGPTQ for BitNet quant path (we disable Triton but package import may still occur)
need = importlib.util.find_spec("auto_gptq") is None
if need:
    pip_install("install", "--upgrade", "auto-gptq==0.7.1")
PY

cd /home/axz402/bitnet-hpc/lmms-eval

echo "\n========== DEFINE TASK GROUPS =========="
# Image tasks (vision-language)
IMAGE_TASKS=(
  mmbench mmvet gqa vqav2 textvqa ok_vqa docvqa chartqa mathvista mathverse
  coco_cap nocaps refcoco refcoco+ refcocog visualwebbench mmstar
)

# Video tasks (vision-language)
VIDEO_TASKS=(
  mvbench nextqa vatex youcook2_val moviechat video_detail_description longvideobench videomme videommmu videoevalpro
)

# Language-only control tasks (for backbone/quantization sensitivity):
# Drop openai_math (dataset script issues -> no docs). Keep stable text tasks.
LANG_TASKS=(hellaswag mmlu gsm8k)

# Paths to BitNet models: quantized and fp16 baseline (set via sbatch --export or edit below)
PRETRAINED_QUANT_DEFAULT="/home/axz402/bitnet-hpc/bitnet_b1_58-3B_quantized"
PRETRAINED_FP16_DEFAULT="/home/axz402/bitnet-hpc/bitnet_b1_58-3B_quantized"  # replace with fp16 baseline if available

PRETRAINED_QUANT="${PRETRAINED_QUANT:-$PRETRAINED_QUANT_DEFAULT}"
PRETRAINED_FP16="${PRETRAINED_FP16:-$PRETRAINED_FP16_DEFAULT}"

# Common CLI settings
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:-0}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda}"

echo "\n========== DETECT MODEL CAPABILITIES =========="
python - <<'PY'
import sys
sys.path.insert(0, '/home/axz402/bitnet-hpc/lmms-eval')
from lmms_eval.models.simple.bitnet import BitNet
print('BITNET_IS_MULTIMODAL:', getattr(BitNet, 'IS_MULTIMODAL', False))
PY

echo "\n========== RUN LANGUAGE BACKBONE SENSITIVITY (FP16 vs QUANT) =========="
LANG_TASKS_CSV=$(IFS=,; echo "${LANG_TASKS[*]}")
echo "Tasks: $LANG_TASKS_CSV"

echo "FP16 baseline (if available): $PRETRAINED_FP16"
python -m lmms_eval \
  --model bitnet \
  --model_args pretrained=$PRETRAINED_FP16,device=$DEVICE,torch_dtype=$DTYPE \
  --tasks "$LANG_TASKS_CSV" \
  --batch_size $BATCH_SIZE \
  ${LIMIT:+--limit $LIMIT} || true

echo "Quantized: $PRETRAINED_QUANT"
python -m lmms_eval \
  --model bitnet \
  --model_args pretrained=$PRETRAINED_QUANT,device=$DEVICE,torch_dtype=$DTYPE \
  --tasks "$LANG_TASKS_CSV" \
  --batch_size $BATCH_SIZE \
  ${LIMIT:+--limit $LIMIT} || true

echo "\n========== ATTEMPT MULTIMODAL (IMAGE) WITH BITNET: WILL SKIP (TEXT-ONLY) =========="
for t in "${IMAGE_TASKS[@]}"; do
  echo "Skipping image task $t for bitnet (text-only wrapper)."
done

echo "\n========== ATTEMPT MULTIMODAL (VIDEO) WITH BITNET: WILL SKIP (TEXT-ONLY) =========="
for t in "${VIDEO_TASKS[@]}"; do
  echo "Skipping video task $t for bitnet (text-only wrapper)."
done

echo "\nNOTE: To actually run the image/video suite, switch to a VLM model (e.g., 'llava_vid', 'internvideo2', 'videollama3', 'qwen2_5_vl') and set --model and --model_args appropriately, keeping the same IMAGE_TASKS/VIDEO_TASKS lists."

echo "\n========== DONE =========="
date


