#!/bin/bash
#SBATCH --job-name=bit
#SBATCH --output=bit_%j.out
#SBATCH --error=bit_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1

# Exit on any error
set -e

# Print job information
echo "=========================================="
echo "BitNet Evaluation Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Load required modules
echo "Loading modules..."
module purge
module load CUDA/12.8.0
module load Miniconda3/23.10.0-1

# Verify CUDA environment
echo "Verifying CUDA environment..."
if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: CUDA_HOME not set after loading CUDA module"
    exit 1
fi
echo "CUDA_HOME: $CUDA_HOME"

if ! command_exists nvcc; then
    echo "ERROR: nvcc not found in PATH"
    exit 1
fi
echo "NVCC version: $(nvcc --version | head -n 1)"

# Activate conda environment
echo "Activating conda environment..."
if ! command_exists conda; then
    echo "ERROR: conda command not found"
    exit 1
fi

source activate bit

# Verify conda environment
echo "Verifying conda environment..."
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "ERROR: Conda environment not activated"
    exit 1
fi
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Navigate to project directory
PROJECT_DIR="/home/axz402/bitnet-hpc/bitnet_b1_58-3B_quantized"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory $PROJECT_DIR not found"
    exit 1
fi
cd "$PROJECT_DIR"
echo "Changed to project directory: $(pwd)"

# Check if required files exist
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found"
    exit 1
fi

if [ ! -f "eval_ppl.py" ]; then
    echo "ERROR: eval_ppl.py not found"
    exit 1
fi

# Install requirements (excluding flash-attn which needs special handling)
echo "Installing requirements..."
pip install -r requirements.txt

# Install flash-attn separately with CUDA environment
echo "Installing flash-attn with CUDA support..."
pip install flash-attn==2.5.6

# Verify flash-attn installation
echo "Verifying flash-attn installation..."
python -c "import flash_attn; print('Flash-attn successfully imported!')" || {
    echo "ERROR: Failed to import flash_attn"
    exit 1
}

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"

# Run the evaluation script
echo "=========================================="
echo "Starting BitNet evaluation..."
echo "=========================================="
python eval_ppl.py --hf_path ./ --seqlen 2048 --max_dataset_size 1000

echo "=========================================="
echo "Job completed successfully at: $(date)"
echo "=========================================="
