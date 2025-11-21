#!/bin/bash
#SBATCH --job-name=mllm_house_search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --output=mllm_job_%j.out
#SBATCH --error=mllm_job_%j.err

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load CUDA modules
echo "Loading CUDA modules..."
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || echo "Warning: CUDA module not found"
module load cudnn/8.9.7 2>/dev/null || module load cudnn 2>/dev/null || echo "Warning: cuDNN module not found"
echo ""

# Verify code directory exists
CODE_PATH="/projects/scdatahub/amol_dmt/image_to_text_runs/MLLM-House-Search"
if [ ! -d "$CODE_PATH" ]; then
    echo "ERROR: Code directory not found at $CODE_PATH"
    exit 1
fi
echo "Code directory: $CODE_PATH"
echo ""

# Set paths and environment
export model_path="/projects/scdatahub/amol_dmt/model/8b"
export json_input_path="/projects/scdatahub/amol_dmt/image_to_text_runs/MLLM-House-Search/house_image_associations.json"
export output_path="/projects/scdatahub/amol_dmt/image_to_text_runs/output/house_descriptions.parquet"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
mkdir -p "/projects/scdatahub/amol_dmt/image_to_text_runs/output"

echo "Configuration:"
echo "  Model: $model_path"
echo "  Input: $json_input_path"
echo "  Output: $output_path"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate qwen-env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi
echo "Environment activated"
echo ""

# Verify GPU availability
echo "=========================================="
echo "GPU Verification"
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Exit if no GPU detected
python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: No GPU detected by PyTorch!"
    echo "Check: GPU allocation, CUDA modules, PyTorch CUDA support"
    exit 1
fi
echo "✓ GPU ready"
echo ""

# Run the script
echo "=========================================="
echo "Running image-to-text pipeline"
echo "=========================================="
cd "$CODE_PATH/image_to_text_pipeline"

python -u main.py \
    --input "$json_input_path" \
    --output "$output_path" \
    --model-path "$model_path" \
    --resume

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: Job completed"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Job failed with exit code $?"
    echo "=========================================="
    exit 1
fi
