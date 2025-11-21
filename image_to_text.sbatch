#!/bin/bash
#SBATCH --job-name=mllm_house_search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --output=mllm_job_%j.out
#SBATCH --error=mllm_job_%j.err

# how to run
# Defaults
#
# sbatch run_job.sbatch
#
# Overrides
# export model_path="/custom/model/path"
# export json_input_path="/custom/data.json"
# export output_path="/custom/output"
# sbatch run_job.sbatch

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================"

# Create a unique directory for this job
JOB_DIR="/scratch/bohora.a/mllm_job_${SLURM_JOB_ID}"
echo "Creating job directory: $JOB_DIR"
mkdir -p $JOB_DIR

# Navigate to the job directory
cd $JOB_DIR
echo "Working directory: $(pwd)"

# Load necessary modules (adjust based on your cluster)
module load git

# Use existing code from /projects/scdatahub
echo "================================"
echo "Using code from /projects/scdatahub/amol_dmt/image_to_text_runs/MLLM-House-Search"
CODE_PATH="/projects/scdatahub/amol_dmt/image_to_text_runs/MLLM-House-Search"

# Check if code exists
if [ -d "$CODE_PATH" ]; then
    echo "Code directory found: $CODE_PATH"
    ls -la "$CODE_PATH"
else
    echo "Error: Code directory not found at $CODE_PATH"
    exit 1
fi


echo "Setting environment variables for main.py..."

# Set environment variables for main.py
# model_path: Use model directly from projects directory
export model_path="/projects/scdatahub/amol_dmt/model/8b"

# json_input_path: Input JSON file with house image associations
# Use the correct JSON file name
export json_input_path="${json_input_path:-/projects/scdatahub/amol_dmt/image_to_text_runs/MLLM-House-Search/house_image_associations.json}"

# output_path: Output parquet file path
# Save to the output directory in /projects/scdatahub
export output_path="${output_path:-/projects/scdatahub/amol_dmt/image_to_text_runs/output/house_descriptions.parquet}"

# Create output directory
OUTPUT_DIR="/projects/scdatahub/amol_dmt/image_to_text_runs/output"
mkdir -p "$OUTPUT_DIR"

echo "Environment variables set:"
echo "  model_path: $model_path"
echo "  json_input_path: $json_input_path"
echo "  output_path: $output_path"

# Set PyTorch optimizations for transformers
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
echo "PyTorch optimizations configured"

echo "================================"
echo "Setting Python unbuffered mode for real-time logging..."
export PYTHONUNBUFFERED=1

echo "================================"
echo "Activating conda environment..."

# Source conda and activate environment
source /shared/centos7/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate qwen-env

# Check if conda environment activated successfully
if [ $? -eq 0 ]; then
    echo "Conda environment 'qwen-env' activated successfully"
else
    echo "Error: Failed to activate conda environment"
    exit 1
fi

echo "================================"
echo "Running main.py..."
echo "Input: $json_input_path"
echo "Output: $output_path"
echo "Model: $model_path"
echo "================================"

# Navigate to the image_to_text_pipeline directory
cd "$CODE_PATH/image_to_text_pipeline"

# Run the main script with proper arguments (using python -u for unbuffered output)
python -u main.py \
    --input "$json_input_path" \
    --output "$output_path" \
    --model-path "$model_path" \
    --resume

# Check if script ran successfully
if [ $? -eq 0 ]; then
    echo "main.py completed successfully"
else
    echo "Error: main.py failed with exit code $?"
    exit 1
fi

echo "================================"
echo "End Time: $(date)"
echo "Job completed successfully"
