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

# Clone the repository with recursive submodules
echo "================================"
echo "Cloning repository..."
git clone --recursive https://github.com/MrMalfunction/MLLM-House-Search.git

# Check if clone was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully"
    echo "Repository location: $JOB_DIR/MLLM-House-Search"
    ls -la MLLM-House-Search/
else
    echo "Error: Failed to clone repository"
    exit 1
fi

echo "================================"
echo "Copying model to local storage to avoid network latency..."

# Source model location (network storage)
SOURCE_MODEL_PATH="${model_path:-/projects/scdatahub/amol_dmt/model/8b}"

# Local model path (on scratch - much faster)
LOCAL_MODEL_PATH="$JOB_DIR/model_cache"

# Check if source model exists and copy it
if [ -d "$SOURCE_MODEL_PATH" ] && [ "$(ls -A $SOURCE_MODEL_PATH)" ]; then
    echo "Found model at: $SOURCE_MODEL_PATH"
    echo "Copying to local storage: $LOCAL_MODEL_PATH"
    echo "This may take a few minutes..."

    mkdir -p "$LOCAL_MODEL_PATH"

    # Use rsync for faster copying (if available), otherwise use cp
    if command -v rsync &> /dev/null; then
        echo "Using rsync for faster transfer..."
        rsync -avh --progress "$SOURCE_MODEL_PATH/" "$LOCAL_MODEL_PATH/"
    else
        echo "Using cp to copy files..."
        cp -rv "$SOURCE_MODEL_PATH"/* "$LOCAL_MODEL_PATH/"
    fi

    if [ $? -eq 0 ]; then
        echo "Model copied successfully to local storage"
        echo "Model size in local storage:"
        du -sh "$LOCAL_MODEL_PATH"
    else
        echo "Warning: Model copy failed, will download to local storage instead"
        rm -rf "$LOCAL_MODEL_PATH"/*
    fi
else
    echo "Source model not found at $SOURCE_MODEL_PATH"
    echo "Model will be downloaded to local storage instead"
    mkdir -p "$LOCAL_MODEL_PATH"
fi

echo "================================"
echo "Setting environment variables for main.py..."

# Set environment variables for main.py
# model_path: Use local copy of the model for faster loading
export model_path="$LOCAL_MODEL_PATH"

# json_input_path: Input JSON file with house image associations
# Default looks for it in the cloned repository's root directory
export json_input_path="${json_input_path:-$JOB_DIR/MLLM-House-Search/house_image_associations.json}"

# output_path: Directory where output parquet files will be saved
# Default: save to the job directory for easy access
export output_path="${output_path:-$JOB_DIR/output}"

# Create output directory
mkdir -p "$output_path"

echo "Environment variables set:"
echo "  model_path: $model_path (LOCAL SCRATCH STORAGE)"
echo "  json_input_path: $json_input_path"
echo "  output_path: $output_path"

# Set PyTorch optimizations for transformers
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
echo "PyTorch optimizations configured"

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

# Navigate to the image_to_text_pipeline directory
cd MLLM-House-Search/image_to_text_pipeline

# Run the main script
python main.py

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
