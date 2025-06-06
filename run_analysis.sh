#!/bin/bash
#SBATCH --job-name=bias_analysis
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=kisski-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
export HF_TOKEN="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"
export PERSPECTIVE_API_KEY="AIzaSyCDtSOFBvNKeeh5I2rFkYOpBD0BpXuvgUA"
# Quota-freundlich
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
mkdir -p logs results plots /scratch/$USER/hf_cache /scratch/$USER/torch_cache
echo "HF_TOKEN is : $HF_TOKEN"
echo "PERSPECTIVE_API_KEY is : $PERSPECTIVE_API_KEY"
echo "Running on: $(hostname) at $(date)"
echo "Using Python: $(which python)"
source spc_env/bin/activate
python multi_model_bias_analysis.py
