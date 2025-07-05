#!/bin/bash
#SBATCH --job-name=bias_analysis
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=kisski-h100
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00

export HF_TOKEN="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"
export PERSPECTIVE_API_KEY="AIzaSyCDtSOFBvNKeeh5I2rFkYOpBD0BpXuvgUA"
export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE=$HOME/hf_cache
export TORCH_HOME=$HOME/torch_cache
export TORCH_LOAD_FAIL_ON_UNINITIALIZED_PARAMETER=1

mkdir -p logs results plots "$HF_HOME" "$TORCH_HOME"

echo "HF_TOKEN is : $HF_TOKEN"
echo "PERSPECTIVE_API_KEY is : $PERSPECTIVE_API_KEY"
echo "Running on: $(hostname) at $(date)"
echo "Using Python: $(which python)"
echo "Memory info:"
free -h

source spc_env/bin/activate
accelerate launch multi_model_bias_analysis.py
