#!/bin/bash
#SBATCH --job-name=bias_analysis
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=kisski
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
export PERSPECTIVE_API_KEY="AlzaSyCD 
echo "HF_TOKEN is : $HF_TOKEN"
source spc_env/bin/activate
python multi_model_bias_analysis.py
