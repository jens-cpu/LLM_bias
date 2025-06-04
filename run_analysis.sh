#!/bin/bash
#SBATCH --job-name=bias_analysis
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00

source spc_env/bin/activate
python multi_model_bias_analysis.py
