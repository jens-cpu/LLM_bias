#!/bin/bash

# SLURM Job: 
srun --partition=kisski-h100 \
     --gres=gpu:H100:1 \
     --mem=256G \
     --cpus-per-task=6 \
     --time=04:00:00 \
     --pty bash -c "
     source spc_env/bin/activate &&
     python multi_model_bias_analysis.py"
