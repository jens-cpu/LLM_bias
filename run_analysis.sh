#!/bin/bash

# SLURM Job: 1 GPU, 64 GB RAM, 4 CPUs, 4 Stunden Laufzeit
srun --partition=kisski \
     --gres=gpu:A100:1 \
     --mem=256G \
     --cpus-per-task=6 \
     --time=04:00:00 \
     --pty bash -c "
     source spc_env/bin/activate &&
     python spc_bias_analysis.py"
