#!/bin/bash

#SBATCH --job-name=burn
#SBATCH --output=logs/%N-%j-%x
#SBATCH --error=logs/%N-%j-%x

# get all resources
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --qos=burnin

# plenty of time
#SBATCH --time=02:00:00

module reset
module load Miniforge3
module list

env | grep -i "conda"
env | grep -i "^path="

## https://github.com/wwarriner/burn-in

which python
conda activate burnin
echo "$?"

env | grep -i "conda"
env | grep -i "^path="

which python
sleep 5
which python
mkdir -p out
python burn.py --output-file out/$(hostname)-$SLURM_JOB_ID-results.csv
conda deactivate

## PHORONIX | AI-BENCHMARK
## https://gitlab.rc.uab.edu/rc/benchmark

# conda activate phoronix
# module load cuDNN/8.9.2.26-CUDA-11.6.0
# module load rc/phoronix/10.8
# phoronix-test-suite batch-run ai-benchmark
# conda deactivate
