#!/bin/bash

mkdir -p logs

# PASCALNODES
for node in c00{97..99} c0{100..111} c0{113..114}; do
  sbatch \
    --nodelist="${node}" \
    --partition=pascalnodes \
    --gres=gpu:4 \
    --reservation=root_400 \
    job.sh
done

# INTEL-DCB
for node in c0{115..133}; do
  sbatch \
    --nodelist="${node}" \
    --partition=intel-dcb \
    --reservation=root_400 \
    job.sh
done

# AMPERENODES
for node in c0{236..255}; do
  sbatch \
    --nodelist="${node}" \
    --partition=amperenodes \
    --gres=gpu:2 \
    --reservation=root_400 \
    job.sh
done

# for node in c0098; do
#   sbatch \
#     --nodelist="${node}" \
#     --partition=pascalnodes \
#     --gres=gpu:4 \
#     --reservation=root_400 \
#     job.sh
# done
