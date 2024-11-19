#!/bin/bash
#SBATCH --job-name=threadFenceReduction
#SBATCH --output=threadFenceReduction-%j.out
##SBATCH --partition=asccluster
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --ntasks=16
##SBATCH --time=01:00:00

echo "3.2"
echo "" > out/threadFenceReduction.csv
./threadFenceReduction/threadFenceReduction --shmoo --maxblocks=524288 >> out/threadFenceReduction.csv
