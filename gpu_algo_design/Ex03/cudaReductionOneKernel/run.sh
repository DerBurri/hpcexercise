#!/bin/bash
#SBATCH --job-name=reduction
#SBATCH --output=reduction-%j.out
##SBATCH --partition=asccluster
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --ntasks=16
##SBATCH --time=01:00:00

echo "3.3"
echo "" > ../out/reduction.csv
./reduction --shmoo --n=536870912 --type=int |grep ,>> ../out/reduction.csv
./reduction --shmoo --n=536870912 --type=float |grep ,>> ../out/reduction.csv
./reduction --shmoo --n=268435456 --type=double |grep ,>> ../out/reduction.csv

