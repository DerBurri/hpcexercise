#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 30:00
#SBATCH -p exercise-eml
#SBATCH -o ex3_1.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running on CPU"
python exercise3_1.py --no-cuda $true --dataset MNIST

echo "Running on GPU"
python exercise3_1.py --dataset MNIST