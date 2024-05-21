#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH --time 01:00:00
#SBATCH -p exercise-eml
#SBATCH -o exercise4_1_dropout.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml


echo 0.$1
python exercise4_1_1.py --dropout_p 0.$1 --epochs 30


echo "Running plot04"
python plot4_1_1.py

