#!/bin/bash
#SBATCH --gres gpu:0
#SBATCH --mem 5G
#SBATCH --cpus-per-task 1
#SBATCH --time 30:00
#SBATCH -p exercise-eml
#SBATCH -o slurm_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running exercise02"
python exercise2.py --lr 0.1
python exercise2.py --lr 0.01
python exercise2.py --lr 0.001
python exercise2.py --lr 0.5
python exercise2.py --lr 0.05

