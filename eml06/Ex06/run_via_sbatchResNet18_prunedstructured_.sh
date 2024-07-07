#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH --time 01:00:00
#SBATCH -p exercise-eml
#SBATCH -o slurm-pruned_structured_output.log

# load appropriate conda paths, because we are not in a login shell
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running exercise05_template.py"
pip install torchinfo
python "exercise5_2resnet18_structured_pruned.py" --epochs 50 --pruning-rate 0.2
python "exercise5_2resnet18_structured_pruned.py" --epochs 50 --pruning-rate 0.3
python "exercise5_2resnet18_structured_pruned.py" --epochs 50 --pruning-rate 0.4
python "exercise5_2resnet18_structured_pruned.py" --epochs 50 --pruning-rate 0.5
python "exercise5_2resnet18_structured_pruned.py" --epochs 50 --pruning-rate 0.6
