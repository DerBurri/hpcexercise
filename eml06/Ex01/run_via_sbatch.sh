#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --time 30:00
#SBATCH -p exercise
#SBATCH -o my-job-output

echo "Loading conda env"
module load anaconda/3
source ~/.bashrc
conda activate EML_ex01

echo "Running PyTorch test"
python -c "import torch; print(torch.rand(5).to('cuda')); \
print(torch.cuda.is_available())"

echo "Running template.py"
python template.py
