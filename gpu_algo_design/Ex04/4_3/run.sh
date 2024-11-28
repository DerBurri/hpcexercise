#!/bin/bash

make clean
#make shfl_scan
make shfl_scanneo

#srun --output=shfl_orig.txt --gres=gpu:rtx_4080:1  ./shfl_scan
srun --output=shfl_gpu.txt --gres=gpu:rtx_4080:1  ./shfl_scanneo