#!/bin/bash

make clean
make shfl_scanneo

srun --pty --gres=gpu:rtx_4080:1 compute-sanitizer --log-file report.txt ./shfl_scanneo
