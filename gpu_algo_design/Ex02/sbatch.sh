#!/bin/bash

#SBATCH --job-name=fdtd3d
#SBATCH --output=fdtd3d-%j.out
##SBATCH --partition=asccluster
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --ntasks=16
##SBATCH --time=01:00:00

# Loop through radius values from 1 to 10
for radius in {1..10}; do
    # Run with output caching set to true
    ./FDTD3d --radius=$radius --caching=output | grep "FDTD3d-radius*"
    
    # Run with output caching set to false
    ./FDTD3d --radius=$radius --caching=input | grep "FDTD3d-radius*"
done
