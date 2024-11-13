#!/bin/bash

#SBATCH --job-name=fdtd3d
#SBATCH --output=./out/test_fdtd3d_%j.log
##SBATCH --partition=asccluster
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --ntasks=16
##SBATCH --time=01:00:00

./bin/FDTD3d --radius=1
./bin/FDTD3d --radius=2
./bin/FDTD3d --radius=3
./bin/FDTD3d --radius=4
./bin/FDTD3d --radius=5
./bin/FDTD3d --radius=6
./bin/FDTD3d --radius=7
./bin/FDTD3d --radius=8
./bin/FDTD3d --radius=9
./bin/FDTD3d --radius=10
