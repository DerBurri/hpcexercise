#!/bin/bash
#SBATCH --job-name=fdtd3d
#SBATCH --output=threadFenceReduction-%j.out
##SBATCH --partition=asccluster
#SBATCH --gres=gpu:rtx_4080:1
#SBATCH --ntasks=16
##SBATCH --time=01:00:00

# echo "3.2"
echo "" > threadFenceReduction_multi.txt
for b in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 16384 32768 65536 131072 262144 524228 1045876 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456; do
    ./threadFenceReduction --n=1048576 --threads=128 --maxblocks=$b --multipass| grep "Bandwidth:" >> threadFenceReduction_multi.txt
done

echo "" > threadFenceReduction_single.txt
for b in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 16384 32768 65536 131072 262144 524228 1045876 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456; do
    ./threadFenceReduction --n=1048576 --threads=128 --maxblocks=$b | grep "Bandwidth:" >> threadFenceReduction_single.txt
done