#!/bin/bash
#
#SBATCH --job-name=threadFenceReduction
#SBATCH --output=threadFenceReduction-%j.out
#SBATCH --gres=gpu:rtx_4080:1

for (( elems = 1 ; elems <= 536870912 ; elems*=2 )); do
  ../../../bin/x86_64/linux/release/threadFenceReduction --n=$elems --threads=128 --maxblocks=524288 | grep "benchmarkReduce*"
  ../../../bin/x86_64/linux/release/threadFenceReduction --multipass --n=$elems --threads=128 --maxblocks=524288 | grep "benchmarkReduce*"
done
