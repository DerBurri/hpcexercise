#!/bin/bash
#
#SBATCH --job-name=histogram
#SBATCH --output=histogram-%j.out
#SBATCH --gres=gpu:rtx_4080:1

for warpCnt in 1 2 4 ; do
  for (( binNum = 256 ; binNum <= 8192 ; binNum *= 2 )) ; do
      ../../../bin/x86_64/linux/release/histogram --binNum=$binNum --Wc=$warpCnt | grep "histogram,"
  done
done
