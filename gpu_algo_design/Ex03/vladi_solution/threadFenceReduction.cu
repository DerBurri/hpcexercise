/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  Parallel reduction

  This sample shows how to perform a reduction operation on an array of values
  to produce a single value in a single kernel (as opposed to two or more
  kernel calls as shown in the "reduction" CUDA Sample).  Single-pass
  reduction requires global atomic instructions (Compute Capability 1.1 or
  later) and the __threadfence() intrinsic (CUDA 2.2 or later).

  Reductions are a very common computation in parallel algorithms.  Any time
  an array of values needs to be reduced to a single value using a binary
  associative operator, a reduction can be used.  Example applications include
  statistics computations such as mean and standard deviation, and image
  processing applications such as finding the total luminance of an
  image.

  This code performs sum reductions, but any associative operator such as
  min() or max() could also be used.

  It assumes the input size is a power of 2.

  COMMAND LINE ARGUMENTS

  "--n=<N>":         Specify the number of elements to reduce (default 1048576)
  "--threads=<N>":   Specify the number of threads per block (default 128)
  "--maxblocks=<N>": Specify the maximum number of thread blocks to launch 
                     (kernel 6 only, default 64)
  "--cpufinal":      Read back the per-block results and do final sum of block 
                     sums on CPU (default false)
  "--cputhresh=<N>": The threshold of number of blocks sums below which to 
                     perform a CPU final reduction (default 1)
  "--multipass":     Use a multipass reduction instead of a single-pass reduction
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

#define VERSION_MAJOR (CUDART_VERSION / 1000)
#define VERSION_MINOR (CUDART_VERSION % 100) / 10

const char *sSDKsample = "threadFenceReduction";

#if CUDART_VERSION >= 2020
#include "threadFenceReduction_kernel.cuh"
#else
#pragma comment(user, "CUDA 2.2 is required to build for threadFenceReduction")
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, int dev_num);

extern "C" {
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata);
void reduceSinglePass(int size, int threads, int blocks, float *d_idata,
                      float *d_odata);
}

#if CUDART_VERSION < 2020
void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata) {
  printf("reduce(), compiler not supported, aborting tests\n");
}

void reduceSinglePass(int size, int threads, int blocks, float *d_idata,
                      float *d_odata) {
  printf("reduceSinglePass(), compiler not supported, aborting tests\n");
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;
  int dev;

  printf("%s Starting...\n\n", sSDKsample);

  dev = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  printf("GPU Device supports SM %d.%d compute capability\n\n",
         deviceProp.major, deviceProp.minor);

  bool bTestResult = false;

#if CUDART_VERSION >= 2020
  bTestResult = runTest(argc, argv, dev);
#else
  print_NVCC_min_spec(sSDKsample, "2.2", "Version 185");
  exit(EXIT_SUCCESS);
#endif

  exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template <class T>
T reduceCPU(T *data, int size) {
  T sum = data[0];
  T c = (T)0.0;

  for (int i = 1; i < size; i++) {
    T y = data[i] - c;
    T t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads) {
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    threads = (n < maxThreads * 2) ? nextPow2(n / 2) : maxThreads;
    blocks = max(1, n / (threads * 2));
  }

  blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
float benchmarkReduce(int n, int numThreads, int numBlocks, int maxThreads,
                      int maxBlocks, int testIterations, bool multiPass,
                      bool cpuFinalReduction, int cpuFinalThreshold,
                      StopWatchInterface *timer, float *h_odata, float *d_idata,
                      float *d_odata, int dev_num) {
  float gpu_result = 0;
  bool bNeedReadback = true;
  cudaError_t error;

  for (int i = 0; i < testIterations; ++i) {
    gpu_result = 0;
    unsigned int retCnt = 0;

    // Set the global accumulator back to 0
    //error = cudaMemcpyToSymbol(globalAcc, &gpu_result, sizeof(float), 0, cudaMemcpyHostToDevice); 
    //checkCudaErrors(error);

    error = setRetirementCount(retCnt);
    checkCudaErrors(error);

    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    if (multiPass) {
      // execute the kernel
      reduce(n, numThreads, numBlocks, d_idata, d_odata);

      // check if kernel execution generated an error
      getLastCudaError("Kernel execution failed");

      if (cpuFinalReduction) {
        // sum partial sums from each block on CPU
        // copy result from device to host
        error = cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(float),
                           cudaMemcpyDeviceToHost);
        checkCudaErrors(error);

        for (int i = 0; i < numBlocks; i++) {
          gpu_result += h_odata[i];
        }

        bNeedReadback = false;
      } else {
        // sum partial block sums on GPU
        int s = numBlocks;

        while (s > cpuFinalThreshold) {
          int threads = 0, blocks = 0;
          getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

          reduce(s, threads, blocks, d_odata, d_odata);

          s = s / (threads * 2);
        }

        if (s > 1) {
          // copy result from device to host
          error = cudaMemcpy(h_odata, d_odata, s * sizeof(float),
                             cudaMemcpyDeviceToHost);
          checkCudaErrors(error);

          for (int i = 0; i < s; i++) {
            gpu_result += h_odata[i];
          }

          bNeedReadback = false;
        }
      }
    } else {
      getLastCudaError("Kernel execution failed");

      // execute the kernel
      reduceSinglePass(n, numThreads, numBlocks, d_idata, d_odata);
      //reduceSpeciale<float>(n, numThreads, numBlocks, d_idata, d_odata, dev_num);

      // check if kernel execution generated an error
      getLastCudaError("Kernel execution failed");
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
  }

  if (bNeedReadback) {
    // copy final sum from device to host
    error = cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost);
    //error = cudaMemcpyFromSymbol(&gpu_result, globalAcc, sizeof(float), 0, cudaMemcpyDeviceToHost); 
    checkCudaErrors(error);
  }

  return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, int dev_num) {
  int size = 1 << 20;    // number of elements to reduce
  int maxThreads = 128;  // number of threads per block
  int maxBlocks = 64;
  bool cpuFinalReduction = false;
  int cpuFinalThreshold = 1;
  bool multipass = false;
  bool bTestResult = false;

  if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
    size = getCmdLineArgumentInt(argc, (const char **)argv, "n");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "threads")) {
    maxThreads = getCmdLineArgumentInt(argc, (const char **)argv, "threads");
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "blocks")) {
    maxBlocks = getCmdLineArgumentInt(argc, (const char **)argv, "blocks");
  }

  printf("%d elements\n", size);
  printf("%d threads (max)\n", maxThreads);

  cpuFinalReduction = checkCmdLineFlag(argc, (const char **)argv, "cpufinal");
  multipass = checkCmdLineFlag(argc, (const char **)argv, "multipass");

  if (checkCmdLineFlag(argc, (const char **)argv, "cputhresh")) {
    cpuFinalThreshold =
        getCmdLineArgumentInt(argc, (const char **)argv, "cputhresh");
  }

  // create random input data on CPU
  unsigned int bytes = size * sizeof(float);

  float *h_idata = (float *)malloc(bytes);

  for (int i = 0; i < size; i++) {
    // Keep the numbers small so we don't get truncation error in the sum
    h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
  }

  int numBlocks = 0;
  int numThreads = 0;
  getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

  if (numBlocks == 1) {
    cpuFinalThreshold = 1;
  }

  // allocate mem for the result on host side
  float *h_odata = (float *)malloc(numBlocks * sizeof(float));

  printf("%d blocks\n", numBlocks);

  // allocate device memory and data
  float *d_idata = NULL;
  float *d_odata = NULL;

  checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
  checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks * sizeof(float)));

  // copy data directly to device memory
  checkCudaErrors(
      cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(float),
                             cudaMemcpyHostToDevice));

  // warm-up
  reduce(size, numThreads, numBlocks, d_idata, d_odata);
  const int testIterations = 100;

  StopWatchInterface *timer = 0;
  sdkCreateTimer(&timer);

  float gpu_result = 0;

  gpu_result =
      benchmarkReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                      testIterations, multipass, cpuFinalReduction,
                      cpuFinalThreshold, timer, h_odata, d_idata, d_odata, dev_num);

  float reduceTime = sdkGetAverageTimerValue(&timer);
  printf("benchmarkReduce-%s-%delems-%db-%dt, ", multipass ? "multipass" : "singlepass", size, numBlocks, numThreads);
  printf("Average time= %f ms, ", reduceTime);
  printf("Bandwidth= %f GB/s\n",
         (size * sizeof(int)) / (reduceTime * 1.0e6));

  // compute reference solution
  float cpu_result = reduceCPU<float>(h_idata, size);

  printf("GPU result = %0.12f, ", gpu_result);
  printf("CPU result = %0.12f,", cpu_result);

  double threshold = 1e-8 * size;
  double diff = abs((double)gpu_result - (double)cpu_result);
  bTestResult = (diff < threshold);
  printf("Test successfull: %d\n", bTestResult ? 0 : 1);

  // cleanup
  sdkDeleteTimer(&timer);

  free(h_idata);
  free(h_odata);
  cudaFree(d_idata);

  return bTestResult;
}
