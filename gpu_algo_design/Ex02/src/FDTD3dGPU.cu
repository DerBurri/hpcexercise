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

#include <helper_cuda.h>
#include <helper_functions.h>

#include <algorithm>
#include <iostream>

#include "FDTD3dGPU.h"
#include "FDTD3dGPUKernel.cuh"

#ifndef I_BENCHMARK
#define I_BENCHMARK 100
#endif

#define GPU_PROFILING

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc,
                                  const char **argv) {
  int deviceCount = 0;
  int targetDevice = 0;
  size_t memsize = 0;

  // Get the number of CUDA enabled GPU devices
  printf(" cudaGetDeviceCount\n");
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  // Query target device for maximum memory allocation
  printf(" cudaGetDeviceProperties\n");
  struct cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

  memsize = deviceProp.totalGlobalMem;

  // Save the result
  *result = (memsize_t)memsize;
  return true;
}

bool fdtdGPU(float *output, const float *input, const float *coeff,
             const int dimx, const int dimy, const int dimz, const int radius,
             const int timesteps, const int argc, const char **argv,
             bool outputCaching) {
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = outerDimx * outerDimy * outerDimz;
  int deviceCount = 0;
  int targetDevice = 0;
  float *bufferOut = 0;
  float *bufferIn = 0;
  dim3 dimBlock;
  dim3 dimGrid;

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
  cudaEvent_t profileStart = 0;
  cudaEvent_t profileEnd = 0;
  const int profileTimesteps = timesteps - 1;

  if (profileTimesteps < 1) {
    printf(
        " cannot profile with fewer than two timesteps (timesteps=%d), "
        "profiling is disabled.\n",
        timesteps);
  }

#endif

  // Get the number of CUDA enabled GPU devices
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaSetDevice(targetDevice));

  // Allocate memory buffers
  checkCudaErrors(
      cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

  // Check for a command-line specified block size
  int userBlockSize;

  if (checkCmdLineFlag(argc, (const char **)argv, "block-size")) {
    userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
    // Constrain to a multiple of k_blockDimX
    userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

    // Constrain within allowed bounds
    userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
  } else {
    userBlockSize = k_blockSizeMax;
  }

  // Check the device limit on the number of threads
  struct cudaFuncAttributes funcAttrib;

#define CHECK_KERNEL_INPUT(n)                                            \
  case n:                                                                \
    checkCudaErrors(                                                     \
        cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<n>)); \
    break;
#define CHECK_KERNEL_OUTPUT(n)                                                \
  case n:                                                                     \
    checkCudaErrors(                                                          \
        cudaFuncGetAttributes(&funcAttrib, FiniteDifferences3DBoxKernel<n>)); \
    break;

  if (outputCaching) {
    switch (radius) {
      CHECK_KERNEL_OUTPUT(1)
      CHECK_KERNEL_OUTPUT(2)
      CHECK_KERNEL_OUTPUT(3)
      CHECK_KERNEL_OUTPUT(4)
      CHECK_KERNEL_OUTPUT(5)
      CHECK_KERNEL_OUTPUT(6)
      CHECK_KERNEL_OUTPUT(7)
      CHECK_KERNEL_OUTPUT(8)
      CHECK_KERNEL_OUTPUT(9)
      CHECK_KERNEL_OUTPUT(10)
      default:
        std::cerr << "Radius must be between 1 and 10." << std::endl;
        exit(EXIT_FAILURE);
    }
  } else {
    switch (radius) {
      CHECK_KERNEL_INPUT(1)
      CHECK_KERNEL_INPUT(2)
      CHECK_KERNEL_INPUT(3)
      CHECK_KERNEL_INPUT(4)
      CHECK_KERNEL_INPUT(5)
      CHECK_KERNEL_INPUT(6)
      CHECK_KERNEL_INPUT(7)
      CHECK_KERNEL_INPUT(8)
      CHECK_KERNEL_INPUT(9)
      CHECK_KERNEL_INPUT(10)
      default:
        std::cerr << "Radius must be between 1 and 10." << std::endl;
        exit(EXIT_FAILURE);
    }
  }

  userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

  // Set the block size
  dimBlock.x = k_blockDimX;
  // Visual Studio 2005 does not like std::min
  //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX,
  //    (size_t)k_blockDimMaxY);
  dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY)
                   ? (userBlockSize / k_blockDimX)
                   : (size_t)k_blockDimMaxY;
  dimGrid.x = (unsigned int)ceil((float)dimx / dimBlock.x);
  dimGrid.y = (unsigned int)ceil((float)dimy / dimBlock.y);
  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

  // Check the block size is valid
  if (dimBlock.x < radius || dimBlock.y < radius) {
    printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n",
           dimBlock.x, dimBlock.y, radius);
    exit(EXIT_FAILURE);
  }

  // Copy the coefficients to the device coefficient buffer
  checkCudaErrors(
      cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));

#ifdef GPU_PROFILING
  // Create the events
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));
#endif

  double throughputSum = 0;
  double avgElapsedTimeSum = 0;
  size_t pointsComputed = dimx * dimy * dimz;
  for (int meas_it = 0; meas_it < 20; meas_it++) {
      // Warmup kernel
    emptyKernel<<<1,1>>>();
  }

  //We copy every time to the device to get best accuracy for real usage (because then also the coefficients are copied to the device)
  // Can be changed if data resides on the device for further computations!!
  for (int meas_it = 0; meas_it < I_BENCHMARK; meas_it++) {
    // Copy the input to the device input buffer
    checkCudaErrors(cudaMemcpy(bufferIn + padding, input,
                               volumeSize * sizeof(float),
                               cudaMemcpyHostToDevice));

  // Copy the input to the device output buffer (actually only need the halo)
  checkCudaErrors(cudaMemcpy(bufferOut + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));
  //check which stencil has been used
  if (outputCaching)
  { 
    printf("output caching\n");
    checkCudaErrors(
      cudaMemcpyToSymbol(stencil3d, (void *)coeff, (radius*2+1) * (radius*2+1) * (radius*2+1) * sizeof(float)));
      }
  else {
  // Copy the coefficients to the device coefficient buffer 
  checkCudaErrors(
      cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));
  }
#ifdef GPU_PROFILING


  // Create the events
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

  // Execute the FDTD
  float *bufferSrc = bufferIn + padding;
  float *bufferDst = bufferOut + padding;
  printf(" GPU FDTD loop\n");
    // Copy the input to the device output buffer (actually only need the halo)
    checkCudaErrors(cudaMemcpy(bufferOut + padding, input,
                               volumeSize * sizeof(float),
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());

#ifdef GPU_PROFILING
    checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

    for (int it = 0; it < timesteps; it++) {

#define CALL_KERNEL_INPUT(n)                                               \
    case n:                                                                \
      FiniteDifferencesKernel<n>                                           \
          <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz); \
      break;
#define CALL_KERNEL_OUTPUT(n)                                              \
    case n:                                                                \
      FiniteDifferences3DBoxKernel<n>                                      \
          <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz); \
      break;

      if (outputCaching) {
        switch (radius) {
          CALL_KERNEL_OUTPUT(1)
          CALL_KERNEL_OUTPUT(2)
          CALL_KERNEL_OUTPUT(3)
          CALL_KERNEL_OUTPUT(4)
          CALL_KERNEL_OUTPUT(5)
          CALL_KERNEL_OUTPUT(6)
          CALL_KERNEL_OUTPUT(7)
          CALL_KERNEL_OUTPUT(8)
          CALL_KERNEL_OUTPUT(9)
          CALL_KERNEL_OUTPUT(10)
          default:
            std::cerr << "Radius must be between 1 and 10." << std::endl;
            exit(EXIT_FAILURE);
        }
      } else {
        switch (radius) {
          CALL_KERNEL_INPUT(1)
          CALL_KERNEL_INPUT(2)
          CALL_KERNEL_INPUT(3)
          CALL_KERNEL_INPUT(4)
          CALL_KERNEL_INPUT(5)
          CALL_KERNEL_INPUT(6)
          CALL_KERNEL_INPUT(7)
          CALL_KERNEL_INPUT(8)
          CALL_KERNEL_INPUT(9)
          CALL_KERNEL_INPUT(10)
          default:
            std::cerr << "Radius must be between 1 and 10." << std::endl;
            exit(EXIT_FAILURE);
        }
      }
      // Toggle the buffers
      // Visual Studio 2005 does not like std::swap
      //    std::swap<float *>(bufferSrc, bufferDst);
      float *tmp = bufferDst;
      bufferDst = bufferSrc;
      bufferSrc = tmp;
    }

#ifdef GPU_PROFILING
  // Enqueue end event
  // It was like that in the standard implementation DO NOT CHANGE
  checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

    // Read the result back, result is in bufferSrc (after final toggle)
    checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float),
                               cudaMemcpyDeviceToHost));

    float elapsedTimeMS = 0;

    if (profileTimesteps > 0) {
      checkCudaErrors(
          cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
    }

    if (profileTimesteps > 0) {
      // Convert milliseconds to seconds
      double elapsedTime = elapsedTimeMS * 1.0e-3;
      double avgElapsedTime = elapsedTime / (double)profileTimesteps;
      // Determine throughput
      double throughputM = 1.0e-6 * (double)pointsComputed / avgElapsedTime;

      avgElapsedTimeSum += avgElapsedTime;
      throughputSum += throughputM;
    }
  }

  printf("FDTD3d-radius%d, caching %s: Avg. throughput = %.4f MPoints/s, Avg. time "
      "= %.5f s, Size = %u Points, Blocksize = %u\n",
       radius, outputCaching ? "output" : "input", throughputSum / I_BENCHMARK, avgElapsedTimeSum / I_BENCHMARK, 
       pointsComputed, dimBlock.x * dimBlock.y);

  // Cleanup
  if (bufferIn) {
    checkCudaErrors(cudaFree(bufferIn));
  }

  if (bufferOut) {
    checkCudaErrors(cudaFree(bufferOut));
  }

#ifdef GPU_PROFILING

  if (profileStart) {
    checkCudaErrors(cudaEventDestroy(profileStart));
  }

  if (profileEnd) {
    checkCudaErrors(cudaEventDestroy(profileEnd));
  }

#endif
  return true;
}
