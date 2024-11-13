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

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc,
                                  const char **argv) {
  int deviceCount = 0;
  int targetDevice = 0;
  size_t memsize = 0;

  // Get the number of CUDA enabled GPU devices
#ifndef SUPRESS_OUTPUT
  printf(" cudaGetDeviceCount\n");
#endif
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  // Query target device for maximum memory allocation
#ifndef SUPRESS_OUTPUT
  printf(" cudaGetDeviceProperties\n");
#endif
  struct cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

  memsize = deviceProp.totalGlobalMem;

  // Save the result
  *result = (memsize_t)memsize;
  return true;
}

bool fdtdGPU(float *output, const float *input, const float *coeff,
             const int dimx, const int dimy, const int dimz, const int radius,
             const int timesteps, const int kDim, const int argc,
             const char **argv) {
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

#ifdef DEBUG
  printf("DEBUG - Using radius %d\n", radius);
  fflush(stdout);
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

#ifdef DEBUG
  printf("DEBUG - Allocated bufferIn and BufferOut on device\n");
  fflush(stdout);
#endif

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
  if (kDim == 2) {
    switch (radius) {
      case 1:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<1>));
        break;
      case 2:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<2>));
        break;
      case 3:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<3>));
        break;
      case 4:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<4>));
        break;
      case 5:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<5>));
        break;
      case 6:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<6>));
        break;
      case 7:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<7>));
        break;
      case 8:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<8>));
        break;
      case 9:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<9>));
        break;
      case 10:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<10>));
        break;
      default:
        fprintf(stderr, "Error: Unsupported radius %d\n", radius);
        exit(EXIT_FAILURE);
    }
  } else if (kDim == 3) {
    switch (radius) {
      case 1:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FullStencilKernel<1>));
        break;
      case 2:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FullStencilKernel<2>));
        break;
      case 3:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FullStencilKernel<3>));
        break;
      case 4:
        checkCudaErrors(
            cudaFuncGetAttributes(&funcAttrib, FullStencilKernel<4>));
        break;
      default:
        fprintf(stderr, "Error: Unsupported radius %d\n", radius);
        exit(EXIT_FAILURE);
    }
  } else {
    fprintf(stderr, "Error: Unsupported dimensionality %d\n", kDim);
    exit(EXIT_FAILURE);
  }

#ifdef DEBUG
  printf("DEBUG - Checked function attributes\n");
  fflush(stdout);
#endif

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

  if (kDim == 2) {
#ifndef SUPRESS_OUTPUT
    printf(" set block size to %dx%dx%d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf(" set grid size to %dx%dx%d\n", dimGrid.x, dimGrid.y, dimGrid.z);
#endif

    // Check the block size is valid
    if (dimBlock.x < radius || dimBlock.y < radius) {
      printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n",
             dimBlock.x, dimBlock.y, radius);
      exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
      printf("DEBUG - block size is valid\n");
      fflush(stdout);
#endif
    }
  } else if (kDim == 3) {
    dimBlock.y = radius;
    dimBlock.z = radius;
    dimBlock.x =
        MIN(radius, funcAttrib.maxThreadsPerBlock / (dimBlock.y * dimBlock.z));

    if (dimBlock.x * dimBlock.y * dimBlock.z > funcAttrib.maxThreadsPerBlock) {
      fprintf(stderr,
              "Error: The number of threads in a block exceeds the maximum "
              "number of threads per block\n");
      exit(EXIT_FAILURE);
    }
#ifndef SUPRESS_OUTPUT
    printf(" set block size to %dx%dx%d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf(" set grid size to %dx%dx%d\n", dimGrid.x, dimGrid.y, dimGrid.z);
#endif
#ifdef DEBUG
    printf("DEBUG - block size is valid\n");
    fflush(stdout);
#endif
  }

  // Copy the input to the device input buffer
  checkCudaErrors(cudaMemcpy(bufferIn + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));
  // Copy the input to the device output buffer (actually only need the halo)
  checkCudaErrors(cudaMemcpy(bufferOut + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));
#ifdef DEBUG
  printf("DEBUG - copied data to bufferIn and bufferOut\n");
  fflush(stdout);
#endif
  // Copy the coefficients
  if (kDim == 2) {
    if (radius > MAX_RADIUS_2D) {
      fprintf(stderr,
              "Error: Unsupported radius %d, maximum supported radius is %d\n",
              radius, MAX_RADIUS_2D);
      exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff,
                                       (radius + 1) * sizeof(float)));
  } else if (kDim == 3) {
    if (radius > MAX_RADIUS_3D) {
      fprintf(stderr,
              "Error: Unsupported radius %d, maximum supported radius is %d\n",
              radius, MAX_RADIUS_3D);
      exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaMemcpyToSymbol(
        stencil3D, (void *)coeff,
        ((2 * radius + 1) * (2 * radius + 1) * (2 * radius + 1)) *
            sizeof(float)));
  } else {
    fprintf(stderr, "Error: Unsupported dimensionality %d\n", kDim);
    exit(EXIT_FAILURE);
  }
#ifdef DEBUG
  printf("DEBUG - copied coefficients to device\n");
  fflush(stdout);
#endif

#ifdef GPU_PROFILING
  // Create the events
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));
#endif

  // Execute the FDTD
  float *bufferSrc = bufferIn + padding;
  float *bufferDst = bufferOut + padding;
#ifndef SUPRESS_OUTPUT
  printf(" GPU FDTD loop\n");
#endif

#ifdef GPU_PROFILING
  // Enqueue start event
  checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

// Launch the kernel
#ifdef DEBUG
  printf("DEBUG - Launching kernel with radius %d\n", radius);
  fflush(stdout);
#endif

  for (int it = 0; it < timesteps; it++) {
#ifndef SUPRESS_OUTPUT
    printf("\tt = %d\n", it);
#endif
    // Launch the appropriate kernel based on the radius and dimensionality
    if (kDim == 2) {
#ifdef DEBUG
      printf("DEBUG - Launching 2D kernel\n");
      fflush(stdout);
#endif
      switch (radius) {
        case 1:
          FiniteDifferencesKernel<1>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 2:
          FiniteDifferencesKernel<2>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 3:
          FiniteDifferencesKernel<3>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 4:
          FiniteDifferencesKernel<4>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 5:
          FiniteDifferencesKernel<5>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 6:
          FiniteDifferencesKernel<6>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 7:
          FiniteDifferencesKernel<7>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 8:
          FiniteDifferencesKernel<8>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 9:
          FiniteDifferencesKernel<9>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 10:
          FiniteDifferencesKernel<10>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        default:
          fprintf(stderr, "Error: Unsupported radius %d\n", radius);
          exit(EXIT_FAILURE);
      }
      // launch 3D kernel
    } else if (kDim == 3) {
#ifdef DEBUG
      printf(
          "DEBUG - Launching 3D kernel with %d threads per block and %d "
          "blocks "
          "per grid\n",
          dimBlock.x * dimBlock.y * dimBlock.z,
          dimGrid.x * dimGrid.y * dimGrid.z);
      fflush(stdout);
#endif
      switch (radius) {
        case 1:
          FullStencilKernel<1>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 2:
          FullStencilKernel<2>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 3:
          FullStencilKernel<3>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 4:
          FullStencilKernel<4>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        default:
          fprintf(stderr, "Error: Unsupported radius %d\n", radius);
          exit(EXIT_FAILURE);
      }
    } else {
      fprintf(stderr, "Error: Unsupported dimensionality %d\n", kDim);
      exit(EXIT_FAILURE);
    }

    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());

    // Toggle the buffers
    // Visual Studio 2005 does not like std::swap
    //    std::swap<float *>(bufferSrc, bufferDst);
    float *tmp = bufferDst;
    bufferDst = bufferSrc;
    bufferSrc = tmp;
  }

#ifdef GPU_PROFILING
  // Enqueue end event
  checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());

  // Read the result back, result is in bufferSrc (after final toggle)
  checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float),
                             cudaMemcpyDeviceToHost));

// Report time
#ifdef GPU_PROFILING
  float elapsedTimeMS = 0;

  if (profileTimesteps > 0) {
    checkCudaErrors(
        cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
  }

  if (profileTimesteps > 0) {
    // Convert milliseconds to seconds
    double elapsedTime = elapsedTimeMS * 1.0e-3;
    double avgElapsedTime = elapsedTime / (double)profileTimesteps;
    // Determine number of computations per timestep
    size_t pointsComputed = dimx * dimy * dimz;
    // Determine throughput
    double throughputM = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
    printf(
        "FDTD3d, radius %d: Throughput = %.4f MPoints/s, Time = %.5f s, Size "
        "= "
        "%u Points, "
        "NumDevsUsed = %u, Blocksize = %u\n",
        radius, throughputM, avgElapsedTime, pointsComputed, 1,
        dimBlock.x * dimBlock.y);
  }
#endif

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