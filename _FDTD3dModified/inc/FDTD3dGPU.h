/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef _FDTD3DGPU_H_
#define _FDTD3DGPU_H_


#include <cstddef>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

#include "FDTD3dShared.h"


// TODO (DONE) : Rescale these values to CUDA 3.2 version of \
//                computing capabilities:                     
#define k_blockDimX    2
#define k_blockDimY    2
#define k_blockDimMaxY 16
#define k_blockSizeMin 4
#define k_blockSizeMax (k_blockDimX * k_blockDimMaxY)
#define k_gridDimX    2
#define k_gridDimY    2
#define k_gridDimMaxY 4
#define k_gridSizeMin 4
#define k_gridSizeMax (k_gridDimX * k_gridDimMaxY)


struct structReloadMemChunkArgs
{
  const int padding,
            idxGrid;
  const size_t volumeSize;
  const float *input;

  float *output,
        *bufferSrc,
        *bufferIn;
};

struct structKernelPThreadArgs
{
  const int dimx,
            dimy,
            dimz;
  const FieldComponents_t ***input;
  const xyz_t **TFSFsrcE;
  const xyz_t **TFSFsrcH;

  int dimGrid,
      dimBlock,
      maxSharedMemPerBlock;
  float *output;
};

typedef struct structKernelPThreadArgs KernelPThreadArgs_t;

bool getTargetDeviceGlobalMemSize(int *totalblockspermp, int *totalthreadspermp, int *totalmps, memsize_t *totalmem, const int argc, const char **argv);
bool fdtdGPU ( float *, const float *,             \
               const xyz_t ***,                    \
               const UpdateCoefficients_t &,       \
               const int , const int , const int , \
               const int , const int ,             \
               const int , const char **,          \
               int , int                           \
             );                                     
void * launchKernelPThreadAsync ( void * );


#endif
