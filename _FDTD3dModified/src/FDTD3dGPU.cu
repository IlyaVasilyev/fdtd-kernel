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

#include "FDTD3dShared.h"
#include "FDTD3dGPU.h"

#include <iostream>
#include <algorithm>
#include <pthread.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// TODO : Figure out what is `padding'.

void *launchKernelPThreadAsync ( void *arguments )
{
  KernelPThreadArgs_t * &args = arguments;

  FiniteDifferencesKernel<<<args.dimGrid, args.dimBlock, args.maxSharedMemPerBlock>>>(args.bufferDst, args.bufferSrc, args.dimx, args.dimy, args.dimz);

  return NULL;
}

void *reloadGlobal2SharedMemChunk ( void *arguments )
{
  ReloadMemChunkArgs_t * &args = arguments;

  while ( bReadyForNewSharedMemoryChunk[args.idxGrid] ) {}
  checkCudaErrors(cudaMemcpyAsync(args.output, &((GridMap_t *)args.bufferSrc)[args.idxGrid], args.volumeSize * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpyAsync(&((GridMap_t *)(args.bufferIn + args.padding))[args.idxGrid], args.input, args.volumeSize * sizeof(float), cudaMemcpyHostToDevice));

  return NULL;
}

// TODO : Move these defs below somewhere else PLEASE !!!
#define float3darray_t(TYPEDEFNAME, DIMI, DIMJ, DIMK) float TYPEDEFNAME[DIMI][DIMJ][DIMK]
#define float2darray_t(TYPEDEFNAME, DIMJ, DIMK) float TYPEDEFNAME[DIMJ][DIMK]
typedef float3darray_t(f3da_t,RADIUS_SHARED,VOLUME_SHARED,VOLUME_SHARED);
typedef float2darray_t(f2da_t,VOLUME_SHARED,VOLUME_SHARED);

__host__ __device__ inline float getFDTDTFSFsrcNull ( int enumXY, int ix, int iy, int iz )
{
  return 0;
}

__host__ __device__ inline float getFDTDTFSFsrcE ( int enumXY, int ix, int iy, int iz, int it )
{
  return sinsrc[enumXY][(int)(omega_src * (it % T_src) - k_src * n_inc * iz)];
}

__host__ __device__ inline float getFDTDTFSFsrcH ( int enumXY, int ix, int iy, int iz )
{
  return epsilonsqrt_src * sinsrc[enumXY][(int)(omega_src * (it % T_src) - k_src * n_inc * iz)] / musqrt_src;
}

__host__ __device__ inline void fdtdRefCurlXE ( float &C, xyz_t F, int ix, int iy, int iz )
{
  C = ( F.Z[ix][iy + 1][iz] - F.Z[ix][iy][iz] ) / dy - ( F.Y[ix][iy][iz + 1] - F.Y[ix][iy][iz] ) / dz;
}

__host__ __device__ inline void fdtdRefCurlXH ( float &C, xyz_t F, int ix, int iy, int iz )
{
  C = ( F.Z[ix][iy][iz] - F.Z[ix][iy - 1][iz] ) / dy - ( F.Y[ix][iy][iz] - F.Y[ix][iy][iz - 1] ) / dz;
}

__host__ __device__ inline void fdtdRefCurlYE ( float &C, xyz_t F, int ix, int iy, int iz )
{
  C = ( F.X[ix][iy][iz + 1] - F.X[ix][iy][iz] ) / dz - ( F.Z[ix + 1][iy][iz] - F.Z[ix][iy][iz] ) / dx;
}

__host__ __device__ inline void fdtdRefCurlYH ( float &C, xyz_t F, int ix, int iy, int iz )
{
  C = ( F.X[ix][iy][iz] - F.X[ix][iy][iz - 1] ) / dz - ( F.Z[ix][iy][iz] - F.Z[ix - 1][iy][iz] ) / dx;
}

__host__ __device__ inline void fdtdRefCurlZE ( float &C, xyz_t F, int ix, int iy, int iz )
{
  C = ( F.Y[ix + 1][iy][iz] - F.Y[ix][iy][iz] ) / dx - ( F.X[ix][iy + 1][iz] - F.X[ix][iy][iz] ) / dy;
}

__host__ __device__ inline void fdtdRefCurlZH ( float &C, xyz_t F, int ix, int iy, int iz )
{
  C = ( F.Y[ix][iy][iz] - F.Y[ix - 1][iy][iz] ) / dx - ( F.X[ix][iy][iz] - F.X[ix][iy - 1][iz] ) / dy;
}

// H and D; D Media and Volume
// RENAME THIS FUNCTION TO SOMETHING LIKE `fdtdRefFieldCOMMON' and remove `Wrapper' from function with the name `...Wrapper' below after this one.
__host__ __device__ inline void fdtdRefFieldWindowMedia ( xyz_t &A, xyz_t F, f3_t &IC, float &C, float Xm1C, float Xm2C, float Ym1C, float Ym2C, float Zm2C, float Zm3C, int ix, int iy, int iz, Curl_t curlx, Curl_t curly, Curl_t curlz, TFSF_t TFSFsrc )
{
  (*curlx) ( C, F, ix, iy, iz );
  C += (*TFSFsrc) ( YSRC, ix, iy, iz ) / dz;
  A.X[ix][iy][iz] = Xm1C * F.X[ix][iy][iz] + Xm2C * C;
  (*curly) ( C, F, ix, iy, iz );
  C += (*TFSFsrc) ( XSRC, ix, iy, iz ) / dz;
  A.Y[ix][iy][iz] = Ym1C * F.Y[ix][iy][iz] + Ym2C * C;//check this here
  (*curlz) ( C, F, ix, iy, iz );
  IC[ix][iy][iz] += C;// IC should be different for H, D, E fields and their components
  A.Z[ix][iy][iz] = F.Z[ix][iy][iz] + Zm2C * C + Zm3C * IC[ix][iy][iz];// Zm3C Should be ZERO (!) if called from any other fdtdRefFieldWindow....
}

//Function call wrapper for scalar update parameters
// H ONLY
// We probabaly need different functions for Xm2C for D and H fields, because of different PML array lengths (sic!) in two types and incorrect conversion from base type (!!!) (TODO)
// Type cast of Xm2C to DXYM2_t is dangerous, because only one member `.Window.Media[SCALARIDX]' has the same type and structure for both H and D fields ...
inline void fdtdRefFieldWindowMediaHWrapper ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib, TFSF_t TFSFsrc )
{
  /*
  HXYM2_t * HXm2C = dynamic_cast<HXYM2_t *>( &Xm2C ),
          * HYm2C = dynamic_cast<HXYM2_t *>( &Ym2C );
  DXYM2_t * DXm2C = dynamic_cast<DXYM2_t *>( &Xm2C ),
          * DYm2C = dynamic_cast<DXYM2_t *>( &Ym2C );
  */

  // No `const float &' here - the member type is not constant, but struct variable is.
  // TODO: Inline these two line assignments in one function.
  /*
  float &Xm2CWindowMediaScalar = ( HXm2C != NULL ) ? \
   HXm2C->Window.Media[SCALARIDX]                   : \
   ( DXm2C != NULL )              ? \
    DXm2C->Window.Media[SCALARIDX] : \
    *(float *)NULL,
        &Ym2CWindowMediaScalar = ( HYm2C != NULL ) ? \
   HYm2C->Window.Media[SCALARIDX]                   : \
   ( DYm2C != NULL )              ? \
    DYm2C->Window.Media[SCALARIDX] : \
    *(float *)NULL;
  */

  // All this above would have work fine but only if fdtdRefFieldWindowMediaWrapper IS NOT INLINED

  /*!!!PML instead of a Window for Zm2C!!!*/
  /*
  fdtdRefFieldWindowMedia ( A, F, IC, C, ((HXYM01_t)Xm1C).Window.Media[SCALARIDX], Xm2CWindowMediaScalar, ((HXYM01_t)Ym1C).Window.Media[SCALARIDX], Ym2CWindowMediaScalar, ((DZM2_t)Zm2C).PML.Media[SCALARIDX], 0, ix, iy, iz );
  */

  fdtdRefFieldWindowMedia ( A, F, IC, C, ((HXYM01_t &)Xm1C).Window.Media[SCALARIDX], ((HXYM2_t &)Xm2C).Window.Media[SCALARIDX], ((HXYM01_t &)Ym1C).Window.Media[SCALARIDX], ((HXYM2_t &)Ym2C).Window.Media[SCALARIDX], ((HZM2_t &)Zm2C).PML.Media[SCALARIDX], 0.0, ix, iy, iz, &fdtdRefCurlXE, &fdtdRefCurlYE, &fdtdRefCurlZE, TFSFsrc );
}

// D Media and Volume
inline void fdtdRefFieldWindowMediaDWrapper ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib, TFSF_t TFSFsrc )
{
  fdtdRefFieldWindowMedia ( A, F, IC, C, ((HXYM01_t &)Xm1C).Window.Media[SCALARIDX], ((DXYM2_t &)Xm2C).Window.Media[SCALARIDX], ((HXYM01_t &)Ym1C).Window.Media[SCALARIDX], ((DXYM2_t &)Ym2C).Window.Media[SCALARIDX], ((DZM2_t &)Zm2C).PML.Media[SCALARIDX], 0.0, ix, iy, iz, &fdtdRefCurlXH, &fdtdRefCurlYH, &fdtdRefCurlZH, TFSFsrc );
}

// H
inline void fdtdRefFieldWindowVolume ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib, TFSF_t TFSFsrc )
{
  fdtdRefFieldWindowMedia ( A, F, IC, C, ((HXYM01_t &)Xm1C).Window.Media[SCALARIDX], ((f2da_t &)(((HXYM2_t &)Xm2C).Window.Volume))[ia][ib], ((HXYM01_t &)Ym1C).Window.Media[SCALARIDX], ((f2da_t &)(((HXYM2_t &)Ym2C).Window.Volume))[ia][ib], ((f2da_t &)(((HZM2_t &)Zm2C).PML.Volume))[ia][ib]/*PML!!!!!!*/, 0, ix, iy, iz, &fdtdRefCurlXE, &fdtdRefCurlYE, &fdtdRefCurlZE, TFSFsrc );
}

// H
inline void fdtdRefFieldPMLMediaH ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib, TFSF_t TFSFsrc )
{
  fdtdRefFieldWindowMedia ( A, F, IC, C, ((HXYM01_t &)Xm1C).PML.Media[il], ((HXYM2_t &)Xm2C).PML.Media[il], ((HXYM01_t &)Ym1C).PML.Media[il], ((HXYM2_t &)Ym2C).PML.Media[il], ((HZM2_t &)Zm2C).PML.Media[il], ((HZM3_t &)Zm3C).PML.Media[il], ix, iy, iz, &fdtdRefCurlXE, &fdtdRefCurlYE, &fdtdRefCurlZE, TFSFsrc );
}

// D PML Media and Volume; overloadng of fdtdRefFieldPMLMedia
inline void fdtdRefFieldPMLMediaD ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib, TFSF_t TFSFsrc )
{
  fdtdRefFieldWindowMedia ( A, F, IC, C, ((DXYM01_t &)Xm1C).PML.Media[il], ((DXYM2_t &)Xm2C).PML.Media[il], ((DXYM01_t &)Ym1C).PML.Media[il], ((DXYM2_t &)Ym2C).PML.Media[il], ((DZM2_t &)Zm2C).PML.Media[SCALARIDX], ((DZM3_t &)Zm3C).PML.Media[il], ix, iy, iz, &fdtdRefCurlXH, &fdtdRefCurlYH, &fdtdRefCurlZH, TFSFsrc );
}

//H
inline void fdtdRefFieldPMLVolumeH ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib, TFSF_t TFSFsrc )
{
  fdtdRefFieldWindowMedia ( A, F, IC, C, ((HXYM01_t &)Xm1C).PML.Media[il]/*check if media instead of volume!!*/, ((f3da_t &)(((HXYM2_t &)Xm2C).PML.Volume))[il][ia][ib], ((HXYM01_t &)Ym1C).PML.Media[il]/*same note here!!!*/, ((f3da_t &)(((HXYM2_t &)Ym2C).PML.Volume))[il][ia][ib], ((f2da_t &)(((HZM2_t &)Zm2C).PML.Volume))[ia][ib], ((f3da_t &)(((HZM3_t &)Zm3C).PML.Volume))[il][ia][ib], ix, iy, iz, &fdtdRefCurlXE, &fdtdRefCurlYE, &fdtdRefCurlZE, TFSFsrc );
}

// H and D
// Call for both device and host
__host__ __device__ inline void fdtdRefSingleXY ( int xa, int xb, int ya, int yb, int &ix, int &iy, FieldComponents_t &FCout, FieldComponents_t &FCin, UpdateCoefficients_t &UC, f3_t &ICA, f3_t &ICB, float &C, int iz, int &il, int ilMin, fdtdRefField_t fdtdRFH, fdtdRefField_t fdtdRFD, TFSF_t TFSFsrcE, TFSF_t TFSFsrcH )
{
  (*fdtdRFH) ( FCout.H, FCin.H, ICA, C, UC.H.X.m1, UC.H.X.m2, UC.H.Y.m1, UC.H.Y.m2, UC.H.Z.m2, UC.H.Z.m3, ix, iy, iz, il - ilMin, ix - xa, iy - ya, TFSFsrcE );
  (*fdtdRFD) ( FCout.D, FCin.D, ICB, C, UC.D.X.m1, UC.D.X.m2, UC.D.Y.m1, UC.D.Y.m2, UC.D.Z.m2, UC.H.Z.m3, ix, iy, iz, il - ilMin, ix - xa, iy - ya, TFSFsrcH );
}

// TODO : transform all those functions to class methods with members instead of a function parameters. Use xisting struct declarations in FDTD3dShared.h
__device__ inline void fdtdRef4SingleXY ( int xhalfpre, int xhalfpost, int yhalfpre, int yhalfpost, int dimx, int dimy, int &ix, int &iy, FieldComponents_t &FCout, FieldComponents_t &FCin, UpdateCoefficients_t &UC, f3_t &ICA, f3_t &ICB, float &C, int iz, int il, fdtdRefField_t fdtdRFH, fdtdRefField_t fdtdRFD, TFSF_t TFSFsrcE, TFSF_t TFSFsrcH )
{
  fdtdRefSingleXY ( 0, xhalfpre, 0, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
  fdtdRefSingleXY ( xhalfpost, dimx, 0, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
  fdtdRefSingleXY ( xhalfpre, xhalfpost, 0, yhalfpre, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
  fdtdRefSingleXY ( xhalfpre, xhalfpost, yhalfpost, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
}

#include "FDTD3dGPUKernel.cuh"

bool getTargetDeviceGlobalMemSize ( int *totalblockspermp, int *totalthreadspermp, int *totalmps, memsize_t *totalmem, const int argc, const char **argv )
{
    int               deviceCount  = 0;
    int               targetDevice = 0;
    int               mpcount      = 0;
    int               mpresthreads = 0;
    size_t            memsize      = 0;

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
    mpcountt = deviceProp.multiProcessorCount;
    mpresthreads = deviceProp.maxThreadsPerMultiProcessor;

    // Save the results
    *totalmem = (memsize_t)memsize;
    *totalmps = mpcount;
    *totalthreadspermp = mpresthreads;
    switch ( deviceProp.major )
    {
      case 3:
        *totalthreadspermp = 16;
        break;
      case 5:
        *totalthreadspermp = 32;
        break;
      default:
        *totalthreadspermp = 8;
        break;
    }
    return true;
}

bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv, int blockSize, int blockXSize)
{
    const int         outerDimx  = dimx + 2 * radius;
    const int         outerDimy  = dimy + 2 * radius;
    const int         outerDimz  = dimz + 2 * radius;
    const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
    int               deviceCount  = 0;
    int               targetDevice = 0;
    float            *bufferOut    = 0;
    float            *bufferIn     = 0;
    dim3              dimBlock;
    dim3              dimGrid;

    // Ensure that the inner data starts on a 128B boundary
    const int padding = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
    cudaEvent_t profileStart = 0;
    cudaEvent_t profileEnd   = 0;
    // In timeframe we need to advance by half-timestep due to different
    // steps in H and D and also in E and B fields update equations
    const int profileTimesteps = 2 * timesteps - 1;

    if (profileTimesteps < 1)
    {
        printf(" cannot profile with fewer than two timesteps (timesteps=%d), profiling is disabled.\n", timesteps);
    }

#endif

    // Check the radius is valid
    if (radius != RADIUS)
    {
        printf("radius is invalid, must be %d - see kernel for details.\n", RADIUS);
        exit(EXIT_FAILURE);
    }

    // Get the number of CUDA enabled GPU devices
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaSetDevice(targetDevice));

    // Allocate memory buffers
    checkCudaErrors(cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

    // Check for a command-line specified block size
    int userBlockSize;

    if ( !checkCmdLineFlag(argc, (const char **)argv, "block-size") )
      userBlockSize = blockSize;

    userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
    // Constrain to a multiple of k_blockDimX
    userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

    // Constrain within allowed bounds
    userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);

    // Check the device limit on the number of threads
    struct cudaFuncAttributes funcAttrib;
    checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));

    userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

    // Set the block size
    //dimBlock.x = k_blockDimX;
    dimBlock.x = blockXSize;
    // Visual Studio 2005 does not like std::min
    //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX, (size_t)k_blockDimMaxY);
    dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY;
    dimGrid.x  = (unsigned int)ceil((float)dimx / dimBlock.x);
    dimGrid.y  = (unsigned int)ceil((float)dimy / dimBlock.y);
    printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
    printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

    // Check the block size is valid
    if (dimBlock.x < RADIUS || dimBlock.y < RADIUS)
    {
        printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", dimBlock.x, dimBlock.y, RADIUS);
        exit(EXIT_FAILURE);
    }

    // Copy the input to the device input buffer
    checkCudaErrors(cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the input to the device output buffer (actually only need the halo)
    checkCudaErrors(cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the coefficients to the device coefficient buffer
    checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));

#ifdef GPU_PROFILING

    // Create the events
    checkCudaErrors(cudaEventCreate(&profileStart));
    checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

    // Execute the FDTD
    float *bufferSrc = bufferIn + padding;
    float *bufferDst = bufferOut + padding;
    printf(" GPU FDTD loop\n");

#ifdef GPU_PROFILING
    // Enqueue start event
    checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

   // In timeframe we need to advance by half-timestep due to different
   // steps in H and D and also in E and B fields update equations
   //TODO : remove half-steps!!!!!

   argsKPT = { .dimx = dimx,\
               .dimy = dimy,\
               .dimz = dimz,\
               .input = input,\
               .dimGrid = dimGrid,\
               .dimBlock = dimBlock,\
               .maxSharedMemPerBlock = maxSharedMemPerBlock; \
               .output = output; };

   for (int ihalft = 0 ; ihalft < 2 * timesteps ; ihalft++)
    {
        printf("\tt = %d ", ihalft);

        for ( idxGrid = 0; idxGrid < maxGrids; idxGrid++ )
        {
          pthread_create ( &pthreadLoaders[idxGrid], NULL reloadGlobal2SharedMemChunk, argsRMC );
        }

        // Launch the kernel
        printf("launch kernel\n");
        if ( pthread_create ( &pthreadKernel, NULL, launchKernelPThreadAsync, argsKPT) )
        {
          fprintf(stderr, "Error creating pthread");
          checkCudaErrors(1);
        }
        pthread_join ( pthreadKernel, NULL );
        for ( idxGrid = 0; idxGrid < maxGrids; idxGrid++ )
        {
          if ( pthread_join ( pthreadLoaders[idxGrid], NULL ) )
          {
            fprintf(stderr, "Error joining pthread");
            checkCudaErrors(1);
          }
        }// TODO : STOPED HERE .
        // TODO : We need to define new kernel with TF/SF corrections enabled and another 2D kernel for calclating boundaries. :(
        // TODO : Also we nee to define pre-boundary value storage as @D dimensionsl array 
        // ... and there are probably 2 kinds of those arrays and 2 sets ofKernel configuration parameters: for 4 long sides and fo 2 short ones (4 belong to x's and y's and 2 -  z's)

        // Toggle the buffers
        // Visual Studio 2005 does not like std::swap
        //    std::swap<float *>(bufferSrc, bufferDst);
        float *tmp = bufferDst;
        bufferDst = bufferSrc;
        bufferSrc = tmp;
    }

    printf("\n");

#ifdef GPU_PROFILING
    // Enqueue end event
    checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

    // Read the result back, result is in bufferSrc (after final toggle)
    checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Report time
#ifdef GPU_PROFILING
    float elapsedTimeMS = 0;

    if (profileTimesteps > 0)
    {
        checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
    }

    if (profileTimesteps > 0)
    {
        // Convert milliseconds to seconds
        double elapsedTime    = elapsedTimeMS * 1.0e-3;
        double avgElapsedTime = elapsedTime / (double)profileTimesteps;
        // Determine number of computations per timestep
        size_t pointsComputed = dimx * dimy * dimz;
        // Determine throughput
        double throughputM    = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        printf("FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, NumDevsUsed = %u, Blocksize = %u\n",
               throughputM, avgElapsedTime, pointsComputed, 1, dimBlock.x * dimBlock.y);
    }

#endif

    // Cleanup
    if (bufferIn)
    {
        checkCudaErrors(cudaFree(bufferIn));
    }

    if (bufferOut)
    {
        checkCudaErrors(cudaFree(bufferOut));
    }

#ifdef GPU_PROFILING

    if (profileStart)
    {
        checkCudaErrors(cudaEventDestroy(profileStart));
    }

    if (profileEnd)
    {
        checkCudaErrors(cudaEventDestroy(profileEnd));
    }

#endif
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    return true;
}
