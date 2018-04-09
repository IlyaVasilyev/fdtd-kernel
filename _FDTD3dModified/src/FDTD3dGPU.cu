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


// TODO : Move these defs below somewhere else PLEASE !!!
#define float3darray_t(TYPEDEFNAME, DIMI, DIMJ, DIMK) float TYPEDEFNAME[DIMI][DIMJ][DIMK]
#define float2darray_t(TYPEDEFNAME, DIMJ, DIMK) float TYPEDEFNAME[DIMJ][DIMK]
typedef float3darray_t(f3da_t,RADIUS_SHARED,VOLUME_SHARED,VOLUME_SHARED);
typedef float2darray_t(f2da_t,VOLUME_SHARED,VOLUME_SHARED);

// TODO : Figure out what is `padding'.

cudaEvent_t *eventReadWrite;
cudaStream_t streamCopyH2D, streamCopyD2H, streamCalc, streamMain;

void *launchKernelPThreadAsync ( void *arguments )
{
  KernelPThreadArgs_t * &args = arguments;

  FiniteDifferencesKernel<<<args.dimGrid, args.dimBlock, args.maxSharedMemPerBlock>>>(args.bufferDst, args.bufferSrc, args.dimx, args.dimy, args.dimz);

  return NULL;
}

void * reloadGlobal2SharedMemSlice ( void *arguments )
{
  ReloadMemChunkArgs_t &args = *((ReloadMemChunkArgs_t *)arguments);

  const float * &srcBuffer = args.bufferSrc;
                 /* dimensions : [args.maxChunks][args.dimSlice] */

  float * &dstBuffer = args.bufferDst,
           /* dimensions : [args.maxChunks][args.dimSlice] */
        * &ioBuffer  = args.ioBuffer;
           /* dimensions : [args.dimSlice] */

#ifdef DEBUG_INFO

  unsigned char * &debugFlags = args.debugFlags;

  debugFlags[args.idxSlice] = 1;

#endif

  // TODO (DONE) : FIX CRITICAL ERROR : Loop does not count \
  //                                     timesteps:          
  // TODO : Verify that there are no more reference-value        \
  //         initializations in threads, launched from loop with \
  //         variable parameters:                                 
  for ( int timestep = 0; timestep < args.timesteps; timestep++ )
  {
    for ( int idxChunk = 0; idxChunk < args.maxChunks; idxChunk++ )
    {
      /* STEP2 : Waiting until current slice will be reloaded. */
      // TODO : Rename to `...WaitWhileRefreshing...':
      while ( args.hostWait4RefreshGlobalSlice[args.idxSlice] ) {}

      // TODO : Figure out where this should be:
      /*
      checkCudaErrors ( cudaEventRecord ( eventReadWrite[args.idxSlice], \
                                          streamCopyD2H                  \
                      )                 );                                
      */

      // TODO (DONE) : Verify if this is copying stream and \
      //                NOT calculating stream:              
      checkCudaErrors ( cudaMemcpyAsync                                 \
                        (                                               \
                          &dstBuffer[args.chunkSize * idxChunk          \
                                     + args.sliceSize * args.idxSlice], \
                          &ioBuffer[args.sliceSize * args.idxSlice],    \
                          args.sliceSize * sizeof ( float ),            \
                          cudaMemcpyDeviceToHost,                       \
                          streamCopyD2H                                 \
                        )                                               \
                      );                                                 

      /* STEP15 : Pausing sream `streamCopyH2D' while waiting each stream */
      /*           `streamCopyD2H' copying date from device back to host. */
      checkCudaErrors ( cudaEventRecord ( eventReadWrite[args.idxSlice], \
                                          streamCopyD2H                  \
                      )                 );                                
      checkCudaErrors ( cudaStreamWaitEvent              \
                        (                                \
                          streamCopyH2D,                 \
                          eventReadWrite[args.idxSlice], \
                          0                              \
                        )                                \
                      );                                  

      // 4.04.16 - fixed idxChunk to idxChunk + 1
      /* STEP16 : Copying new data to device in specific memory location of */
      /*           each slice.                                              */
      // TODO : Optimize two IFs below:
      if ( idxChunk < args.maxChunks - 1 )
        // TODO (DONE) : Verify if this is copying stream and \
        //                NOT calculating stream!!!            
        checkCudaErrors ( cudaMemcpyAsync                                 \
                          (                                               \
                            &ioBuffer[args.sliceSize * args.idxSlice],    \
                            &srcBuffer[args.chunkSize * (idxChunk+1)      \
                                       + args.sliceSize * args.idxSlice], \
                            args.sliceSize * sizeof ( float ),            \
                            cudaMemcpyHostToDevice,                       \
                            streamCopyH2D                                 \
                          )                                               \
                        );                                                 
      if ( idxChunk == args.maxChunks )
        // TODO (DONE) : Verify if this is copying stream and \
        //                NOT (!!!) calculating stream:        
        // Returning to the first chunk for another timestep:
        checkCudaErrors ( cudaMemcpyAsync                               \
                          (                                             \
                            &ioBuffer[args.sliceSize * args.idxSlice],  \
                            &dstBuffer[args.sliceSize * args.idxSlice], \
                            args.sliceSize * sizeof ( float ),          \
                            cudaMemcpyHostToDevice,                     \
                            streamCopyH2D                               \
                          )                                             \
                        );                                               

      /* STEP17 : Resetting flag to "wait" condition. */
      args.hostWait4RefreshGlobalSlice[args.idxSlice] = 1;
    }
  }
  return NULL;
}

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

// TODO : I AM HERE (29.09.16) : Solve the problem with shared memory   \
//                                indexes, select either structured or  \
//                                raw allocation (for both constant and \
//                                shared memory):                        
// TODO (DONE) : IMPORTANT : Insert `__syncthreads()' in this function, \
//                            because it needed by threads, accessing   \
//                            (reading and writing back) shared memory  \
//                            `A' and `F' (which are the same memory    \
//                            addresses):                                
// TODO : RENAME THIS FUNCTION TO SOMETHING LIKE `fdtdRefFieldCOMMON' and \
//         remove "Wrapper" from function with the name `...Wrapper'      \
//         below after this one:                                           
// H and D; D Media and Volume:
__host__ __device__ inline void fdtdRefFieldWindowMedia                     \
                                (                                           \
                                  xyz_t &A, xyz_t F, f3_t &IC,              \
                                  float &C, float &fA,                      \
                                  float Xm1C, float Xm2C,                   \
                                  float Ym1C, float Ym2C,                   \
                                  float Zm2C, float Zm3C,                   \
                                  int ix, int iy, int iz,                   \
                                  Curl_t curlx, Curl_t curly, Curl_t curlz, \
                                  TFSF_t TFSFsrc                            \
                                )                                            
{
  (*curlx) ( C, F, ix, iy, iz );
  C += (*TFSFsrc) ( YSRC, ix, iy, iz ) / dz;
  fA = Xm1C * F.X[ix][iy][iz] + Xm2C * C;
  __syncthreads();
  A.X[ix][iy][iz] = fA;
  __syncthreads();
  (*curly) ( C, F, ix, iy, iz );
  C += (*TFSFsrc) ( XSRC, ix, iy, iz ) / dz;
  fA = Ym1C * F.Y[ix][iy][iz] + Ym2C * C;
  __syncthreads();
  A.Y[ix][iy][iz] = fA;  // Check this here.
  __syncthreads();
  (*curlz) ( C, F, ix, iy, iz );
  // `IC' should be different for H, D, E fields and their components:
  IC[ix][iy][iz] += C;
  // `Zm3C' Should be ZERO (!) if called from any other \
  //  `fdtdRefFieldWindow...':                           
  fA = F.Z[ix][iy][iz] + Zm2C * C + Zm3C * IC[ix][iy][iz];
  __syncthreads();
  A.Z[ix][iy][iz] = fA;
  __syncthreads();
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

// D PML Media and Volume; overloadng of `fdtdRefFieldPMLMedia':
inline void fdtdRefFieldPMLMediaD                                 \
            (                                                     \
              xyz_t &A, xyz_t F, f3_t &IC, float &C,              \
              structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, \
              structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, \
              structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, \
              structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, \
              structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, \
              structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, \
              int ix, int iy, int iz,                             \
              int il, int ia, int ib,                             \
              TFSF_t TFSFsrc                                      \
            )                                                      
{
  fdtdRefFieldWindowMedia ( A, F, IC, C,                                    \
                            ((DXYM01_t &)Xm1C).PML.Media[il],               \
                            ((DXYM2_t &)Xm2C).PML.Media[il],                \
                            ((DXYM01_t &)Ym1C).PML.Media[il],               \
                            ((DXYM2_t &)Ym2C).PML.Media[il],                \
                            ((DZM2_t &)Zm2C).PML.Media[SCALARIDX],          \
                            ((DZM3_t &)Zm3C).PML.Media[il],                 \
                            ix, iy, iz,                                     \
                            &fdtdRefCurlXH, &fdtdRefCurlYH, &fdtdRefCurlZH, \
                            TFSFsrc                                         \
                          );                                                 
}

//H
inline void fdtdRefFieldPMLVolumeH ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib, TFSF_t TFSFsrc )
{
  fdtdRefFieldWindowMedia ( A, F, IC, C, ((HXYM01_t &)Xm1C).PML.Media[il]/*check if media instead of volume!!*/, ((f3da_t &)(((HXYM2_t &)Xm2C).PML.Volume))[il][ia][ib], ((HXYM01_t &)Ym1C).PML.Media[il]/*same note here!!!*/, ((f3da_t &)(((HXYM2_t &)Ym2C).PML.Volume))[il][ia][ib], ((f2da_t &)(((HZM2_t &)Zm2C).PML.Volume))[ia][ib], ((f3da_t &)(((HZM3_t &)Zm3C).PML.Volume))[il][ia][ib], ix, iy, iz, &fdtdRefCurlXE, &fdtdRefCurlYE, &fdtdRefCurlZE, TFSFsrc );
}

// H and D:
// Function for both device and host:
__host__ __device__ inline void fdtdRefSingleXY                    \
                                (                                  \
                                  int xa, int xb, int ya, int yb,  \
                                  int &ix, int &iy,                \
                                  FieldComponents_t &FCout,        \
                                  FieldComponents_t &FCin,         \
                                  UpdateCoefficients_t &UC,        \
                                  f3_t &ICA, f3_t &ICB,            \
                                  float &C, float &F,              \
                                  int iz, int &il, int ilMin,      \
                                  fdtdRefField_t fdtdRFH,          \
                                  fdtdRefField_t fdtdRFD,          \
                                  TFSF_t TFSFsrcE, TFSF_t TFSFsrcH \
                                )                                   
{
  // TODO : FIX A POSSIBLE ERROR : Both functions receive same parameters \
  //                                with iy-ya and ix-ia, and there is no \
  //                                xb and yb variables:                   
  (*fdtdRFH) ( FCout.H, FCin.H, ICA, C, F,                 \
               UC.H.X.m1, UC.H.X.m2, UC.H.Y.m1, UC.H.Y.m2, \
               UC.H.Z.m2, UC.H.Z.m3,                       \
               ix, iy, iz,                                 \
               il - ilMin, ix - xa, iy - ya,               \
               TFSFsrcE                                    \
             );                                             
  (*fdtdRFD) ( FCout.D, FCin.D, ICB, C, F,                 \
               UC.D.X.m1, UC.D.X.m2, UC.D.Y.m1, UC.D.Y.m2, \
               UC.D.Z.m2, UC.H.Z.m3,                       \
               ix, iy, iz,                                 \
               il - ilMin, ix - xa, iy - ya,               \
               TFSFsrcH                                    \
             );                                             
}

// TODO : Transform all those functions into class methods with members      \
//         instead of a function parameters. Use xisting struct declarations \
//         in FDTD3dShared.h:                                                 
__device__ inline void fdtdRef4SingleXY                                  \
                       ( int xhalfpre, int xhalfpost,                    \
                         int yhalfpre, int yhalfpost,                    \
                         int dimx, int dimy,                             \
                         int &ix, int &iy,                               \
                         FieldComponents_t &FCout,                       \
                         FieldComponents_t &FCin,                        \
                         UpdateCoefficients_t &UC,                       \
                         f3_t &ICA, f3_t &ICB,                           \
                         float &C, float &F,                             \
                         int iz, int il,                                 \
                         fdtdRefField_t fdtdRFH, fdtdRefField_t fdtdRFD, \
                         TFSF_t TFSFsrcE, TFSF_t TFSFsrcH                \
                       )                                                  
{
  fdtdRefSingleXY ( 0, xhalfpre, 0, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, F, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
  fdtdRefSingleXY ( xhalfpost, dimx, 0, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, F, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
  fdtdRefSingleXY ( xhalfpre, xhalfpost, 0, yhalfpre, ix, iy, FCout, FCin, UC, ICA, ICB, C, F, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
  fdtdRefSingleXY ( xhalfpre, xhalfpost, yhalfpost, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, F, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrcE, TFSFsrcH );
}

#include "FDTD3dGPUKernel.cuh"

// TODO : Rename `getTargetDeviceGlobalMemSize' \
//         to `getTargetDeviceProperties':       
// TODO : Use references `&' instead of pointers `*' for all \
//         returned arguments:                                
bool getTargetDeviceProperties ( int *totalblockspermp,  \
                                 int *totalthreadspermp, \
                                 int *totalmps,          \
                                 memsize_t *totalmem,    \
                                 int &maxTotalThreads,   \
                                 const int argc,         \
                                 const char **argv       \
                               )                          
{
    int    deviceCount  = 0;
    int    targetDevice = 0;
    int    mpcount      = 0;
    int    mpresthreads = 0;
    size_t memsize      = 0;

    // Get the number of CUDA enabled GPU devices:
    printf(" cudaGetDeviceCount\n");
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default):
    targetDevice = findCudaDevice(argc, (const char **)argv);

    // Query target device for maximum memory allocation:
    printf(" cudaGetDeviceProperties\n");
    struct cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

    // Query target device maximum number of concurrent threads:
    CUresult error = cuDeviceGetAttribute                                  \
                     (                                                     \
                       &maxTotalThreads,                                   \
                       CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, \
                       targetDevice                                        \
                     );                                                     

    memsize = deviceProp.totalGlobalMem;
    mpcountt = deviceProp.multiProcessorCount;
    mpresthreads = deviceProp.maxThreadsPerMultiProcessor;

    // Save the results:
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

void constrainCompUnitDims ( int &userUnitSize,
                             int dimInternalX, int dimInternalY,
                             const int unitDimX, const int unitDimY,
                             const int unitDimYMax,
                             const int unitSizeMax, const int unitSizeMin,
                             dim3 &unitDims,
                             int maxCudaUnitSize,
                             int minInternalUnitSize,
                             int argc, char **argv, char *arg_name
                           )
{
  // TODO : Ensure that all constants have proper values          \
  //         for maximum number of available computational units:  
  // Check for a command-line specified unit size:
  if ( !checkCmdLineFlag ( argc, (const char **)argv, arg_name ) )
    userUnitSize = dimInternalX * dimInternalY;
  else
    userUnitSize = getCmdLineArgumentInt ( argc, argv, arg_name );

  // Constrain to a multiple of unitDimX:
  userUnitSize = userUnitSize/unitDimX * unitDimX;

  // Set the unit size:
  unitDims.x = dimInternalX;

  // Constrain within allowed bounds:
  if ( userUnitSize < unitSizeMin )
  {
    userUnitSize = unitSizeMin;
    unitDims.x = unitDimX;
  }
  if ( userUnitSize > unitSizeMax )
  {
    userUnitSize = unitSizeMax;
    unitDims.x = unitDimX;
  }

  if ( minInternalUnitSize > 0 )
    userUnitSize = MIN ( userUnitSize, maxCudaUnitSize );
  else
    userUnitSize = MIN ( minInternalUnitSize*userUnitSize, maxCudaUnitSize );

  // Visual Studio 2005 does not like `std::min':
  unitDims.y = ( (userUnitSize/unitDims.x) < (size_t)unitDimMaxY ) \
                ? (userUnitSize/unitDims.x)                        \
                : (size_t)unitDimY;                                 

  return NULL;
}

// TODO : Change all `float *' to `FieldComponents_t *' and also \
//         all types of related variables                        \
//         from `float' to `FieldComponents_t':                   
// TODO : Remove three-dimensional structures `***' \
//         and use simple arrays `*' instead :       
bool fdtdGPU ( int argc, char **argv,                                     \
               const FieldComponents_t *input, FieldComponents_t *output, \
               const FieldComponents_t *inputTFSFsrc,                     \
               const UpdateCoefficients_t &coeffs,                        \
               int timesteps,                                             \
               const int radius,                                          \
               int maxChunks, int chunkSize,                              \
               int sliceSize, int dimSlice,                               \
               int blockSize,                                             \
               int dimx, int dimy, int dimz,                              \
               int dimxBlock, int dimyBlock,                              \
               int dimThreadsX, int dimThreadsY,                          \
               int maxthreadspermp                                        \
             )                                                             
{
  // TODO : I AM HERE - VERIFY SYNCHRONIZATION BETWEEN ALL PARALLEL STREAMS, \
  //         THREADS, PTHREADS:                                               
  // TODO : Move all declarations to the top of the function:

  unsigned char *debugFlags;

  int userBlockSize, userGridSize,
      targetDevice = 0,
      deviceCount  = 0;

  FieldComponents_t *ioBuffer  = 0,
                    *bufferSrc = input,
                    *bufferDst = output;

  clock_t *globalClockGPU;

  dim3 dimBlock,
       dimGrid;

  const int     outerDimx  = dimx + 2 * radius;
  const int     outerDimy  = dimy + 2 * radius;
  const int     outerDimz  = dimz + 2 * radius;
  const size_t  volumeSize = outerDimx * outerDimy * outerDimz;
  int           deviceCount  = 0;
  int           targetDevice = 0;

  // Synchronization flags:
  unsigned char *bContinue,/* byte */
                *hostWait4RefreshingChunk_WhileLoadingSlices,/* byte */
                *hostWaitWhileLoadingGlobalChunk,/* byte */
                *hostWait4RefreshGlobalSlice,/* array of bytes */
                *deviceGlobalRefreshFlags,/* array of bytes */
                *deviceWaitWhileLoadingGlobalChunk;/* byte */

  struct cudaFuncAttributes funcAttrib;

  // TODO : Make sure that all fields in assigned structs are in right \
  //         order with variable's struct fields:                       
  // TODO : Figure out why argsCKSA does not have reference type:
  // CKSA - Control Kernel Stream Arguments.
  // Initialization of `argsCKSA' const members using copy constructor:
  TestControlKernelArguments_t argsCKSA \
  (                                     \
    NULL, NULL, NULL, NULL, NULL, NULL, \
    dimThreadsX, dimThreadsY,           \
    dimSlice,                           \
    maxChunks,                          \
    dimxBlock,                          \
    dimyBlock,                          \
    timesteps                           \
  );                                     
  TestKernelArguments_t argsKPT                                     \
  (                                                                 \
    dimx, dimy, dimz,                                               \
    dimxBlock, dimyBlock, dimSlice,                                 \
    dimThreadsX, dimThreadsY,                                       \
    /* TODO : Extra zdimblock dimension !!! VERIFY all           */ \
    /*         deviceGlobalRefreshFlags and linked host flags!!! */ \
    maxChunks,                                                      \
    timesteps,                                                      \
    NULL, NULL, NULL, NULL, NULL                                    \
  );                                                                 
  ReloadMemChunkArgs_t *argsRMC;

  //pthread_t pthreadLoaders[maxChunks];
  pthread_t pthreadLoaders[dimSlice];

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING

    cudaEvent_t profileStart = 0;
    cudaEvent_t profileEnd   = 0;
    // In timeframe we need to advance by half-timestep due to different \
    //  steps in H and D and also in E and B fields update equations:     
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

#ifdef DEBUG_INFO

  debugFlags = (unsigned char *)(calloc ( dimSlice, 1 ));

#endif

  // Allocate memory buffers:
  checkCudaErrors ( cudaMalloc (                                       \
                                 (void **)&ioBuffer,                   \
                                 chunkSize * sizeof(FieldComponents_t) \
                  )            );                                       
  checkCudaErrors ( cudaMalloc (                           \
                                 (void **)&globalClockGPU, \
                                 sizeof(clock_t)           \
                  )            );                           
  checkCudaErrors ( cudaMalloc (                                     \
                                 (void **)&deviceGlobalRefreshFlags, \
                                 dimxBlock * dimyBlock * dimSlice    \
                                  * dimThreadsX * dimThreadsY        \
                  )            );                                     
  checkCudaErrors ( cudaMalloc                                     \
                    (                                              \
                      (void **)&deviceWaitWhileLoadingGlobalChunk, \
                      1                                            \
                    )                                              \
                  );                                                
  // If there are pinned memory required for syncronization purposes by \
  //  many threads or concurrent blocks, then it's better to use        \
  //  `cudaHostAlloc' with `cudaHostAllocPortable' flag rather than     \
  //  `cudaMallocHost', because in the first case memory will be        \
  //  considered as pinned "... by all CUDA contexts, not just the one  \
  //  that performed the allocation."                                    
  checkCudaErrors ( cudaHostAlloc                            \
                    (                                        \
                      (void **)&hostWait4RefreshGlobalSlice, \
                      dimSlice,                              \
                      cudaHostAllocPortable                  \
                    )                                        \
                  );// PINNED.                                
  checkCudaErrors                                            \
  (                                                          \
    cudaHostAlloc                                            \
    (                                                        \
      (void **)&hostWait4RefreshingChunk_WhileLoadingSlices, \
      1,                                                     \
      cudaHostAllocPortable                                  \
    )                                                        \
  );// PINNED.                                                
  checkCudaErrors ( cudaHostAlloc                                \
                    (                                            \
                      (void **)&hostWaitWhileLoadingGlobalChunk, \
                      1,                                         \
                      cudaHostAllocPortable                      \
                    )                                            \
                  );// PINNED.                                    
  checkCudaErrors ( cudaHostAlloc           \
                    (                       \
                      (void **)&bContinue,  \
                      1,                    \
                      cudaHostAllocPortable \
                    )                       \
                  );// PINNED.               

  checkCudaErrors ( cudaMemset ( deviceGlobalRefreshFlags,        \
                                 0,                               \
                                 dimxBlock * dimyBlock * dimSlice \
                                  * dimThreadsX * dimThreadsY     \
                  )            );                                  
  checkCudaErrors ( cudaMemset ( deviceWaitWhileLoadingGlobalChunk, \
                                 1, 1                               \
                  )            );                                    
  memset ( hostWait4RefreshGlobalSlice, 1, dimSlice );
  // TODO : Ensure that `0' stands for "not waiting" and also that we need \
  //         "not waiting" state from the begining of program execution in \
  //         `hostWait4RefreshingChunk_WhileLoadingSlices'                 \
  //         and                                                           \
  //         `hostWaitWhileLoadingGlobalChunk'                             \
  //         variables:                                                     
  *hostWait4RefreshingChunk_WhileLoadingSlices = 1;
  *hostWaitWhileLoadingGlobalChunk = 1;
  *bContinue = 0;

  // Initialization of `argsCKSA' non-const fields:
  argsCKSA.hostWait4RefreshGlobalSlice = hostWait4RefreshGlobalSlice;
  argsCKSA.hostWait4RefreshingChunk_WhileLoadingSlices = \
   hostWait4RefreshingChunk_WhileLoadingSlices;           
  argsCKSA.hostWaitWhileLoadingGlobalChunk = \
   hostWaitWhileLoadingGlobalChunk;           
  argsCKSA.deviceWaitWhileLoadingGlobalChunk = \
   deviceWaitWhileLoadingGlobalChunk;           
  argsCKSA.bContinue = bContinue;
  argsCKSA.deviceGlobalRefreshFlags = deviceGlobalRefreshFlags;

  // Check the device limit on the number of threads:
  checkCudaErrors ( cudaFuncGetAttributes ( &funcAttrib,            \
                                            FiniteDifferencesKernel \
                  )                       );                         

  constrainCompUnitDims ( userBlockSize, dimThreadsX , dimThreadsY, \
                          k_blockDimX, k_blockDimY, k_blockDimMaxY, \
                          k_blockSizeMax, k_blockSizeMin,           \
                          dimBlock,                                 \
                          0, funcAttrib.maxThreadsPerBlock,         \
                          argc, argv, "block-size"                  \
                        );                                           

  constrainCompUnitDims ( userBlockSize, dimxBlock , dimyBlock,      \
                          k_gridDimX, k_gridDimY, k_gridDimMaxY,     \
                          k_gridSizeMax, k_gridSizeMin,              \
                          dimGrid,                                   \
                          0, funcAttrib.maxThreadsPerMultiProcessor, \
                          argc, argv, "grid-size"                    \
                        );                                            

  // TODO (DONE) : Implement `k_gridDimX', `k_gridDimY', `k_gridDimMaxY', \
  //                and etc. in the same manner as for block dimensions,  \
  //                but be sure to include thread dimension checks        \
  //                for the grid:                                          
  // Check if block and grid sizes are valid:
  if ( ( dimBlock.x < outerDimx )    \
       || ( dimBlock.y < outerDimy ) \
     )                                
  {
    printf("invalid block size, x (%d) and y (%d) must be >= radius (%d or %d).\n", dimBlock.x, dimBlock.y, outerDimx, outerDimy);
    exit(EXIT_FAILURE);
  }
  if ( ( dimBlock.x * dimGrid.x <= outerDimx )    \
       || ( dimBlock.y * dimGrid.y <= outerDimy ) \
     )                                             
  {
    printf ( "invalid grid size, x (%d) and y (%d) must be >= radius (%d or %d).\n", dimBlock.x * dimGrid.x, dimBlock.y * dimGrid.y, outerDimx, outerDimy );
    exit ( EXIT_FAILURE );
  }

  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

#ifdef GPU_PROFILING

  // Create the events:
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

  // Execute the FDTD:
  printf(" GPU FDTD loop\n");

#ifdef GPU_PROFILING
  // Enqueue start event:
  checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

  checkCudaErrors ( cudaStreamCreate(&streamCalc) );
  checkCudaErrors ( cudaStreamCreate(&streamCopyH2D) );
  checkCudaErrors ( cudaStreamCreate(&streamCopyD2H) );
  checkCudaErrors ( cudaStreamCreate(&streamMain) );

  // TODO : Ensure that all `cudaMemcpy' sources and destinations
  //         specified correctly:
  // Copy the input to the device input buffer:
  checkCudaErrors ( cudaMemcpy ( ioBuffer,                                 \
                                 (void *)bufferSrc,                        \
                                 /* 1.04.16 - changed source and dest. */  \
                                 chunkSize * sizeof ( FieldComponents_t ), \
                                 cudaMemcpyHostToDevice                    \
                  )            );                                           

  lengthCoeffs = sizeof ( UpdateCoefficients_t );
  if ( lengthCoeffs != sizeof ( updateCoeffs ) )
  {
    fprintf ( stderr, \
              "Error : sizes of static type `UpdateCoefficients_t' and `updateCoeffs' differ." \
            );
    return 255;
  }

  // TODO : I AM HERE (30.09.16) : Implement copy operations for constant \
  //                                coefficients:                          
  // Copy the coefficients to the device coefficient buffer:
  checkCudaErrors ( cudaMemcpyToSymbol ( updateCoeffs,    \
                                         (void *)&coeffs, \
                                         lengthCoeffs     \
                  )                    );                  

  // TODO : Multiply on correct values everywhere!!
  //         sliceSize, chunkSize, blockSize, threadSize. etc.
  argsKPT.bContinue = bContinue;
  argsKPT.deviceWaitWhileLoadingGlobalChunk = \
   deviceWaitWhileLoadingGlobalChunk;          
  argsKPT.deviceGlobalRefreshFlags = deviceGlobalRefreshFlags;
  argsKPT.buffer = ioBuffer;
  argsKPT.global_now = globalClockGPU;

  // TODO : Check the order of parameters:

#ifdef DEBUG_INFO

  argsRMC = new ReloadMemChunkArgs_t                         \
                (                                            \
                  maxChunks, dimSlice, sliceSize, chunkSize, \
                  timesteps, 0,                              \
                  bufferSrc, bufferDst, ioBuffer,            \
                  hostWait4RefreshGlobalSlice,               \
                  debugFlags                                 \
                );                                            

#else

  argsRMC = new ReloadMemChunkArgs_t                         \
                (                                            \
                  maxChunks, dimSlice, sliceSize, chunkSize, \
                  timesteps, 0,                              \
                  bufferSrc, bufferDst, ioBuffer,            \
                  hostWait4RefreshGlobalSlice                \
                );                                            

#endif

  // TODO : I AM HERE (23.11.16) : Continue merging source code:

   // TODO : Remove half-steps:
   // In timeframe we need to advance by half-timestep due to different
   //  steps in H and D and also in E and B fields update equations:
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
