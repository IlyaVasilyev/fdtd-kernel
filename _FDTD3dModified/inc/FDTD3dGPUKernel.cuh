//# vi:syntax=cuda
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

#ifndef _FDTD3DGPUKERNEL_CUH_
#ifndef _FDTD3DGPUKERNEL_CUH_

#include "FDTD3dGPU.h"

// Note: If you change the RADIUS or VOLUME, you should also change the unrolling below:
#define VOLUME VOLUME_SHARED
#define RADIUS RADIUS_SHARED

// TODO : Implement `cudaMemcpyToSymbol' for all constants:
__constant__ UpdateCoefficients_t updateCoeffs;
// TODO : Implement constant types and variables:
__constant__ int T_src;
__constant__ float omega_src;
__constant__ float n_inc;
__constant__ float k_src;
__constant__ float musqrt_src;
__constant__ float epsilonsqrt_src;

__constant__ int blockDimXY;
// Both "slice" and "layer" X and Y dimensions are equal \
//  (not to be confused with "tile" dimensions) :         
__constant__ int dimx;
__constant__ int dimy;
__constant__ int outerDimx;
__constant__ int outerDimy;

// TODO : Move all consts to const memory.              \
//         CRITICAL WITH FDTD IMPLEMENTED INSIDE KERNEL~~
// TODO (DONE) : CRITICAL : FIX ANOTHER ERROR!!!            \
//                           Shared memory is different for \
//                           2 different streams!!!~~~~~~~~~~
// TODO (DONE) : Threads, pthreads, streams, or processes \
//                hangs.~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
extern __shared__ char memPack []; /* Test: 16 * 16 * 24 bytes. */
/*
extern __shared__ float tile [];
*/

// TODO : Ensure that control stream do not wait for threads while they \
//         are reach chunk end and only waits for slices!!! Send        \
//         signals for slice reloading after each slice finished. Also  \
//         ensure that threads will wait after finishing with chunk     \
//         until new chunk is copletelly loaded by per-slice pthreads.   
__global__ void testKernelControlStream                   \
                (                                         \
                  const TestControlKernelArguments_t args \
                )                                          
{
  const int Xthreads = args.dimThreadsX,
            Ythreads = args.dimThreadsY,
            dimzBlock = args.dimSlice,
            dimyBlock = args.dimyBlock,
            dimxBlock = args.dimxBlock;

  // TODO : You need to use multiplications while indexxing elements.    \
  //         Otherwise you may want to use devceQuery in Makefile before \
  //         compiling code to link it with proper static header with    \
  //         particular GPU architecture description. There is no other  \
  //         way to optimize dynamic array indexing.                      
  // TODO : Comment each flag that people understand what flags are set \
  //         for.                                                        
  unsigned char *deviceWaitWhileLoadingGlobalChunk =      \
                  args.deviceWaitWhileLoadingGlobalChunk,  
                *deviceGlobalRefreshFlags = args.deviceGlobalRefreshFlags,
                *hostWaitWhileLoadingGlobalChunk =      \
                  args.hostWaitWhileLoadingGlobalChunk,  
                *hostWait4RefreshGlobalSlice =      \
                  args.hostWait4RefreshGlobalSlice,  
                *hostWait4RefreshingChunk_WhileLoadingSlices =      \
                  args.hostWait4RefreshingChunk_WhileLoadingSlices;  

  // Peviously there are was a rule that initial values of `i' and `j' \
  //  should be `MAXSHORTINT - 1 - (dimThreadsX | dimThreadsY)';       \
  //  but now it's OBSOLETE.                                            
  int i, j, k, l, m, n, iGRFdevice, idx_io;

  /* STEP4 : Waiting while main process sets continuation flag. */
  // TODO (DONE) : FIX CRITICAL ERROR : Loop does not count \
  //                                     timesteps:          
  // TODO I AM HERE (09.03.16) : FIX AN ERROR : Implement an empty loop     \
  //                                             that waits for `bContinue' \
  //                                             flag:                       
  // TODO : Use loop with timesteps instead of bContinue flag:
  while ( !*(args.bContinue) ) {}
  for ( int timestep = 0; timestep < args.timesteps; timestep++ )
  {
    for ( k = 0; k < args.maxChunks; k++ )
    {
      for ( l = 0; l < dimzBlock; l++ )
      {
        for ( i = 0; i < dimxBlock; i+=iGRFdevice )
          for ( j = 0; j < dimyBlock; j+=iGRFdevice )
            do
            {
              iGRFdevice = 1;
              for ( m = 0; m < Xthreads; m++ )
              {
                for ( n = 0; n < Ythreads; n++ )
                {
                  idx_io = l * dimxBlock * dimyBlock             \
                           * Xthreads * Ythreads                 \
                           + i * dimyBlock * Xthreads * Ythreads \
                           + j * Xthreads * Ythreads             \
                           + m * Ythreads                        \
                           + n;                                   

                  /* STEP9 : Wait until all threads in all blocks sets flags */
                  /*          indicating that next slice can be reloaded     */
                  /*          from host memory to device memory.             */
                  iGRFdevice *= deviceGlobalRefreshFlags[idx_io];
                  // TODO : ` i = i & ( (~f ^ f) | f ); '
                }
              }
            }
            while ( !iGRFdevice );

        /* STEP14 : Control stream sets flag telling main process that it   */
        /*           can copy slice from the device memory to host and load */
        /*           new data into device memory.                           */
        // `iz' or `l' - is a XxY blocks slice index in global memory:
        hostWait4RefreshGlobalSlice[l] = 0;
      }

      /* STEP18 : Waiting until PThreads reload all slices from device to */
      /*           host memory and load new slices to device.             */
      // TODO : This is much more optimal than overflow technique,   \
                 because it has lesser number of conditional checks:  
      // TODO I AM HERE (07.30.16) : Test the loop below:
      l = 0;
      while ( l < dimzBlock )
        l += hostWait4RefreshGlobalSlice[l];

      /* STEP19 : Telling host process that global chunk can be reloaded, */
      /*           evaluated, updated, etc.                               */
      *hostWait4RefreshingChunk_WhileLoadingSlices = 0;

      // TODO : Do I need to block acces to volatile host-device memory \
      //         section?? ANSWER : AT LEAST YOU NEED VOLATILE          \
      //         MODIFICATOR:                                            
      /* STEP21 : Waiting until global chunk is reloaded. */
      while ( *hostWaitWhileLoadingGlobalChunk ) {}

      /* STEP22 : Resetting flag to its "wait" state. */
      *hostWaitWhileLoadingGlobalChunk = 1;

      /* STEP23 : Telling kernel threads to continue calculations. */
      *deviceWaitWhileLoadingGlobalChunk = 0;

      /* STEP28 : Wait for first kernel thread reached the line telling   */
      /*           others that all flags, indicating that specific global */
      /*           memory part needs to be updated in shared memory, have */
      /*           been reset for all particular blocks and threads.      */
      while ( !(*deviceWaitWhileLoadingGlobalChunk) ) {}
    }
  }
}

__global__ void FiniteDifferencesKernelWithSource          \
                (                                          \
                  FDTDKernelWithSourceArguments_t args     \
                  /* TODO : Pass sinsrc with input data */ \
                )                                           
{
    bool validr = true;
    bool validw = true;
    // TODO : Remove extra `const':
    // TODO : Optimize!!!
    const int /*gtidx = blockIdx.x * blockDim.x + threadIdx.x,*/
              /*gtidy = blockIdx.y * blockDim.y + threadIdx.y,*/
              ltidx = threadIdx.x,
              ltidy = threadIdx.y,
              blkx  = blockIdx.x,
              blky  = blockIdx.y;

    // TODO : Does it correct to get values by reference from struct \
    //         that passed as a parameter to cuda kernel?             
    unsigned char *deviceWaitWhileLoadingGlobalChunk =      \
                    args.deviceWaitWhileLoadingGlobalChunk,  
                  *deviceGlobalRefreshFlags = args.deviceGlobalRefreshFlags;
    // TODO (IMPORTANT and INTERESTING) : Figure out why reference types \
    //                                     behaves as register pointers  \
    //                                     in CUDA:                       
    const int /*&dimx          = args.dimx,*/
              /* `dimx' is whole XxY blocks global memory slices !!! */
              /*  e.g. slice No 1, slice No 2, ... , slice No N.     */
              /*&dimy          = args.dimy,*/
              /* dimy in whole XxY blocks global memory slices !!! */
              /*&*/dimz        = args.dimz,
              /* `dimz' is a thread memory length in z direction !!! */
              /* TODO : `dimz' in working kernel should be 3 times lesser */
              /*         than normal.                                     */
              /*&*/dimxBlock   = args.dimxBlock,
              /*&*/dimyBlock   = args.dimyBlock,
              /* `dimxBlock' and `dimyBlock' is a length of grid in blocks */
              /*  alongside x and y direction respectively.                */
              /*&*/dimzBlock   = args.dimSlice,
              /* `dimzBlock' is a number of slices per chunk. */
              /*&*/dimThreadsX = args.dimThreadsX,
              /*&*/dimThreadsY = args.dimThreadsY;

    int k, iz, iiz, idx_io, idx_sync, idx_shared,
        offset_I            = dimz * sizeof(FieldComponents_t),
        /* TODO : Test integer type rounding in media and volume dimensions: */
        /* In a timeframe we need to advance by half-timestep due to        */
        /*  different steps in update equations associated with H and D and */
        /*  also E and B fields:                                            */
        dimXYHalfNearVolume = ( dimy - VOLUME_SHARED ) / 2,
        dimXYHalfAfterVolume = dimy - dimXYHalfNearVolume,
        dimXYPostPML = RADIUS_SHARED,
        dimXYPrePML = outerDimy - RADIUS_SHARED,
        dimZPrePML = outerDimz - RADIUS_SHARED;


    // TODO : ALIGN CODE LINES!!!
    float  fResult, ioItem, Curl,
          *ioBuffer = args.buffer,
          /* ioBuffer[dimSlice][(*dimxBlock)][(*dimyBlock)] */
          /*         [dimThreadsX][dimThreadsY]             */
          /* TODO : Implement `buffer_IH' and `buffer_ID'  */
          /*         in `FDTDKernelWithSourceArguments_t': */
          *ioBuffer_IH = args.buffer_IH,
          *ioBuffer_ID = args.buffer_ID,
          /* TODO : Reevaluate `memPack' size when passing it in kernel: */
          *ioTile = (float *)(memPack + 2*offset_I),
          *ioTile_IH = (float *)memPack,
          *ioTile_ID = (float *)(memPack + offset_I);
          /* ioTile[dimThreadsX][dimThreadsY] */

    // TODO : Move `fdtdTFSFsrc...' to shared memory, because source signal \
    //         2D tile changes in every chunk in it's first slice, and      \
    //         equal to zero in other slices:                                
    TFSF_t fdtdTFSFsrcE,
           fdtdTFSFsrcH;

  /* STEP6 : Waiting when main process sets continuation flag. */
  // TODO (DONE) : Put here global loop over all global memory CHUNCKS \
  //                (index is irrelevant - host responsible for        \
  //                proper memory loading and unloading):               
  while ( !*(args.bContinue) ) {}
  for ( int timestep = 0; timestep < args.timesteps; timestep++ )
  {
    for ( k = 0; k < args.maxChunks; k++ )
    {
      // Per-block loop alongside z direction:

      // Additional loop cycle launced to evaluate z-axis PML borders and \
      //  TF-SF layers before handling the main gomogeneous bulk:          
      iz = 0;
      // TODO : Implement first loop cycle here by wrapping loop body code in \
      //         inline function and by calling it here.                       

      // The last loop cycle should be implemented \
      //  after the main bulk calculations:         
//#pragma unroll 3
      for ( iz = 1; iz < (dimzBlock-1); iz++ )
      {
        // ioBuffer size is equal to                               \
        //  dimSlice*gridDim*gridDim*blockDim*blockDim*threadSize:  
        idx_io = dimxBlock*dimyBlock*dimThreadsX*dimThreadsY*dimz*iz \
                 + dimyBlock*dimThreadsX*dimThreadsY*dimz*blkx       \
                 + dimThreadsX*dimThreadsY*dimz*blky                 \
                 + dimThreadsY*dimz*ltidx                            \
                 + dimz*ltidy;                                        
        idx_sync = dimxBlock*dimyBlock*dimThreadsX*dimThreadsY*iz \
                   + dimyBlock*dimThreadsX*dimThreadsY*blkx       \
                   + dimThreadsX*dimThreadsY*blky                 \
                   + dimThreadsY*ltidx                            \
                   + ltidy;                                        

        memcpy                                                               \
        (                                                                    \
          (FieldComponents_t *)&ioTile[dimThreadsY*dimz*ltidx + dimz*ltidy], \
          (FieldComponents_t *)&ioBuffer[idx_io],                            \
          dimz * sizeof(FieldComponents_t)                                   \
        );                                                                    
        memcpy                                                                  \
        (                                                                       \
          (FieldComponents_t *)&ioTile_IH[dimThreadsY*dimz*ltidx + dimz*ltidy], \
          (FieldComponents_t *)&ioBuffer_IH[idx_io],                            \
          dimz * sizeof(FieldComponents_t)                                      \
        );                                                                       
        memcpy                                                                  \
        (                                                                       \
          (FieldComponents_t *)&ioTile_ID[dimThreadsY*dimz*ltidx + dimz*ltidy], \
          (FieldComponents_t *)&ioBuffer_ID[idx_io],                            \
          dimz * sizeof(FieldComponents_t)                                      \
        );                                                                       

        /* STEP10 : Wait until all threads comes to this line. */
        __syncthreads();

        // TODO : Implement fields `TFSFsrc.E' and `TFSFsrc.H' \
        //         in `FDTDKernelWithSourceArguments_t':        
        // TODO : Change `ix' / `iy' to specific coordinates associated   \
        //         with threads and blocks (both should be `idx_shared'):  
        // TODO : I AM HERE (18.09.16) : I am still here until TF-SF       \
        //                                corrections will be moved out of \
        //                                main loop to first cycle:         
        // Step through xy-planes:
        // Inside PML pseudoboundaries:
        // TF-SF corrections:

        fdtdTFSFsrcE.sinsrc = args.TFSFsrc.E;
        fdtdTFSFsrcE.getTFSF = &getFDTDTFSFsrcE;

        fdtdTFSFsrcH.sinsrc = args.TFSFsrc.H;
        fdtdTFSFsrcH.getTFSF = &getFDTDTFSFsrcH;

        // TODO : Implement similar approach for passing arguments `ioTile', \
        //         `ioTile_ID', and `ioTile_IH': both with derefencing       \
        //         or both without it:                                        
        // TODO : Rename `ioItem' and also `float &F' in wrapped functions \
        //         into `fA':                                               
        fdtdRef4SingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume,  \
                           dimXYHalfNearVolume, dimXYHalfAfterVolume,  \
                           dimx, dimy, ltidx, ltidy, *ioTile, *ioTile, \
                           updateCoeffs, ioTile_IH, ioTile_ID,         \
                           Curl, ioItem,                               \
                           RADIUS_SHARED, 0,                           \
                           fdtdRefFieldWindowMediaHWrapper,            \
                           fdtdRefFieldWindowMediaDWrapper,            \
                           fdtdTFSFsrcE, 0                             \
                         );                                             

        fdtdRefSingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                          dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                          ltidx, ltidy, *ioTile, *ioTile,            \
                          updateCoeffs, ioTile_IH, ioTile_ID,        \
                          Curl, ioItem,                              \
                          RADIUS_SHARED, 0, 0,                       \
                          fdtdRefFieldWindowVolume,                  \
                          fdtdRefFieldWindowMediaDWrapper,           \
                          fdtdTFSFsrcE, 0                            \
                        );                                            

        fdtdRef4SingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume,  \
                           dimXYHalfNearVolume, dimXYHalfAfterVolume,  \
                           dimx, dimy, ltidx, ltidy, *ioTile, *ioTile, \
                           updateCoeffs, ioTile_IH, ioTile_ID,         \
                           Curl, ioItem,                               \
                           RADIUS_SHARED + 1, 0,                       \
                           fdtdRefFieldWindowMediaHWrapper,            \
                           fdtdRefFieldWindowMediaDWrapper,            \
                           0, fdtdTFSFsrcH                             \
                         );                                             

        fdtdRefSingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                          dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                          ltidx, ltidy, *ioTile, *ioTile,            \
                          updateCoeffs, ioTile_IH, ioTile_ID,        \
                          Curl, ioItem,                              \
                          RADIUS_SHARED + 1, 0, 0,                   \
                          fdtdRefFieldWindowVolume,                  \
                          fdtdRefFieldWindowMediaDWrapper,           \
                          0, fdtdTFSFsrcH                            \
                        );                                            

        // Field around sources (PML):

        for ( int iiz = RADIUS_SHARED ; iiz < RADIUS_SHARED + 2; iiz++ )
        {
          // TODO : I AM HERE (18.09.16) : Fix an error described below:
          // TODO : FIX AN ERROR : Parameters mismatch:
          // TODO : Ensure that block coordinates        \
          //         blkidx and blkidy are not required:  
          // TODO : Figure out why is there a dereferencing `*':
          fdtdRefSingleXY ( 0, dimXYPostPML, 0, outerDimy,                \
                            ltidx, ltidy, *ioTile, *ioTile, updateCoeffs, \
                            ioTile_IH, ioTile_ID,                         \
                            Curl, ioItem,                                 \
                            iiz, ltidx, 0,                                \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               

          // TODO : Check if `dimx' is correct parameter:
          fdtdRefSingleXY ( dimXYPrePML, outerDimx, 0, outerDimy,         \
                            ltidx, ltidy, *ioTile, *ioTile, updateCoeffs, \
                            ioTile_IH, ioTile_ID,                         \
                            Curl, ioItem,                                 \
                            iiz, ix, dimx,                                \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               

          fdtdRefSingleXY ( dimXYPostPML, dimXYPrePML, 0, dimXYPostPML,   \
                            ltidx, ltidy, *ioTile, *ioTile, updateCoeffs, \
                            ioTile_IH, ioTile_ID,                         \
                            Curl, ioItem,                                 \
                            iiz, ltidy, 0,                                \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               

          fdtdRefSingleXY ( dimXYPostPML, dimXYPrePML,                    \
                            dimXYPrePML, outerDimy,                       \
                            ltidx, ltidy, *ioTile, *ioTile, updateCoeffs, \
                            ioTile_IH, ioTile_ID,                         \
                            Curl, ioItem,                                 \
                            iiz, ltidy, dimy,                             \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               
        }

        for (int iiz = RADIUS_SHARED + 2 ; iiz < dimZPrePML ; iiz++)
        {
          fdtdRef4SingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                             dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                             dimx, dimy,                                \
                             ltidx, ltidy, *ioTile, *ioTile,            \
                             updateCoeffs,                              \
                             ioTile_IH, ioTile_ID,                      \
                             Curl, ioItem, iiz, 0,                      \
                             fdtdRefFieldWindowMediaHWrapper,           \
                             fdtdRefFieldWindowMediaDWrapper,           \
                             &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull   \
                           );                                            

          fdtdRefSingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                            dimXYHalfNearVolume, dimXYHalfAfterVolume, \
                            ltidx, ltidy, *ioTile, *ioTile,            \
                            updateCoeffs,                              \
                            ioTile_IH, ioTile_ID,                      \
                            Curl, ioItem,                              \
                            iiz, 0, 0,                                 \
                            fdtdRefFieldWindowVolume,                  \
                            fdtdRefFieldWindowMediaDWrapper,           \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull   \
                          );                                            

          fdtdRefSingleXY ( 0, dimXYPostPML, 0, outerDimy,                \
                            ltidx, ltidy, *ioTile, *ioTile,               \
                            updateCoeffs, ioTile_IH, ioTile_ID,           \
                            Curl, ioItem, iiz, ltidx, 0,                  \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               

          fdtdRefSingleXY ( dimXYPrePML, outerDimx, 0, outerDimy,         \
                            ltidx, ltidy, *ioTile, *ioTile,               \
                            updateCoeffs, ioTile_IH, ioTile_ID,           \
                            Curl, ioItem, iiz, ltidx, dimx,               \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               

          fdtdRefSingleXY ( dimXYPostPML, dimXYPrePML, 0, dimXYPostPML,   \
                            ltidx, ltidy, *ioTile, *ioTile,               \
                            updateCoeffs, ioTile_IH, ioTile_ID,           \
                            Curl, ioItem, iiz, ltidy, 0,                  \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               

          fdtdRefSingleXY ( dimXYPostPMLu, dimXYPrePML,                   \
                            dimXYPrePML, outerDimy,                       \
                            ltidx, ltidy, *ioTile, *ioTile,               \
                            updateCoeffs, ioTile_IH, ioTile_ID,           \
                            Curl, ioItem, iiz, ltidy, dimy,               \
                            fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                          );                                               
        }

        for (int iiz = 0 ; iiz < RADIUS_SHARED ; iiz++)
        {
          fdtdRef4SingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume,    \
                             dimXYHalfNearVolume, dimXYHalfAfterVolume,    \
                             dimx, dimy, ltidx, ltidy, *ioTile, *ioTile,   \
                             updateCoeffs, ioTile_IH, ioTile_ID,           \
                             Curl, ioItem, iiz, iiz,                       \
                             fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, \
                             &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull      \
                           );                                               

          fdtdRef4SingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume,       \
                             dimXYHalfNearVolume, dimXYHalfAfterVolume,       \
                             dimx, dimy, ltidx, ltidy, *ioTile, *ioTile,      \
                             updateCoeffs, ioTile_IH, ioTile_ID,              \
                             Curl, ioItem,                                    \
                             RADIUS_SHARED + dimz + iiz, RADIUS_SHARED - iiz, \
                             fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD,    \
                             &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull         \
                           );                                                  

          fdtdRefSingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume,     \
                            dimXYHalfNearVolume, dimXYHalfAfterVolume,     \
                            ltidx, ltidy, *ioTile, *ioTile, updateCoeffs,  \
                            ioTile_IH, ioTile_ID, Curl, ioItem,            \
                            iiz, RADIUS_SHARED - iiz, 0,                   \
                            fdtdRefFieldPMLVolumeH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull       \
                          );                                                
          // `RADIUS_SHARED - iiz' - that may cause an error when passing   \
          //  an argument in subrouting by reference and not by it's value. \
          //  It is need `min' and `max' values and not only `min' value.    
          fdtdRefSingleXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume,     \
                            dimXYHalfNearVolume, dimXYHalfAfterVolume,     \
                            ltidx, ltidy, *ioTile, *ioTile, updateCoeffs,  \
                            ioTile_IH, ioTile_ID, Curl, ioItem,            \
                            iiz + dimz + RADIUS_SHARED, iiz, 0,            \
                            fdtdRefFieldPMLVolumeH, fdtdRefFieldPMLMediaD, \
                            &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull       \
                          );                                                
        }

        memcpy ( (float *)&ioBuffer[idx_io],                    \
                 (float *)&ioTile[ltidx * dimThreadsY + ltidy], \
                 dimzBlock * sizeof(float)                      \
               );                                                

        /* STEP13 : Thread sets its flag indicating that it completes */
        /*           evaluating its part of current slice.            */
        // TODO : I AM HERE (09.03.16) : FIX AN ERROR : All processes block   \
        //                                               each other and hang:  
        // TODO (DONE) : FIX AN ERROR!!! Flags have to include both \
        //                                xy-block and z indexes:    
        // TODO (DONE) : Find out why there are only thread indices but no \
        //                block's ones:                                     
        // TODO (DONE) : We need control stream for synchronization with host:
        deviceGlobalRefreshFlags[idx_sync] = 1;
        // 1 - ready, 0 - wait.
      }

      __syncthreads();

      /* STEP24 : Waiting for new slices. */
      // TODO : How to reset???
      while ( *deviceWaitWhileLoadingGlobalChunk ) {}

      /* STEP25 : Resetting flag to its "wait" state. */
      // TODO : Verify if this reset works properly:
      for ( iz = 0 ; iz < dimzBlock ; iz++ )
      {
        idx_sync = dimxBlock*dimyBlock*dimThreadsX*dimThreadsY*iz \
                   + dimyBlock*dimThreadsX*dimThreadsY*blkx       \
                   + dimThreadsX*dimThreadsY*blky                 \
                   + dimThreadsY*ltidx                            \
                   + ltidy;                                        
        deviceGlobalRefreshFlags[idx_sync] = 0;  // 1 - ready, 0 - wait.
      }

      /* STEP26 : Wait until all threads comes to this line. */
      __syncthreads();//???? TODO : remove??? __syncthreads()

      /* STEP27 : Resetting flag to its "wait" state. */
      *deviceWaitWhileLoadingGlobalChunk = __any(1);

      //Stop and wait here in GLOBAL loop for new CHUNK \
      // (synchronize with host)                         
    }
  }
}


