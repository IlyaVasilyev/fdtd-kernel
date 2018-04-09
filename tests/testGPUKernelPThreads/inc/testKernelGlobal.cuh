//# vi:syntax=cuda

#ifndef _TESTKERNELGLOBAL_CUH_
#define _TESTKERNELGLOBAL_CUH_


// TODO : Move all consts to const memory.              \
//         CRITICAL WITH FDTD IMPLEMENTED INSIDE KERNEL  

// TODO (DONE) : CRITICAL : FIX ANOTHER ERROR!!!            \
//                           Shared memory is different for \
//                           2 different streams!!!          
// TODO (DONE) : Threads, pthreads, streams, or processes \
//                hangs.                                   
extern __shared__ char memPack []; /* Test: 16 * 16 * 24 bytes. */

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

__global__ void testKernel ( TestKernelArguments_t args )
{
  /*
  extern __shared__ float tile [];
  */
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
  // TODO (IMPORTANT and INTERESTING) (DONE) : Figure out why reference types \
  //                                           behaves as register pointers   \
  //                                           in CUDA: all function          \
  //                                           call parameters do not saved   \
  //                                           in device memory.               
  const int /*&dimx          = args.dimx,*/
            /* `dimx' is whole XxY blocks global memory slices !!! */
            /*  e.g. slice No 1, slice No 2, ... , slice No N.     */
            /*&dimy          = args.dimy,*/
            /* dimy in whole XxY blocks global memory slices !!! */
            /*&*/dimz        = args.dimz,
            /* `dimz' is a thread memory length in z direction !!! */
            /*&*/dimxBlock   = args.dimxBlock,
            /*&*/dimyBlock   = args.dimyBlock,
            /* `dimxBlock' and `dimyBlock' is a length of grid in blocks */
            /*  alongside x and y direction respectively.                */
            /*&*/dimzBlock   = args.dimSlice,
            /* `dimzBlock' is a number of slices per chunk. */
            /*&*/dimThreadsX = args.dimThreadsX,
            /*&*/dimThreadsY = args.dimThreadsY;

  int k, iz, iiz, idx_io, idx_sync, idx_shared;

  // TODO : ALIGN CODE LINES!!!
  float  fResult,
        *ioBuffer = args.buffer,
        /* ioBuffer[dimSlice][(*dimxBlock)][(*dimyBlock)] */
        /*         [dimThreadsX][dimThreadsY]             */
        *ioTile   = (float *)memPack;
        /* ioTile[dimThreadsX][dimThreadsY] */

  clock_t    start,
             now,
             cycles,
          * &global_now = args.global_now;

  /* STEP6 : Waiting when main process sets continuation flag. */
  // TODO (DONE) : FIX CRITICAL ERROR : Loop does not count \
  //                                     timesteps:          
  // TODO (DONE) : FIX AN ERROR : Implement an empty loop     \
  //                               that waits for `bContinue' \
  //                               flag:                       
  // TODO (DONE) : Use loop with timesteps instead of bContinue flag:
  // TODO (DONE) : Put here global loop over all global memory CHUNCKS \
  //                (index is irrelevant - host responsible for        \
  //                proper memory loading and unloading):               
  while ( !*(args.bContinue) ) {}
  for ( int timestep = 0; timestep < args.timesteps; timestep++ )
  {
    for ( k = 0; k < args.maxChunks; k++ )
    {
      // Per-block loop alongside z direction:
//#pragma unroll 3
      for ( iz = 0; iz < dimzBlock; iz++ )
      {
        // ioBuffer size is equal to                                    \
        //  dimSlice{4}*gridDim{2}*gridDim{2}*blockDim{16}*blockDim{16} \
        //  *threadSize{128}.                                            
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

        memcpy                                                   \
        (                                                        \
          (float *)&ioTile[dimThreadsY*dimz*ltidx + dimz*ltidy], \
          (float *)&ioBuffer[idx_io],                            \
          dimz * sizeof(float)                                   \
        );                                                        

        /* STEP10 : Wait until all threads comes to this line. */
        __syncthreads();

        /* TODO (DONE) : ERROR: Out-fo-range exception */
        /*                       in cuda kernel!       */
//#pragma unroll 3
        for ( iiz = 0; iiz < dimz - 1/* Optimize !! */; iiz++ )
        {
          idx_shared = dimThreadsY*dimz*ltidx \
                       + dimz*ltidy           \
                       + iiz;                  
          fResult = ioTile[idx_shared]        \
                    + ioTile[idx_shared + 1];  

          start = clock ();
          for (;;)
          {
            now = clock();
            cycles = now > start ? now - start : now + (0xffffffff - start);
            // TODO : I AM HERE (07.30.16) : Ensure that `cycles' don't cause \
            //                                hanging.                         
            if ( cycles >= 10000 )
              break;
          }
          *global_now = now;

          fResult *= ioTile[idx_shared]        \
                     - ioTile[idx_shared + 1];  

          /* STEP11 : Wait until all threads comes to this line. */
          __syncthreads();

          ioTile[idx_shared] = fResult;

          /* STEP12 : Wait until all threads comes to this line. */
          __syncthreads();
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

#endif
