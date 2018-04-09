//
/*************************************************************************/
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*                             testKernelGPU.cu                          */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*************************************************************************/
//


#include <iostream>
#include <algorithm>
#include <pthread.h>

#ifndef _HELPERFUNCTIONS_HELPERCUDA__
#define _HELPERFUNCTIONS_HELPERCUDA__

#include <helper_functions.h>
#include <helper_cuda.h>

#endif

#include "testKernelGPU.h"
#include "testKernelGlobal.cuh"


cudaEvent_t *eventReadWrite;
cudaStream_t streamCopyH2D, streamCopyD2H, streamCalc, streamMain;

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

// TODO : IMPORTANT !!!                       \
// (DONE) Figure out why is there are two arguments: \
//  `maxChunks' and `maxGlobChunks' -         \
//  are they duplicates?                       
// TODO (DONE) : What difference betweeen `sliceSize' and `dimSlice'

// `input' and `output' should never be const parameters - \
//  they are interchanged during time cycles.               
bool testGPU                            \
     (                                  \
       int argc, char **argv,           \
       float *input, float *output,     \
       int timesteps,                   \
       int maxChunks, int chunkSize,    \
       int sliceSize, int dimSlice,     \
       int blockSize,                   \
       int dimx, int dimy, int dimz,    \
       int dimxBlock, int dimyBlock,    \
       int dimThreadsX, int dimThreadsY \
     )                                   
{
  // TODO : I AM HERE - VERIFY SYNCHRONIZATION BETWEEN ALL PARALLEL STREAMS, \
  //         THREADS, PTHREADS.                                               
  // TODO : Move all declarations to the top of the function.

  unsigned char *debugFlags;

  int deviceCount  = 0,
      targetDevice = 0;

  // TODO : What is the function of these two references below? // DONE.
  float *ioBuffer = 0,
        *bufferSrc = input,
        *bufferDst = output;

  clock_t *globalClockGPU;

  dim3 dimBlock,
       dimGrid;

  // Synchronization flags:
  unsigned char *bContinue,/* byte */
                *hostWait4RefreshingChunk_WhileLoadingSlices,/* byte */
                *hostWaitWhileLoadingGlobalChunk,/* byte */
                *hostWait4RefreshGlobalSlice,/* array of bytes */
                *deviceGlobalRefreshFlags,/* array of bytes */
                *deviceWaitWhileLoadingGlobalChunk;/* byte */

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

#ifdef GPU_PROFILING

  cudaEvent_t profileStart = 0;
  cudaEvent_t profileEnd   = 0;
  const int profileTimesteps = timesteps - 1;

  if ( profileTimesteps < 1 )
    printf(" cannot profile with fewer than two timesteps (timesteps=%d), profiling is disabled.\n", timesteps);

#endif

  // Get the number of CUDA enabled GPU devices
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaSetDevice(targetDevice));

#ifdef DEBUG_INFO

  debugFlags = (unsigned char *)(calloc ( dimSlice, 1 ));

#endif

  // Allocate memory buffers:
  checkCudaErrors ( cudaMalloc (                           \
                                 (void **)&ioBuffer,       \
                                 chunkSize * sizeof(float) \
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
  // TODO : Ensure that `0' stands for "not waiting" and also that we need\
  //         "not waiting" state from the begining of program execution in\
  //         `hostWait4RefreshingChunk_WhileLoadingSlices'                \
  //         and                                                          \
  //         `hostWaitWhileLoadingGlobalChunk'                            \
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

  // Set the block size
  dimBlock.x = dimThreadsX;
  dimBlock.y = dimThreadsY;
  dimGrid.x  = dimxBlock;
  dimGrid.y  = dimyBlock;
  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

#ifdef GPU_PROFILING

  // Create the events
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

  // Execute the FDTD
  printf ( " GPU FDTD loop\n" );

#ifdef GPU_PROFILING
  // Enqueue start event
  checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

  checkCudaErrors ( cudaStreamCreate(&streamCalc) );
  checkCudaErrors ( cudaStreamCreate(&streamCopyH2D) );
  checkCudaErrors ( cudaStreamCreate(&streamCopyD2H) );
  checkCudaErrors ( cudaStreamCreate(&streamMain) );

  // TODO : Ensure that all `cudaMemcpy' sources and destinations
  //         specified correctly:
  checkCudaErrors ( cudaMemcpy ( ioBuffer,                           \
                                 (void *)bufferSrc,                  \
                                 /* 1.04.16 - changed src and dst */ \
                                 chunkSize * sizeof ( float ),       \
                                 cudaMemcpyHostToDevice              \
                  )            );                                     

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

  eventReadWrite = (cudaEvent_t *)calloc ( dimSlice,              \
                                           sizeof ( cudaEvent_t ) \
                                         );                        

  for ( int idxSlice = 0; idxSlice < dimSlice; idxSlice++ )
    checkCudaErrors ( cudaEventCreate( &eventReadWrite[idxSlice] ) );

  /* STEP 1 : Launching PThreads. */
  for ( int idxSlice = 0; idxSlice < dimSlice; idxSlice++ )
  {
    if ( pthread_create (                              \
                          &pthreadLoaders[idxSlice],   \
                          NULL,                        \
                          reloadGlobal2SharedMemSlice, \
                          argsRMC                      \
       )                )                               
    {
      fprintf ( stderr, "Error creating pthread" );
      return 255;
      //checkCudaErrors(30);
    }
    if ( idxSlice < dimSlice - 1 )

#ifdef DEBUG_INFO

      argsRMC = new ReloadMemChunkArgs_t                         \
                    (                                            \
                      maxChunks, dimSlice, sliceSize, chunkSize, \
                      timesteps, idxSlice + 1,                   \
                      bufferSrc, bufferDst, ioBuffer,            \
                      hostWait4RefreshGlobalSlice,               \
                      debugFlags                                 \
                    );                                            

#else

      argsRMC = new ReloadMemChunkArgs_t                         \
                    (                                            \
                      maxChunks, dimSlice, sliceSize, chunkSize, \
                      timesteps, idxSlice + 1,                   \
                      bufferSrc, bufferDst, ioBuffer,            \
                      hostWait4RefreshGlobalSlice                \
                    );                                            

#endif

    // EXPLANATION.                                                   \
    // Q : Why not just reassign member variable?                     \
    // A : Because we need different structures for initialization of \
    //      pthreads, and pthread_create do not accept its parameters \
    //      argument-wise.                                             
  }

#ifdef DEBUG_INFO

  for ( int idxSlice = 0; idxSlice < dimSlice; idxSlice++ )
  {
    while ( !debugFlags[idxSlice] ) {}
    printf ( "pthread %d launched\n", idxSlice );
  }

#endif

  // TODO : Verify that ..... WHAT????!!!!                                \
  //         Forget to mention something here, need to figure out what...  
  // Launching the kernel:
  printf("launch control kernel\n");
  /* STEP3 : Launching control stream. */
  testKernelControlStream<<<1, 1, 0, streamMain>>>(argsCKSA);
  printf("launch kernel\n");
  /* STEP5 : Launching calculation stream. */
  testKernel<<<dimGrid,                             \
               dimBlock,                            \
               blockSize * sizeof (float)/* maxSharedMemPerBlock */, \
               streamCalc>>>                        \
            ( argsKPT );                             

  /* STEP7 : Setting continuation flag. */
  *bContinue = 1;
  for ( int it = 0 ; it < timesteps ; it++ )
  {

#ifdef DEBUG_INFO

    printf ( "[DEBUG_INFO]\tt = %d\n\n", it );

#endif

    for ( int idxChunk = 0; idxChunk < maxChunks; idxChunk++ )
    {

#ifdef DEBUG_INFO

      printf ( "[DEBUG_INFO]\tidxChunk = %d\n\n", idxChunk );
      printf ( "[DEBUG_INFO]\t%s\n\n",                    \
               "Waiting until control stream sets `0'..." \
             );                                            

#endif

      /* STEP8 : Waiting while loading slices       */
      /*          to reload whole chunk after that. */
      // TODO : I AM HERE (09.03.16) : FIX AN ERROR with syncronisation      \
      //                                in main loop - steps 2, 8, 9, and 24 \
      //                                conflict with each other             \
      //                                and process hangs:                    
      // Waiting until control stream sets `0':
      while ( *hostWait4RefreshingChunk_WhileLoadingSlices ) {}

#ifdef DEBUG_INFO

      printf ( "[DEBUG_INFO]\t%s%s\n\n",                               \
               "Resetting the value to `1' for the next wait cycle" );  

#endif

      /* STEP20 : Resetting flag to its "wait" state */
      /*           for the next wait cycle.          */
      *hostWait4RefreshingChunk_WhileLoadingSlices = 1;

      // TODO : It looks like the line below is unnecessary!
      /*
      checkCudaErrors                                            \
      (                                                          \
        cudaMemcpyAsync                                          \
        (                                                        \
          ioBuffer,                                              \
          &bufferSrc[idxChunk * chunkSize * sizeof ( float )],   \
          chunkSize * sizeof ( float ),                          \
          cudaMemcpyHostToDevice,                                \
          / * TODO : Verify if this is really main stream!!! * / \
          streamMain                                             \
        )                                                        \
      );                                                          
      */

#ifdef DEBUG_INFO

      printf ( "[DEBUG_INFO]\t%s%s\n\n",                               \
               "Resetting the value to `1' for the next wait cycle" );  

#endif

      /* STEP21 : Sending "continue" signal to kernel threads */
      /*           through control stream.                    */
      if ( idxChunk < maxChunks-1 )
        *hostWaitWhileLoadingGlobalChunk = 0;
    }

    // TODO (DONE) : Implement blocking operation during below three lines:
    // Toggle the buffers                           \
    //  Visual Studio 2005 does not like std::swap  \
    //  `std::swap<float *>(bufferSrc, bufferDst);'  
    float *tmp = bufferDst;
    bufferDst = bufferSrc;
    bufferSrc = tmp;

    /* STEP21 : Sending "continue" signal to threads through control stream. */
    // TODO (DONE) : Fix possible error (same line as in above loop):
    if ( it < timesteps-1 )
      *hostWaitWhileLoadingGlobalChunk = 0;
  }
  // TODO (DONE) : Reset all flags properly here, before setting `bContinue' \
  //                to zero to allow all pthreads, kernel threads and        \
  //                streams catch termination signal; execution loop should  \
  //                be at steps 21-23 at this execution point.                
  *bContinue = 0;
  *hostWaitWhileLoadingGlobalChunk = 0;

  //pthread_join ( pthreadKernel, NULL );

  for ( int idxSlice = 0; idxSlice < dimSlice; idxSlice++ )
  {
    if ( pthread_join ( pthreadLoaders[idxSlice], NULL ) )
    {
      fprintf(stderr, "Error joining pthread");
      return 255;
      //checkCudaErrors(1);
    }
  }

  printf("\n");

#ifdef GPU_PROFILING
  // Enqueue end event
  checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());

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
        size_t pointsComputed = dimx * dimy * dimz;// TODO (DONE) : dimz, dimy, dimx is equal to 16??? dix = dimy = 16 and dimz = 128.
        // Determine throughput
        double throughputM    = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        printf("FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, NumDevsUsed = %u, Blocksize = %u\n",
               throughputM, avgElapsedTime, pointsComputed, 1, dimBlock.x * dimBlock.y);
    }

#endif

    // Cleanup
    if (ioBuffer)
    {
        checkCudaErrors(cudaFree(ioBuffer));
    }

for (int idxSlice = 0; idxSlice < dimSlice; idxSlice++)
  checkCudaErrors ( cudaEventDestroy ( eventReadWrite[idxSlice] ) );

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
