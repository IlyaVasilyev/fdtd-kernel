#include <iostream>
#include <iomanip>
#ifndef _HELPERFUNCTIONS_HELPERCUDA__
#define _HELPERFUNCTIONS_HELPERCUDA__

#include <helper_functions.h>
#include <helper_cuda.h>

#endif
#include <math.h>
#include <assert.h>

#include "testKernelGPU.h"
#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif


// Forward declarations
bool runTest(int argc, char **argv);

int main(int argc, char **argv)
{
    bool bTestResult = false;
    // Start the log
    printf("%s Starting...\n\n", argv[0]);

    // Execute
    bTestResult = runTest(argc, argv);

    // Finish
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void generateRandomData(float *data, const float lowerBound, const float upperBound, int maxChunks, int chunkSize)
{
    srand(0);

    for (int i = 0 ; i < maxChunks * chunkSize; i++)
        data[i] = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
}

// TODO : I AM HERE. VERIFY THAT ALL SYNCRONISATIONS WORK !!! WRITE MAKEFILE FOR COMPILATION WITH PTHREADS AND PROFILER. ENSURE THAT ALL MEMORY ASSIGNMENTS HAVE CORRECT TYPES. TYPES MUST SATISFY THE SIZE OF A PARTICULAR MEMORY BLOCK, CHUNK, SLICE, ETC.
bool runTest(int argc, char **argv)
{
  /* Computational units: */
  const int blockDim = 2,
            gridDim = 2,
            maxThreads = blockDim * blockDim,// Threads per block.
            maxBlocks = gridDim * gridDim,
            // Threads per grid. Overall 2*2*16*16=1024 number of \
            //  resident threads per multiprocessor                
  /* Memory units: */
            threadSize = 24,
            dimx = blockDim,
            dimy = blockDim,
            dimz = threadSize,
            blockSize = maxThreads * threadSize,
            // `threadSize' - thread length in z direction.
            // 32768 bytes of shared memory per thread block
            dimSlice = 4,
            sliceSize = maxBlocks * blockSize,
            maxChunks = 4,
            chunkSize = dimSlice * sliceSize;

  float *output;
  float *input;

  void *tmpGenRef;

  int outerDimx;
  int outerDimy;
  int outerDimz;
  int timesteps;

  const float lowerBound = 0.0f;
  const float upperBound = 1.0f;

  timesteps = 10;

  // Determine volume size
  outerDimx = maxChunks * chunkSize;
  outerDimy = maxChunks * chunkSize;
  outerDimz = maxChunks * chunkSize;
  long volumeSize = outerDimx * outerDimy * outerDimz;

  // Allocate memory
  input = (float *)malloc(maxChunks * chunkSize * sizeof(float));

  // Generate data
  printf(" generateRandomData\n\n");
  generateRandomData ( input, lowerBound, upperBound, maxChunks, chunkSize );// TODO : Find generateRandomData.
  printf("FDTD on %d x %d x %d volume for %d timesteps...\n\n", outerDimx, outerDimy, outerDimz, timesteps);

  // Allocate memory
  output = (float *)(calloc ( maxChunks * chunkSize, sizeof(float) ));

  // Execute on the device
  printf ( "fdtdGPU...\n" );
  // TODO (DONE) : Figure out what is the parameters int dimxBlock, int dimyBlock, int dimzBlock, int dimThreadsX, int dimThreadsY
  // TODO (DONE) : Error in ORDER OF parameters!!!
  testGPU ( argc, argv, input, output, timesteps, maxChunks, chunkSize, sliceSize, dimSlice, blockSize, dimx, dimy, dimz, gridDim, gridDim, blockDim, blockDim );

  printf ( "fdtdGPU complete\n" );
}

// TODO : Double check all parameters due to changes in testGPU

#ifdef DEBUG_INFO

structReloadMemChunkArgs::structReloadMemChunkArgs                      \
                          (                                             \
                            const int maxChunks,                        \
                            const int dimSlice,                         \
                            const int sliceSize,                        \
                            const int chunkSize,                        \
                            const int timesteps,                        \
                            int idxSlice,                               \
                            const float* bufferSrc,                     \
                            float* bufferDst,                           \
                            float* ioBuffer,                            \
                            unsigned char *hostWait4RefreshGlobalSlice, \
                            unsigned char *debugFlags                   \
                          ) :                                            
  maxChunks (maxChunks), dimSlice (dimSlice),
  sliceSize (sliceSize), chunkSize (chunkSize),
  timesteps (timesteps),
  bufferSrc (bufferSrc)
{
  this->idxSlice = idxSlice;
  this->bufferDst = bufferDst;
  this->ioBuffer = ioBuffer;
  this->hostWait4RefreshGlobalSlice = hostWait4RefreshGlobalSlice;
  this->debugFlags = debugFlags;
}

#else

structReloadMemChunkArgs::structReloadMemChunkArgs                     \
                          (                                            \
                            const int maxChunks,                       \
                            const int dimSlice,                        \
                            const int sliceSize,                       \
                            const int chunkSize,                       \
                            const int timesteps,                       \
                            int idxSlice,                              \
                            const float* bufferSrc,                    \
                            float* bufferDst,                          \
                            float* ioBuffer,                           \
                            unsigned char *hostWait4RefreshGlobalSlice \
                          ) :                                           
  maxChunks (maxChunks), dimSlice (dimSlice),
  sliceSize (sliceSize), chunkSize (chunkSize),
  timesteps (timesteps),
  bufferSrc (bufferSrc)
{
  this->idxSlice = idxSlice;
  this->bufferDst = bufferDst;
  this->ioBuffer = ioBuffer;
  this->hostWait4RefreshGlobalSlice = hostWait4RefreshGlobalSlice;
}

#endif

structSharedKernelArgumentsBase::structSharedKernelArgumentsBase \
  (                                                              \
    unsigned char *bContinue,                                    \
    unsigned char *deviceWaitWhileLoadingGlobalChunk,            \
    unsigned char *deviceGlobalRefreshFlags,                     \
    const int dimThreadsX, const int dimThreadsY,                \
    const int dimSlice,                                          \
    const int maxChunks,                                         \
    const int dimxBlock,                                         \
    const int dimyBlock,                                         \
    const int timesteps                                          \
  ) :                                                             
  bContinue (bContinue),
  deviceWaitWhileLoadingGlobalChunk (deviceWaitWhileLoadingGlobalChunk),
  deviceGlobalRefreshFlags (deviceGlobalRefreshFlags),
  dimThreadsX (dimThreadsX),
  dimThreadsY (dimThreadsY),
  dimSlice (dimSlice),
  maxChunks (maxChunks),
  dimxBlock (dimxBlock),
  dimyBlock (dimyBlock),
  timesteps (timesteps)
{}

structTestKernelControlStreamArguments::structTestKernelControlStreamArguments \
  (                                                             \
    unsigned char *hostWait4RefreshGlobalSlice,                 \
    unsigned char *hostWait4RefreshingChunk_WhileLoadingSlices, \
    unsigned char *hostWaitWhileLoadingGlobalChunk,             \
    unsigned char *deviceWaitWhileLoadingGlobalChunk,           \
    unsigned char *bContinue,                                   \
    unsigned char *deviceGlobalRefreshFlags,                    \
    const int dimThreadsX,                                      \
    const int dimThreadsY,                                      \
    const int dimSlice,                                         \
    const int maxChunks,                                        \
    const int dimxBlock,                                        \
    const int dimyBlock,                                        \
    const int timesteps                                         \
  ) :                                                            
  structSharedKernelArgumentsBase      \
  (                                    \
    bContinue,                         \
    deviceWaitWhileLoadingGlobalChunk, \
    deviceGlobalRefreshFlags,          \
    dimThreadsX, dimThreadsY,          \
    dimSlice,                          \
    maxChunks,                         \
    dimxBlock,                         \
    dimyBlock,                         \
    timesteps                          \
  ),                                    
  hostWait4RefreshGlobalSlice (hostWait4RefreshGlobalSlice),
  hostWait4RefreshingChunk_WhileLoadingSlices (hostWait4RefreshingChunk_WhileLoadingSlices),
  hostWaitWhileLoadingGlobalChunk (hostWaitWhileLoadingGlobalChunk)
{}

structTestKernelArguments::structTestKernelArguments  \
  (                                                   \
    const int dimx,                                   \
    const int dimy,                                   \
    const int dimz,                                   \
    const int dimxBlock,                              \
    const int dimyBlock,                              \
    const int dimSlice,                               \
    const int dimThreadsX,                            \
    const int dimThreadsY,                            \
    const int maxChunks,                              \
    const int timesteps,                              \
    float *buffer,                                    \
    clock_t *global_now,                              \
    unsigned char *deviceWaitWhileLoadingGlobalChunk, \
    unsigned char *bContinue,                         \
    unsigned char *deviceGlobalRefreshFlags           \
  ) :                                                  
  structSharedKernelArgumentsBase      \
  (                                    \
    bContinue,                         \
    deviceWaitWhileLoadingGlobalChunk, \
    deviceGlobalRefreshFlags,          \
    dimThreadsX, dimThreadsY,          \
    dimSlice,                          \
    maxChunks,                         \
    dimxBlock,                         \
    dimyBlock,                         \
    timesteps                          \
  ),                                    
  dimx (dimx), dimy (dimy), dimz (dimz)
{
  this->buffer = buffer;
  this->global_now = global_now;
}

//
