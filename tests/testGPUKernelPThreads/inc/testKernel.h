//
/*************************************************************************/
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*                             testKernel.h                              */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*************************************************************************/
//

#ifndef _TESTKERNEL_H_
#define _TESTKERNEL_H_

// TODO : Insert copylefts.

// TODO : Move memory declarations to base structure for \
//         `structTestKernelControlStreamArguments' and \
//         `structTestKernelArguments':
// TODO : Implement constructors:
struct structSharedKernelArgumentsBase
{
  // TODO (DONE) : Use `unsigned char' everywhere for bitfields.
  // TODO : Figure out how to work with volatile bitfields from different \
  //         CUDA threads, streams, etc.                                   
  // Single flag if nothing noted or array of flags indicating (value `1') \
  //  that ...                                                              
                /* ... FDTD is running, e.g. threads in calculating stream  \
                 *  or one thread of control stream should process data     \
                 *  (`0' means FDTD calculations has finished):            */
  unsigned char *bContinue,
                /* ... loading of chunk from host input buffer to global  \
                 *  device memory in progress:                           */
                *deviceWaitWhileLoadingGlobalChunk,
                /* ... particular thread completed calculations and wrote  \
                 *  changes to global memory                               \
                 *  (this is a linear                                      \
                 *  blocksX x blocksY x threadsX x threadsY 1D array       \
                 *  of byte-flags):                                       */
                *deviceGlobalRefreshFlags;
  // TODO : Remember why modifier `const' is set for below fields:
  const int dimThreadsX,
            dimThreadsY,
            dimxBlock,
            dimyBlock,
            dimSlice,
            maxChunks,
            timesteps;

  structSharedKernelArgumentsBase                         \
  (                                                       \
    unsigned char * , unsigned char * , unsigned char * , \
    const int , const int , const int , const int ,       \
    const int , const int , const int                     \
  );                                                       
};

struct structTestKernelControlStreamArguments : public structSharedKernelArgumentsBase
{
  // TODO (DONE) : Use `unsigned char' everywhere for bitfields.
  // Single flag indicating (value `1') that ...
                /* ... particular slice loaded from device IO buffer to     \
                 *  host output buffer and then new slice loaded from host  \
                 *  input buffer back to device IO buffer:                 */
  unsigned char *hostWait4RefreshGlobalSlice,
                /* ... chunk still loaded to device global memory and host   \
                 *  should wait counting chunks during each count before     \
                 *  counter reach total number of chunks and then host swap  \
                 *  src and dst buffers for the next time cycle             */
                *hostWait4RefreshingChunk_WhileLoadingSlices,
                *hostWaitWhileLoadingGlobalChunk;

  structTestKernelControlStreamArguments                  \
  (                                                       \
    unsigned char * , unsigned char * , unsigned char * , \
    unsigned char * , unsigned char * , unsigned char * , \
    const int , const int , const int , const int ,       \
    const int , const int , const int                     \
  );                                                       
};

struct structTestKernelArguments : public structSharedKernelArgumentsBase
{
  // TODO : Remember why modifier `const' is set for below fields:
  const int dimx,
            dimy,
            dimz;
  float *buffer;
  clock_t *global_now;

  structTestKernelArguments                                     \
  (                                                             \
    const int , const int , const int , const int , const int , \
    const int , const int , const int , const int , const int , \
    float * , clock_t * ,                                       \
    unsigned char * , unsigned char * , unsigned char *         \
  );                                                             
};

struct structReloadMemChunkArgs
{
  // TODO : Verify that `sliceSize' is the only unchanged member:
  const int maxChunks,
            dimSlice,
            sliceSize,
            chunkSize,
            timesteps;
  const float *bufferSrc;
  unsigned char *hostWait4RefreshGlobalSlice;

#ifdef DEBUG_INFO

  unsigned char *debugFlags;

#endif

  int idxSlice;
  float *ioBuffer,
        *bufferDst;

#ifdef DEBUG_INFO

  structReloadMemChunkArgs ( const int , const int ,             \
                             const int , const int , const int , \
                             int ,                               \
                             const float* , float* , float* ,    \
                             unsigned char * , unsigned char *   \
                           );                                     

#else

  structReloadMemChunkArgs ( const int , const int ,             \
                             const int , const int , const int , \
                             int ,                               \
                             const float* , float* , float* ,    \
                             unsigned char * , unsigned char * , \
                             unsigned char *                     \
                           );                                     

#endif

};

typedef struct structTestKernelControlStreamArguments TestControlKernelArguments_t;
typedef struct structTestKernelArguments TestKernelArguments_t;
typedef struct structKernelPThreadArgs KernelPThreadArgs_t;
typedef struct structReloadMemChunkArgs ReloadMemChunkArgs_t;

#endif
