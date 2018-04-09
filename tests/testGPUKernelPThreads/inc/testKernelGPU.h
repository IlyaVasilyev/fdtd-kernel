//
/*************************************************************************/
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*                            testKernelGPU.h                            */
/*                                                                       */
/*                                                                       */
/*                                                                       */
/*************************************************************************/
//


#ifndef _TESTKERNELGPU_H_
#define _TESTKERNELGPU_H_

/*
#ifndef _HELPERFUNCTIONS_HELPERCUDA__
#define _HELPERFUNCTIONS_HELPERCUDA__
*/

/*
#include <helper_functions.h>
#include <helper_cuda.h>
*/
#include <vector_types.h>

/*
#endif
*/

#include "testKernel.h"


struct structKernelPThreadArgs : public structTestKernelArguments
{
  dim3 dimGrid,
       dimBlock;
  unsigned char *bContinue;

  size_t maxSharedMemPerBlock;

  cudaStream_t streamCalc;
};

bool testGPU                                                              \
     (                                                                    \
       int argc, char **argv,                                             \
       float *input, float *output,                                       \
       int timesteps,                                                     \
       int maxChunks,                                                     \
       int chunkSize, /*!< Chunk size in bytes. */                        \
       int sliceSize, /*!< Slice size in bytes. */                        \
       int dimSlice, /*!< Number of slices in single chunk. */            \
       int blockSize,                                                     \
       int dimx, /*!< Length in x-direction of shared memory per block. */\
       int dimy, /*!< Length in y-direction of shared memory per block. */\
       int dimz, /*!< Thread memory length in z direction. */             \
       /* `dimxBlock' and `dimyBlock' - X and Y kernel grid dimensions */ \
       /*                               (in blocks).                   */ \
       int dimxBlock, /*!< Number of kernel grid X blocks. */             \
       int dimyBlock, /*!< Number of kernel grid Y blocks. */             \
       int dimThreadsX, /*!< Number of X threads per block in 2d grid. */ \
       int dimThreadsY  /*!< Number of Y threads per block in 2d grid. */ \
     );                                                                    

#endif
//
