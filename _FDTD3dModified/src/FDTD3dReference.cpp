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
#include "FDTD3dReference.h"

#include <cstdlib>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <stdio.h>


void generateRandomData(float *data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound)
{
    srand(0);

    for (int iz = 0 ; iz < dimz ; iz++)
    {
        for (int iy = 0 ; iy < dimy ; iy++)
        {
            for (int ix = 0 ; ix < dimx ; ix++)
            {
                *data = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
}

// TODO : Include H and D source data into `FieldComponents_t ***sourceData':
// The parameter `sourceData' has three dimensions: \
//  last two are used for tile reference frame,     \
//  and the first one - for time reference frame,   \
//  e.g. it represents tiles over time:              
void generateSinSource ( xyz_t ***sourceData, const int dimx, const int dimy, \
                         const int timesteps, const float srcOmega,           \
                         const float wgLength, const float wgRadius,          \
                         const float T                                        \
                       )                                                       
{
  // TODO (DONE) : Continue to implement code for source generation:
  float iFrame, jFrame,
        y,
        rho, sin_phi,
        phase;

  for ( int timestep = 0; timestep < timesteps; timestep++ )
    for ( int i = 0; i < dimx; i++ )
      for ( int j = 0; j < dimy; j++ )
      {
        iFrame = i-dimx;
        jFrame = j-dimy;
        y = abs(jFrame)/dimy;
        rho = sqrt ( pow(wgLength, 2)                                  \
                     * ( pow (2 * abs(iFrame) / dimx, 2) + pow(y, 2) ) \
                   );                                                   
        sin_phi = wgLength*y / rho;

        phase = sin_phi * sin ( 2*M_PI * rho / (wgRadius/10) ) \
                * sin ( srcOmega * timestep/T );                

        sourceData[timestep][i][j].X = 0;
        sourceData[timestep][i][j].Y = 0;
        sourceData[timestep][i][j].Z = phase;
      }
}

void generatePatternData(float *data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound)
{
    for (int iz = 0 ; iz < dimz ; iz++)
    {
        for (int iy = 0 ; iy < dimy ; iy++)
        {
            for (int ix = 0 ; ix < dimx ; ix++)
            {
                *data = (float)(lowerBound + ((float)iz / (float)dimz) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
}

// H and D
inline void fdtdRefLoopXY ( int xa, int xb, int ya, int yb, int &ix, int &iy, FieldComponents_t &FCout, FieldComponents_t &FCin, UpdateCoefficients_t &UC, f3_t &ICA, f3_t &ICB, float &C, int iz, int &il, int ilMin, fdtdRefField_t fdtdRFH, fdtdRefField_t fdtdRFD, TFSF_t TFSFsrc )
{
  for (ix = xa ; ix < xb ; ix++)
  {
    for (iy = ya ; iy < yb ; iy++)
    {
      fdtdRefSingleXY ( xa, xb, ya, yb, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, ilMin, fdtdRFH, fdtdRFD, TFSFsrc );
      /*
      (*fdtdRFH) ( FCout.H, FCin.H, ICA, C, UC.H.X.m1, UC.H.X.m2, UC.H.Y.m1, UC.H.Y.m2, UC.H.Z.m2, UC.H.Z.m3, ix, iy, iz, il - ilMin, ix - xa, iy - ya );
      (*fdtdRFD) ( FCout.D, FCin.D, ICB, C, UC.D.X.m1, UC.D.X.m2, UC.D.Y.m1, UC.D.Y.m2, UC.D.Z.m2, UC.H.Z.m3, ix, iy, iz, il - ilMin, ix - xa, iy - ya );
      */
    }
  }
}

inline void fdtdRef4LoopXY ( int xhalfpre, int xhalfpost, int yhalfpre, int yhalfpost, int dimx, int dimy, int &ix, int &iy, FieldComponents_t &FCout, FieldComponents_t &FCin, UpdateCoefficients_t &UC, float &ICA, float &ICB, float &C, int iz, int il, fdtdRefField_t fdtdRFH, fdtdRefField_t fdtdRFD, TFSF_t TFSFsrc )
{
  fdtdRefLoopXY ( 0, xhalfpre, 0, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrc );
  fdtdRefLoopXY ( xhalfpost, dimx, 0, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrc );
  fdtdRefLoopXY ( xhalfpre, xhalfpost, 0, yhalfpre, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrc );
  fdtdRefLoopXY ( xhalfpre, xhalfpost, yhalfpost, dimy, ix, iy, FCout, FCin, UC, ICA, ICB, C, iz, il, 0, fdtdRFH, fdtdRFD, TFSFsrc );
}

/*
bool fdtdReference(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
*/
bool fdtdReference(FieldComponents_t *output, const FieldComponents_t *input, const UpdateCoefficients_t updateCoeffs, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
{
    const int     outerDimx    = dimx + 2 * radius;
    const int     outerDimy    = dimy + 2 * radius;
    const int     outerDimz    = dimz + 2 * radius;
    const size_t  volumeSize   = outerDimx * outerDimy * outerDimz;
    const int     stride_y     = outerDimx;
    const int     stride_z     = stride_y * outerDimy;
    int ix, iy;
    float IH[dimx + 2 * RADIUS_SHARED][dimy + 2 * RADIUS_SHARED][dimz + 2 * RADIUS_SHARED],
          ID[dimx + 2 * RADIUS_SHARED][dimy + 2 * RADIUS_SHARED][dimz + 2 * RADIUS_SHARED],
          Curl;
    float        *intermediate = 0;
    const FieldComponents_t * bufsrc       = 0;
    FieldComponents_t        *bufdst       = 0;
    FieldComponents_t        *bufdstnext   = 0;

    // Allocate temporary buffer
    printf(" calloc intermediate\n");
    intermediate = (float *)calloc(volumeSize, sizeof(float));

    // Decide which buffer to use first (result should end up in output)
    // In timeframe we need to advance by half-timestep due to different
    // steps in H and D and also in E and B fields update equations
    if (((2 * timesteps) % 2) == 0)
    {
        bufsrc     = input;
        bufdst     = intermediate;
        bufdstnext = output;
    }
    else
    {
        bufsrc     = input;
        bufdst     = output;
        bufdstnext = intermediate;
    }

    // Run the FDTD (naive method)
    printf(" Host FDTD loop\n");

    // In a timeframe we need to advance by half-timestep due to different \
    //  steps in update equations associated with H and D and              \
    //  also E and B fields:                                                
    dimzNearPML = dimz - 1;
    dimyNearPML = dimy - 1;
    dimxNearPML = dimx - 1;
    dimXYHalfNearVolume = ( dimy - VOLUME_SHARED ) / 2;
    dimXYHalfAfterVolume = dimy - dimXYHalfNearVolume;
    dimXYPostPML = RADIUS_SHARED;
    dimXYPrePML = outerDimy - RADIUS_SHARED;
    dimZPrePML = outerDimz - RADIUS_SHARED;
    for (int it = 0 ; it < timesteps ; it++)
    {

      // Inside PML pseudoboundaries

      fdtdRef4LoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, dimx, dimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, 0, fdtdRefFieldWindowMediaHWrapper, fdtdRefFieldWindowMediaDWrapper, &getFDTDTFSFsrcE, &getFDTDTFSFsrcNull );

      fdtdRefLoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, 0, 0, fdtdRefFieldWindowVolume, fdtdRefFieldWindowMediaDWrapper, &getFDTDTFSFsrcE, &getFDTDTFSFsrcNull );

      fdtdRef4LoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, dimx, dimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, 0, fdtdRefFieldWindowMediaHWrapper, fdtdRefFieldWindowMediaDWrapper, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcH );

      fdtdRefLoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, 0, 0, fdtdRefFieldWindowVolume, fdtdRefFieldWindowMediaDWrapper, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcH );



      for ( int iz = RADIUS_SHARED ; iz < RADIUS_SHARED + 2 ; iz++ )
      {
        fdtdRefLoopXY ( 0, dimXYPostPML, 0, outerDimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, ix, 0, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
        fdtdRefLoopXY ( dimXYPrePML, outerDimx, 0, outerDimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, ix, dimx, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
        fdtdRefLoopXY ( dimXYPostPML, dimXYPrePML, 0, dimXYPostPML, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, iy, 0, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
        fdtdRefLoopXY ( dimXYPostPMLu, dimXYPrePML, dimXYPrePML, outerDimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, iy, dimy, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
      }

      for (int iz = RADIUS_SHARED + 2 ; iz < dimZPrePML ; iz++)
      {
        fdtdRef4LoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, dimx, dimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, 0, fdtdRefFieldWindowMediaHWrapper, fdtdRefFieldWindowMediaDWrapper, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );

        fdtdRefLoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, 0, 0, fdtdRefFieldWindowVolume, fdtdRefFieldWindowMediaDWrapper, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );

        fdtdRefLoopXY ( 0, dimXYPostPML, 0, outerDimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, ix, 0, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
        fdtdRefLoopXY ( dimXYPrePML, outerDimx, 0, outerDimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, ix, dimx, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
        fdtdRefLoopXY ( dimXYPostPML, dimXYPrePML, 0, dimXYPostPML, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, iy, 0, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
        fdtdRefLoopXY ( dimXYPostPMLu, dimXYPrePML, dimXYPrePML, outerDimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, iy, dimy, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
      }

      for (int iz = 0 ; iz < RADIUS_SHARED ; iz++)
      {
        fdtdRef4LoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, dimx, dimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, iz, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
        fdtdRef4LoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, dimx, dimy, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, RADIUS_SHARED + dimz + iz, RADIUS_SHARED - iz, fdtdRefFieldPMLMediaH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );

        fdtdRefLoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz, RADIUS_SHARED - iz, 0, fdtdRefFieldPMLVolumeH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );//RADIUS_SHARED - iz - that may cause an error due to passing argument inf subrouting by reference and not value --- Need min and max values and not only min value
        fdtdRefLoopXY ( dimXYHalfNearVolume, dimXYHalfAfterVolume, dimXYHalfNearVolume, dimXYHalfAfterVolume, ix, iy, *output, *input, updateCoeffs, IH, ID, Curl, iz + dimz + RADIUS_SHARED, iz, 0, fdtdRefFieldPMLVolumeH, fdtdRefFieldPMLMediaD, &getFDTDTFSFsrcNull, &getFDTDTFSFsrcNull );
      }
      // Rotate buffers
      FieldComponents_t *tmp = bufdst;
      bufdst     = bufdstnext;
      bufdstnext = tmp;
      bufsrc = (const FieldComponents_t *)tmp;


    }

    printf("\n");

    if (intermediate)
        free(intermediate);

    return true;
}

bool compareData(const float *output, const float *reference, const int dimx, const int dimy, const int dimz, const int radius, const float tolerance)
{
    for (int iz = -radius ; iz < dimz + radius ; iz++)
    {
        for (int iy = -radius ; iy < dimy + radius ; iy++)
        {
            for (int ix = -radius ; ix < dimx + radius ; ix++)
            {
                if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                {
                    // Determine the absolute difference
                    float difference = fabs(*reference - *output);
                    float error;

                    // Determine the relative error
                    if (*reference != 0)
                        error = difference / *reference;
                    else
                        error = difference;

                    // Check the error is within the tolerance
                    if (error > tolerance)
                    {
                        printf("Data error at point (%d,%d,%d)\t%f instead of %f\n", ix, iy, iz, *output, *reference);
                        return false;
                    }
                }

                ++output;
                ++reference;
            }
        }
    }

    return true;
}
