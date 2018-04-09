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

#include "FDTD3d.h"

#include <iostream>
#include <iomanip>

#include "FDTD3dShared.h"
#include "FDTD3dReference.h"
#include "FDTD3dGPU.h"

#include <helper_functions.h>

#include <math.h>
#include <assert.h>

#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif

//// Name of the log file
//const char *printfFile = "FDTD3d.txt";

const float c_0 = 299792458;// m/s

// Forward declarations
bool runTest(int argc, const char **argv);
void showHelp(const int argc, const char **argv);
float3x3Tens_t **fillCoeffsWithTestCircWGData ( int );

int main(int argc, char **argv)
{
    bool bTestResult = false;
    // Start the log
    printf("%s Starting...\n\n", argv[0]);

    // Check help flag
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
        bTestResult = true;
    }
    else
    {
        // Execute
        bTestResult = runTest(argc, (const char **)argv);
    }

    // Finish
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void showHelp(const int argc, const char **argv)
{
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;

    std::cout << std::endl << "Syntax:" << std::endl;
    std::cout << std::left;
    std::cout << "    " << std::setw(20) << "--device=<device>" << "Specify device to use for execution" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimx=<N>" << "Specify number of elements in x direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimy=<N>" << "Specify number of elements in y direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimz=<N>" << "Specify number of elements in z direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--radius=<N>" << "Specify radius of stencil" << std::endl;
    std::cout << "    " << std::setw(20) << "--timesteps=<N>" << "Specify number of timesteps" << std::endl;
    std::cout << "    " << std::setw(20) << "--block-size=<N>" << "Specify number of threads per block" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(20) << "--noprompt" << "Skip prompt before exit" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(20) << "--wavelength" << "The wavelength in nm (633 nm by default)" << std::endl;
    std::cout << std::endl;
}

// TODO : Move these defs below somewhere else PLEASE!!!
// NOTE : Compiler requires by some reason unspecified length ( `[]' ) for first dimension in this construction:
// typedef float type_t[][X][Y]
#define float3darray_t(TYPEDEFNAME, DIMI, DIMJ, DIMK) float TYPEDEFNAME[DIMI][DIMJ][DIMK]
#define float2darray_t(TYPEDEFNAME, DIMJ, DIMK) float TYPEDEFNAME[DIMJ][DIMK]
typedef float3darray_t(f3da_t,RADIUS_SHARED,VOLUME_SHARED,VOLUME_SHARED);
typedef float2darray_t(f2da_t,VOLUME_SHARED,VOLUME_SHARED);

bool runTest(int argc, const char **argv)
{
    //unsigned char **coeff;
    FieldComponents_t *host_output_;
    FieldComponents_t *device_output;
    FieldComponents_t *input_;
    FieldComponents_t *host_output;
    FieldComponents_t *input;
    FieldComponents_t ***inputTFSFsrc;
    //float *coeffRanges;
    float *coeff;
    float3x3Tens_t **epsion_volume;
    float3x3Tens_t **mu_volume;

    f3da_t *f3daTmp;
    void *tmpGenRef;

    int defaultDim;
    /* TODO : Be carefull when copying sources from testFDTDGPU - the meaning of `dim..' in those source are slightly different from the one used below. In testFDTDGPU they are memory chunk or slice dimensions,  and here - they are just counts of working cells in FDTD grid. */
    int dimx;
    int dimy;
    int dimz;
    int outerDimx;
    int outerDimy;
    int outerDimz;
    int radius;
    int timesteps;
    int totalthreadspermp;
    int totalmps;
    int totalblockspermp;
    int totalthreadsperblock;
    int totalxthreadsperblock;
    int totalblocksingrid;// TODO: merge in one declaration with one `int' keyword
    float halftotalthreadspowperblock;
    float wavelength;
    float dt,
          dnu,
          dx, dy, dz,
          m_Hxy2_precalc,
          mPML_Hz2_precalc,
          mPML_Hxy2_precalc,
          mPML_HDxy0,
          m_HDxy0,
          sigmahalf_i,
          T_a,
          timesteps_df;
    size_t volumeSize;
    memsize_t memsize;
    float3x3Tens_t epsion_media;
    float3x3Tens_t mu_media;
    UpdateCoefficients_t ucLinearWG;

    const float lowerBound = 0.0f;
    const float upperBound = 1.0f;

    // Determine default dimensions
    printf("Set-up, based upon target device GMEM size...\n");
    // Get the memory size of the target device
    printf(" getTargetDeviceGlobalMemSize\n");
    getTargetDeviceGlobalMemSize(&totalblockspermp, &totalthreadspermp, &totalmps, &memsize, argc, argv);
    totalthreadsperblock = totalthreadspermp / totalblockspermp;
    totalblocksingrid = totalmps * totalblockspermp;
    halftotalthreadspowperblock = log2 ( (float)totalthreadsperblock ) / 2;
    totalxthreadsperblock = (int) pow ( 2, MAX( ceil ( halftotalthreadspowperblock ), floor ( halftotalthreadspowperblock ) ) );

    // We can never use all the memory so to keep things simple we aim to
    // use around half the total memory
    memsize /= 2;

    // Most of our memory use is taken up by the input and output buffers -
    // two buffers of equal size - and for simplicity the volume is a cube:
    //   dim = floor( (N/2)^(1/3) )
    defaultDim = (int)floor(pow((memsize / (2.0 * 9.0 * sizeof(float))), 1.0/3.0));

    // By default, make the volume edge size an integer multiple of 128B to
    // improve performance by coalescing memory accesses, in a real
    // application it would make sense to pad the lines accordingly
    int roundTarget = 128 / ( 9 * sizeof(float) );
    defaultDim = defaultDim / roundTarget * roundTarget;
    defaultDim -= k_radius_default * 2;

    // Check dimension is valid
    if (defaultDim < k_dim_min)
    {
        printf("insufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
        exit(EXIT_FAILURE);
    }
    else if (defaultDim > k_dim_max)
    {
        defaultDim = k_dim_max;
    }

    // For QA testing, override default volume size
    if (checkCmdLineFlag(argc, argv, "qatest"))
    {
        defaultDim = MIN(defaultDim, k_dim_qa);
    }

    //set default dim
    dimx = defaultDim;
    dimy = defaultDim;
    dimz = defaultDim;
    radius    = k_radius_default;
    timesteps = k_timesteps_default;
    wavelength = k_wavelength_default;

    // Parse command line arguments
    if (checkCmdLineFlag(argc, argv, "dimx"))
    {
        dimx = CLAMP(getCmdLineArgumentInt(argc, argv, "dimx"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimy"))
    {
        dimy = CLAMP(getCmdLineArgumentInt(argc, argv, "dimy"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimz"))
    {
        dimz = CLAMP(getCmdLineArgumentInt(argc, argv, "dimz"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "radius"))
    {
        radius = CLAMP(getCmdLineArgumentInt(argc, argv, "radius"), k_radius_min, k_radius_max);
    }

    if (checkCmdLineFlag(argc, argv, "wavelength"))
    {
        wavelength = CLAMP(getCmdLineArgumentInt(argc, argv, "wavelength"), k_wavelength_min, k_wavelength_max);
    }

    // Set time interval as 1/32 of period for free space
    dt = (wavelength / c_0) / 32;

    // Cell size
    dx = dy = dz = 1.10 * getSpeedOfLight() * dt * sqrt(3);

    // Desired frequency resolution
    dnu = 1 / ( wavelength * c_0 ) * 0.05;

    // Aquisition time
    T_a = 1.10 * dt * getWGLength() / dx + 1 / dnu ;

    // Number of timesteps ( 10% more than to resolve the frequency
    // response down to a resolution of dnu )
    if ( ( timesteps_df = 1.10 / (dnu*dt) ) / ( timesteps = T_a/dt ) < 1.10 )
      timesteps = timesteps + timesteps_df;

    if (checkCmdLineFlag(argc, argv, "timesteps"))
    {
        timesteps = CLAMP(getCmdLineArgumentInt(argc, argv, "timesteps"), k_timesteps_min, k_timesteps_max);
    }

    // Determine volume size
    outerDimx = dimx + 2 * radius;
    outerDimy = dimy + 2 * radius;
    outerDimz = dimz + 2 * radius;
    volumeSize = outerDimx * outerDimy * outerDimz;

    // Allocate memory
    host_output_ = new FieldComponents(1, volumeSize, volumeSize, volumeSize);
    input_ = new FieldComponents(0, volumeSize, volumeSize, volumeSize);
    host_output = (FieldComponents_t *)calloc(volumeSize, sizeof(FieldComponents_t));
    input       = (FieldComponents_t *)malloc(volumeSize * sizeof(FieldComponents_t));
    coeff       = (FieldComponents_t *)malloc((radius + 1) * sizeof(FieldComponents_t));

    // Create coefficients
    // Update coefficients and PML sigmas
    // radius - is the radius of PML
    // all PML sides have equal sigmas and depth
    //
    // Update coefficients of PML

    epsion_volume = fillCoeffsWithTestCircWGData ( VOLUME_SHARED - 2 * RADIUS_SHARED );
    mu_volume = fillCoeffsWithTestCircWGData ( VOLUME_SHARED - 2 * RADIUS_SHARED );

    mPML_Hz2_precalc = -c_0 * dt;
    ucLinearWG.D.Z.m2.PML.Media[SCALARIDX] = -mPML_Hz2_precalc;
    // Actualy there is only M1 (named as M2 below) coefficient in E vector terms
    ucLinearWG.E.X.m2.PML.Media[SCALARIDX] = 1 / epsion_media[0][0];
    ucLinearWG.E.Y.m2.PML.Media[SCALARIDX] = 1 / epsion_media[1][1];
    ucLinearWG.E.Z.m2.PML.Media[SCALARIDX] = 1 / epsion_media[2][2];
    // Update coefficients inside PML
    ucLinearWG.H.X.m0.Window.Media[SCALARIDX] = ucLinearWG.H.Y.m0.Window.Media[SCALARIDX] = ucLinearWG.D.X.m0.Window.Media[SCALARIDX] = ucLinearWG.D.Y.m0.Window.Media[SCALARIDX] = m_HDxy0 = 1/dt;
    ucLinearWG.H.X.m1.Window.Media[SCALARIDX] = ucLinearWG.H.Y.m1.Window.Media[SCALARIDX] = ucLinearWG.D.X.m1.Window.Media[SCALARIDX] = ucLinearWG.D.Y.m1.Window.Media[SCALARIDX] = ( 1 / m_HDxy0 ) * ( 1/dt );
    m_Hxy2_precalc = (-1) / m_HDxy0 * c_0;
    ucLinearWG.H.X.m2.Window.Media[SCALARIDX] = m_Hxy2_precalc / mu_media[0][0];
    ucLinearWG.H.Y.m2.Window.Media[SCALARIDX] = m_Hxy2_precalc / mu_media[1][1];
    //For H[Z] M2 PML and Window both are the same coefficient
    ucLinearWG.H.Z.m2.PML.Media[SCALARIDX] = mPML_Hz2_precalc / mu_media[2][2];
    ucLinearWG.D.X.m2.Window.Media[SCALARIDX] = c_0 / ucLinearWG.D.X.m0.Window.Media[SCALARIDX];
    ucLinearWG.D.Y.m2.Window.Media[SCALARIDX] = c_0 / ucLinearWG.D.Y.m0.Window.Media[SCALARIDX];
    // Update coefficients inside waveguide and in surroundings
    for ( int j = 0; j < VOLUME_SHARED ; j++ )
      for ( int k = 0; k < VOLUME_SHARED ; k++ )
      {
        // PML update coefficients
        // Actualy there is only M1 (named as M2 below) coefficient in E vector terms
        ((f2da_t &)(ucLinearWG.E.X.m2.PML.Volume))[j][k] = 1 / epsion_volume[j][k][0][0];
        ((f2da_t &)(ucLinearWG.E.Y.m2.PML.Volume))[j][k] = 1 / epsion_volume[j][k][1][1];
        ((f2da_t &)(ucLinearWG.E.Z.m2.PML.Volume))[j][k] = 1 / epsion_volume[j][k][2][2];
        // Update coefficients inside PML
        ((f2da_t &)(ucLinearWG.H.X.m2.Window.Volume))[j][k] = m_Hxy2_precalc / mu_volume[j][k][0][0];
        ((f2da_t &)(ucLinearWG.H.Y.m2.Window.Volume))[j][k] = m_Hxy2_precalc / mu_volume[j][k][1][1];
        //For H[Z] M2 PML and Window both are the same coefficients
        ((f2da_t &)(ucLinearWG.H.Z.m2.PML.Volume))[j][k] = mPML_Hz2_precalc / mu_volume[j][k][2][2];
      }
    for (int i = 0 ; i <= RADIUS_SHARED ; i++)
    {
      /* coeff[i] = 0.1f; */
      // Sigma is dimensionless ( without electric constant )
      sigmahalf_i = (1/(4 * dt)) * pow ( ((i+1)/radius), 3 );
      ucLinearWG.H.X.m0.PML.Media[i] = ucLinearWG.H.Y.m0.PML.Media[i] = ucLinearWG.D.X.m0.PML.Media[i] = ucLinearWG.D.Y.m0.PML.Media[i] = mPML_HDxy0 = 1/dt + sigmahalf_i;//!!!!!!
      ucLinearWG.H.X.m1.PML.Media[i] = ucLinearWG.H.Y.m1.PML.Media[i] = ucLinearWG.D.X.m1.PML.Media[i] = ucLinearWG.D.Y.m1.PML.Media[i] = ( 1 / mPML_HDxy0 ) * ( 1/dt - sigmahalf_i );
      mPML_Hxy2_precalc = (-1) / mPML_HDxy0 * c_0;
      ucLinearWG.H.X.m2.PML.Media[i] = mPML_Hxy2_precalc / mu_media[0][0];
      ucLinearWG.H.Y.m2.PML.Media[i] = mPML_Hxy2_precalc / mu_media[1][1];
      //ucLinearWG.H.Z.m2.PML.Media[i] = mPML_Hz2_precalc / mu_media[2][2];//FAIL//TODO
      ucLinearWG.H.Z.m3.PML.Media[i] = ucLinearWG.H.Z.m2.PML.Media[SCALARIDX] * dt * sigmahalf_i * 2;
      ucLinearWG.D.X.m2.PML.Media[i] = c_0 / ucLinearWG.D.X.m0.PML.Media[i];
      ucLinearWG.D.Y.m2.PML.Media[i] = c_0 / ucLinearWG.D.Y.m0.PML.Media[i];
      ucLinearWG.D.Z.m3.PML.Media[i] = -mPML_Hz2_precalc * 2 * sigmahalf_i;
      // Volume - is the boundary and nearby surroundings of a waveguide
      for ( int j = 0; j <= VOLUME_SHARED ; j++ )
        for ( int k = 0; k <= VOLUME_SHARED ; k++ )
        {
          // mu_volume[i][j][k] in case of Kerr effect
          (*((f3da_t *)(&ucLinearWG.H.X.m2.PML.Volume)))[i][j][k] = mPML_Hxy2_precalc / mu_volume[j][k][0][0];
          (*((f3da_t *)(&ucLinearWG.H.Y.m2.PML.Volume)))[i][j][k] = mPML_Hxy2_precalc / mu_volume[j][k][1][1];
          //(*((f3da_t *)(&ucLinearWG.H.Z.m2.PML.Volume)))[i][j][k] = mPML_Hz2_precalc / mu_volume[j][k][2][2];//FAIL//TODO
          (*((f3da_t *)(&ucLinearWG.H.Z.m3.PML.Volume)))[i][j][k] = (*((f2da_t *)(&ucLinearWG.H.Z.m2.PML.Volume)))[j][k] * dt * sigmahalf_i * 2;
        }
    }

    // TODO : I AM HERE (01.10.16) : Implement data generation for \
    //                                TFSF sources:                 
    // Generate data:
    printf ( "generateRandomData\n\n" );
    generateRandomData ( input, outerDimx, outerDimy, outerDimz, \
                         lowerBound, upperBound                  \
                       );                                         

    // TODO : I AM HERE (03.10.16) : Implement `inputTFSFsrc' handling      \
    //                                through out all procedures and kerels \
    //                                which contain input source            \
    //                                calculations:                          
    printf ( "Genrating sinusoidal source...\n\n" );
    generateSinSource ( inputTFSFsrc, dimx, dimy, timesteps, \
                        srcOmega, wgLength, wgRadius, T      \
                      );                                      

    printf("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);

    // Execute on the host
    printf("fdtdReference...\n");
    // TODO : Pass `inputTFSFsrc' as a parameter to function `fdtdReference':
    fdtdReference(host_output_, input_, ucLinearWG, dimx, dimy, dimz, radius, timesteps);
    printf("fdtdReference complete\n");

    // Allocate memory
    device_output = (FieldComponents_t ***)calloc(volumeSize, sizeof(FieldComponents_t));

    // TODO (23.11.16) : Change `float *' to `FieldComponents_t *':
    // Execute on the device:
    printf("fdtdGPU...\n");
    fdtdGPU ( device_output, input, inputTFSFsrc, ucLinearWG, \
              dimx, dimy, dimz, radius,                       \
              timesteps,                                      \
              argc, argv,                                     \
              totalthreadsperblock,                           \
              totalxthreadsperblock                           \
            );                                                 
    printf("fdtdGPU complete\n");

    // Compare the results
    float tolerance = 0.0001f;
    printf("\nCompareData (tolerance %f)...\n", tolerance);
    return compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);
}

float getWGLength ( void )// Linear
{
  return 0.20;// meters
}

float getSpeedOfLight ( void )
{
  return c_0;
}

float3x3Tens_t **fillCoeffsWithTestCircWGData ( int radiusOuter_ARG )
{
  float3x3Tens_t **tcwd_RES;

  float r;

  if ( radiusOuter_ARG > VOLUME_SHARED )
    radiusOuter_ARG = VOLUME_SHARED;

  tcwd_RES = (float3x3Tens_t **)malloc((VOLUME_SHARED) * sizeof(float3x3Tens_t *));
  for ( int idxi = 0; idxi < VOLUME_SHARED; idxi++ )
  {
    tcwd_RES[idxi] = (float3x3Tens_t *)malloc((VOLUME_SHARED) * sizeof(float3x3Tens_t));
    for ( int idxj = 0; idxi < VOLUME_SHARED; idxi++ )
    {
      r = sqrt ( pow ( idxi, 2 ) + pow ( idxj, 2 ) );
      if ( ( r > radiusOuter_ARG - 16 ) && ( r < radiusOuter_ARG ) )
        for ( int idxk = 0; idxk < 3; idxk++ )
          for ( int idxl = 0; idxl < 3; idxl++ )
            tcwd_RES[idxi][idxj][idxk][idxl] = 1000.0;
      else
        for ( int idxk = 0; idxk < 3; idxk++ )
          for ( int idxl = 0; idxl < 3; idxl++ )
            tcwd_RES[idxi][idxj][idxk][idxl] = 10.0;
    }
  }

  return tcwd_RES;
}
