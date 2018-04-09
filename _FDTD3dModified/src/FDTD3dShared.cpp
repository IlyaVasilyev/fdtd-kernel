#include <FDTD3dShared.h>
#include <cstdlib>


structXYZ::structXYZ(): izeros(0), X(0), Y(0), Z(0), idimX(0), idimY(0), idimZ(0) {};

structXYZ::structXYZ ( unsigned char zeros, int dimX, int dimY, int dimZ ): izeros(zeros), idimX(dimX), idimY(dimY), idimZ(dimZ)
{
  int idxi, idxj;

  X = (float***)doalloc(dimX, sizeof( float** ));
  Y = (float***)doalloc(dimX, sizeof( float** ));
  Z = (float***)doalloc(dimX, sizeof( float** ));
  for ( idxi = 0; idxi < dimY; idxi++ )
  {
    X = (float**)doalloc(dimY, sizeof( float* ));
    Y = (float**)doalloc(dimY, sizeof( float* ));
    Z = (float**)doalloc(dimY, sizeof( float* ));
    for ( idxj = 0; idxj < dimZ; idxj++ )
    {
      X = (float*)doalloc(dimZ, sizeof( float ));
      Y = (float*)doalloc(dimZ, sizeof( float ));
      Z = (float*)doalloc(dimZ, sizeof( float ));
    }
  }
}

structXYZ::~structXYZ ()
{
  int idxi, idxj;

  for ( idxi = 0; idxi < idimY; idxi++ )
  {
    for ( idxj = 0; idxj < idimZ; idxj++ )
    {
      free(X[idxi][idxj]);
      free(Y[idxi][idxj]);
      free(Z[idxi][idxj]);
    }
    free(X[idxi]);
    free(Y[idxi]);
    free(Z[idxi]);
  }
  free(X);
  free(Y);
  free(Z);
}

void * structXYZ::doalloc( int length, size_t size )
{
  if ( izeros )
    return calloc ( length, size );
  else
    return malloc ( length * size );
}

FieldComponents::FieldComponents (): H(), D(), E() {};

FieldComponents::FieldComponents ( int zeros, int dimX, int dimY, int dimZ ): H(zeros, dimX, dimY, dimZ), D(zeros, dimX, dimY, dimZ), E(zeros, dimX, dimY, dimZ) {};

FieldComponents::~FieldComponents () {};
