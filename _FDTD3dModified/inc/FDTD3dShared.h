#ifndef _FDTD3DSHARED_H_
#define _FDTD3DSHARED_H_


#define SCALARIDX 0

#define RADIUS_SHARED 4
#define VOLUME_SHARED 512
#define SPACEMULT 2
#define NSTEPS 100

typedef enum {X = 0, Y} selectXY_t;

template <int iM>
struct structMaterialUpdateCoefficientsBase
{
  float Media [iM];// If both media and volume are the same coefficients then their values stored in `.Media' struct member.
};

#define HXYM01P_t structMaterialUpdateCoefficientsBase<RADIUS_SHARED>
#define HXYM01W_t structMaterialUpdateCoefficientsBase<1>

#define DZM3P_t HXYM01P_t
#define DZM2PW_t HXYM01W_t
#define DXYM2W_t HXYM01W_t
#define DXYM2P_t HXYM01P_t
#define DXYM01W_t HXYM01W_t
#define DXYM01P_t HXYM01P_t

template <int iM, int iV>
struct structMaterialUpdateCoefficients : structMaterialUpdateCoefficientsBase<iM>
{
  float Volume [iV];
};

#define HXYM2P_t structMaterialUpdateCoefficients<RADIUS_SHARED, RADIUS_SHARED* VOLUME_SHARED * VOLUME_SHARED>
#define HZM3P_t HXYM2P_t
#define HXYM2W_t structMaterialUpdateCoefficients<1, VOLUME_SHARED * VOLUME_SHARED>
#define HZM2PW_t HXYM2W_t
#define EXYZM1PW_t HXYM2W_t

/*
struct structSpaceUpdateCoefficientsBasePolymorphic
{
  private:
    virtual void makeMePolymorphic () {};
};
*/

struct structSpaceUpdateCoefficientsBase_NonTemplate {};

//typedef structSpaceUpdateCoefficientsBase_NonTemplate Mbase_t;

template <typename MaterialP>
struct structSpaceUpdateCoefficientsBase : public structSpaceUpdateCoefficientsBase_NonTemplate
{
  MaterialP PML;
};

#define HZM3_t structSpaceUpdateCoefficientsBase<HZM3P_t>
//For H[Z] M2 PML and Window both are the same coefficient
#define HZM2_t structSpaceUpdateCoefficientsBase<HZM2PW_t>

#define DZM3_t structSpaceUpdateCoefficientsBase<DZM3P_t>
#define DZM2_t structSpaceUpdateCoefficientsBase<DZM2PW_t>

#define EXYZM1_t structSpaceUpdateCoefficientsBase<EXYZM1PW_t>

template <typename MaterialP, typename MaterialW>
struct structSpaceUpdateCoefficients : public structSpaceUpdateCoefficientsBase<MaterialP>
{
  MaterialW Window;
};

//typedef structSpaceUpdateCoefficients<HXYM01P_t, HXYM01W_t> HXYM01_t;
#define HXYM01_t structSpaceUpdateCoefficients<HXYM01P_t, HXYM01W_t>
#define HXYM2_t structSpaceUpdateCoefficients<HXYM2P_t, HXYM2W_t>

#define DXYM2_t structSpaceUpdateCoefficients<DXYM2P_t, DXYM2W_t>
#define DXYM01_t HXYM01_t

template <typename SpaceM2>
struct structVectorTermUpdateCoefficientsM2Base
{
  SpaceM2 m2;
};

template <typename SpaceM2, typename SpaceM3>
struct structVectorTermUpdateCoefficientsM23 : structVectorTermUpdateCoefficientsM2Base<SpaceM2>
{
  SpaceM3 m3;
};

// Actualy there is only M1 (named as M2 below) coefficient in E vector terms
#define EXYZ_t structVectorTermUpdateCoefficientsM2Base<EXYZM1_t>

#define HZ_t structVectorTermUpdateCoefficientsM23<HZM2_t, HZM3_t>

#define DZ_t structVectorTermUpdateCoefficientsM23<DZM2_t, DZM3_t>

template <typename SpaceM01, typename SpaceM2>
struct structVectorTermUpdateCoefficientsM012Static : structVectorTermUpdateCoefficientsM2Base<SpaceM2>
{
  /*volatile*/ static SpaceM01 m0, m1;//VOLATILE?????? // CUDA threads may access (read only!) this simultaneously, since there are different window and PML sections having the same constant m0 and m1, but been evaluated in different threads. // Members should be static to prevent unnecessary memory use for X and Y aggregating struct members. // Both X and Y have equal constants m0 and m1 for both H and D field components
};

#define HXY_t structVectorTermUpdateCoefficientsM012Static<HXYM01_t, HXYM2_t>

template <typename SpaceM01, typename SpaceM2>
struct structVectorTermUpdateCoefficientsM012 : structVectorTermUpdateCoefficientsM2Base<SpaceM2>
{
  SpaceM01 m0, m1;
};

#define DXY_t structVectorTermUpdateCoefficientsM012<DXYM01_t, DXYM2_t>

template <typename TermZ>
struct structFieldUpdateCoefficientsZBase
{
  TermZ Z;
};

template <typename TermXY, typename TermZ>
struct structFieldUpdateCoefficientsXZBase : structFieldUpdateCoefficientsZBase<TermZ>
{
  TermXY X;
};

#define D_t structFieldUpdateCoefficients<DXY_t, DZ_t>

template <typename TermXY, typename TermZ>
struct structFieldUpdateCoefficients : structFieldUpdateCoefficientsXZBase<TermXY, TermZ>
{
  TermXY Y;
};

#define H_t structFieldUpdateCoefficients<HXY_t, HZ_t>

#define E_t structFieldUpdateCoefficients<EXYZ_t, EXYZ_t>

struct structUpdateCoefficients
{
  H_t H;
  D_t D;
  E_t E;
};

typedef float ***f3_t;

//__device__ and __host__ common part (NOT for constant memory):
struct structXYZBase
{
  float X,
        Y,
        Z;
};

//__host__ members:
struct structXYZ : structXYZBase
{
  private:
    unsigned char izeros;
    int idimX,
        idimY,
        idimZ;

  public:
    structXYZ ();
    structXYZ ( unsigned char , int , int , int );

    ~structXYZ ();

  private:
    void *doalloc (int , size_t );
};

//typedef for __device__ variables:
typedef structXYZBase xyz_t;

//__device__ and __host__ common part (NOT for constant memory):
struct FieldComponentsBase
{
  xyz_t H,
        D,
        E;
};

//__host__ members:
struct FieldComponents : FieldComponentsBase
{
  FieldComponents ();
  FieldComponents ( int , int , int , int );
  ~FieldComponents();
};

typedef structUpdateCoefficients UpdateCoefficients_t;
// TODO : Rename `FieldComponents_t' into `EHD_t':
//typedef for __device__ variables:
typedef FieldComponentsBase FieldComponents_t;

typedef void ( *fdtdRefField_t ) ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib );

typedef void ( *Curl_t ) ( float&, xyz_t, int, int, int );

struct TFSFstruct
{
  xyz_t **sinsrc;
  void ( *getTFSF ) ( int, int, int );
}

typedef struct TFSFstruct TFSF_t;// C-style `struct' keyword before struct Name token. TODO : add it everywhere else.

//Function call wrapper for scalar update parameters
// H
inline void fdtdRefFieldWindowMediaHWrapper ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib );

// D
inline void fdtdRefFieldWindowMediaDWrapper ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib );

//H
inline void fdtdRefFieldWindowVolume ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib );

//H
inline void fdtdRefFieldPMLMedia ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib );

//D PML Media and Volume; overloadng of fdtdRefFieldPMLMedia
inline void fdtdRefFieldPMLMedia( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib );

//H
inline void fdtdRefFieldPMLVolume ( xyz_t &A, xyz_t F, f3_t &IC, float &C, structSpaceUpdateCoefficientsBase_NonTemplate Xm1C, structSpaceUpdateCoefficientsBase_NonTemplate Xm2C, structSpaceUpdateCoefficientsBase_NonTemplate Ym1C, structSpaceUpdateCoefficientsBase_NonTemplate Ym2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm2C, structSpaceUpdateCoefficientsBase_NonTemplate Zm3C, int ix, int iy, int iz, int il, int ia, int ib );

// H and D
// Call for both device and host
inline void fdtdRefSingleXY ( int xa, int xb, int ya, int yb, int &ix, int &iy, FieldComponents_t &FCout, FieldComponents_t &FCin, UpdateCoefficients_t &UC, f3_t &ICA, f3_t &ICB, float &C, int iz, int &il, int ilMin, fdtdRefField_t fdtdRFH, fdtdRefField_t fdtdRFD );


#endif
