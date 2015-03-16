/*
Guodong Rong

voronoi.h
The header file for voronoi.cpp

Copyright (c) 2005-2006 
School of Computing
National University of Singapore
All Rights Reserved.
*/

#include <math.h>
#include <time.h>
#include "L-BFGS/cutil_inline.h"

#include "L-BFGS/lbfgsb.h"

#define M_PI 3.14159265358979323846

// destination color buffers
const GLenum buffers[] = {
	GL_AUX0,
	GL_AUX1,
	GL_AUX2,
	GL_AUX3
};

const GLenum fbo_attachments[] = {
	GL_COLOR_ATTACHMENT0_EXT,
	GL_COLOR_ATTACHMENT1_EXT,
	GL_COLOR_ATTACHMENT2_EXT,
	GL_COLOR_ATTACHMENT3_EXT
};

typedef struct VertexSiteType{
	float x, y;
}VertexSiteType;

typedef struct SiteType{
	VertexSiteType vertices[1];
	float color[4];
}SiteType;

bool animation = false;
bool site_visible = false;
bool testFPS = false;
bool output = false;

int screenwidth, screenheight;
int mode;
int point_num, line_num, nurbs_num;
int site_num;
int frame_num = 0;
float speed;
int additional_passes, additional_passes_before;
bool bReCompute;
real stpscal;

SiteType *site_list;
SiteType *site_list_dev;
float *controlpoints;

GLuint Processed_Texture[2], Site_Texture, Color_Texture, Energy_Texture[2], IndexColor_Texture;
GLuint FB_objects, RB_object;
int Current_Buffer;
GLuint occlusion_query;
GLuint sampleCount;
GLint oq_available;
GLint oq_bitsSupported;
GLuint vboId;
GLuint colorboId;

cudaGraphicsResource_t grSite;
cudaGraphicsResource_t grVbo;

real* f_tb_host;
real* f_tb_dev;

// For L-BFGS
float *pReadBackValues;
int iSiteTextureHeight;
double EnergyValue;
bool bNewIteration;
int numIter;
FILE *f_result;
cublasHandle_t cublasHd;

GLuint ScreenPointsList;

/* New Cg global variables */
CGcontext Context;
CGprofile VertexProfile, FragmentProfile;

CGprogram VP_DrawSites, FP_DrawSites;
CGprogram VP_Flood, FP_Flood;
CGprogram VP_Scatter, FP_Scatter;
CGprogram VP_DrawNewSites, FP_DrawNewSites;
CGprogram VP_DrawSitesOQ, FP_DrawSitesOQ;
CGprogram VP_FinalRender, FP_FinalRender;
CGprogram VP_ComputeEnergy, FP_ComputeEnergy;
CGprogram VP_Deduction, FP_Deduction;
CGprogram VP_ComputeEnergyCentroid, FP_ComputeEnergyCentroid;
CGprogram VP_ScatterCentroid, FP_ScatterCentroid;

// VP_Flood program uniform variables
CGparameter VP_Flood_Steplength;
CGparameter VP_Flood_Size;

// VP_Scatter program uniform variables
CGparameter VP_Scatter_Size;

// VP_DrawNewSites program uniform variables
CGparameter VP_DrawNewSites_Size;

// VP_DrawSitesOQ program uniform variables
CGparameter VP_DrawSitesOQ_Size;

// FP_FinalRender program uniform variables
CGparameter FP_FinalRender_Size;

// FP_ComputeEnergy program uniform variables
CGparameter FP_ComputeEnergy_Size;

// FP_ComputeEnergyCentroid program uniform variables
CGparameter FP_ComputeEnergyCentroid_Size;

// VP_ScatterCentroid program uniform variables
CGparameter VP_ScatterCentroid_Size;

/*********************************************/
void CheckFramebufferStatus();
void DrawSites(real* x, bool FinalDrawSite, const cudaStream_t& stream);
real BFGSOptimization(void);
real DrawVoronoi(real* x);
void Display(void);
void Keyboard(unsigned char key, int x, int y);
void InitializeGlut(int *argc, char *argv[]);
void CgErrorCallback(void);
void UpdateSites();
void InitializeSites(int point_num, int line_num, int nurbs_num);
void InitCg();

// Time...
#ifdef WIN32

typedef LARGE_INTEGER timestamp;

static inline float LI2f(const LARGE_INTEGER &li)
{
	// Workaround for compiler bug.  Sigh.
	float f = unsigned(li.u.HighPart) >> 16;  f *= 65536.0f;
	f += unsigned(li.u.HighPart) & 0xffff;    f *= 65536.0f;
	f += unsigned(li.u.LowPart) >> 16;        f *= 65536.0f;
	f += unsigned(li.u.LowPart) & 0xffff;
	return f;
}

static inline float operator - (const timestamp &t1, const timestamp &t2)
{
	static LARGE_INTEGER PerformanceFrequency;
	static int status = QueryPerformanceFrequency(&PerformanceFrequency);
	if (status == 0) return 1.0f;

//	return (LI2f(t1) - LI2f(t2)) / LI2f(PerformanceFrequency);
	return (t1.QuadPart - t2.QuadPart)/(double)PerformanceFrequency.QuadPart;
}

static inline void get_timestamp(timestamp &now)
{
	QueryPerformanceCounter(&now);
}

#else

typedef struct timeval timestamp;

static inline float operator - (const timestamp &t1, const timestamp &t2)
{
	return (float)(t1.tv_sec  - t2.tv_sec) +
	       1.0e-6f*(t1.tv_usec - t2.tv_usec);
}

static inline void get_timestamp(timestamp &now)
{
	gettimeofday(&now, 0);
}

#endif

timestamp start_time, end_time;
double elapsed_time, total_time;

timestamp start_time_func, end_time_func;
double elapsed_time_func, total_time_func;

//#define DEBUG_TIME

bool bShowTestResults = true;
int nFuncCall;