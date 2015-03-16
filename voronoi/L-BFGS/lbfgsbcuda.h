/*************************************************************************
GPU Version:
Tsinghua University, Aug. 2012.

Written by Yun Fei in collaboration with
W. Wang and B. Wang

Original:
Optimization Technology Center.
Argonne National Laboratory and Northwestern University.

Written by Ciyou Zhu in collaboration with
R.H. Byrd, P. Lu-Chen and J. Nocedal.

Contributors:
    * Sergey Bochkanov (ALGLIB project). Translation from FORTRAN to
      pseudocode.
      
	  This software is freely available, but we  expect  that  all  publications
	  describing  work using this software, or all commercial products using it,
	  quote at least one of the references given below:
	  * R. H. Byrd, P. Lu and J. Nocedal.  A Limited  Memory  Algorithm  for
	  Bound Constrained Optimization, (1995), SIAM Journal  on  Scientific
	  and Statistical Computing , 16, 5, pp. 1190-1208.
	  * C. Zhu, R.H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
	  FORTRAN routines for  large  scale  bound  constrained  optimization
	  (1997), ACM Transactions on Mathematical Software,  Vol 23,  Num. 4,
	  pp. 550 - 560.
*************************************************************************/

#pragma once

#include "cutil_inline.h"
#include <cublas_v2.h>

extern cublasHandle_t cublasHd;

#define MODU8

#ifdef MODU8
#define Modular(a, b) (a & 7)
#else
#define Modular(a, b) (a % b)
#endif

#define dynamicCall(f, bx, nblockx, nblocky, st, var) \
{\
	switch(bx) { \
	case 9: \
		f<512><<<dim3(nblockx, nblocky), dim3(512), 0, st>>>var;\
		break;\
	case 8: \
		f<256><<<dim3(nblockx, nblocky), dim3(256), 0, st>>>var;\
		break;\
	case 7: \
		f<128><<<dim3(nblockx, nblocky), dim3(128), 0, st>>>var;\
		break;\
	default: \
		f<64><<<dim3(nblockx, nblocky), dim3(64), 0, st>>>var;\
		break;\
	} \
}

#define inv9l2 0.36910312165415137198559104772104
#define invl2 3.3219280948873623478703194294894
#define LBFGSB_CUDA_DOUBLE_PRECISION

#ifdef LBFGSB_CUDA_DOUBLE_PRECISION
typedef double real;
#define machineepsilon 5E-16
#define machinemaximum 1e50
#define maxr(a, b) fmax(a, b)
#define minr(a, b) fmin(a, b)
#define absr(a) fabs(a)
#define sqrtr(a) sqrt(a)
#define rsqrtr(a) rsqrt(a)
#define cublasRtrsm cublasDtrsm
#define cublasRtrsv cublasDtrsv
#define cublasRdot cublasDdot
#define EPSG 5E-16
#define EPSF 5E-16
#define EPSX 5E-16
#define MAXITS 1000
#else
typedef float real;
#define machineepsilon 5E-7f
#define machinemaximum 1e20f
#define maxr(a, b) fmaxf(a, b)
#define minr(a, b) fminf(a, b)
#define absr(a) fabsf(a)
#define sqrtr(a) sqrtf(a)
#define rsqrtr(a) rsqrtf(a)
#define cublasRtrsm cublasStrsm
#define cublasRtrsv cublasStrsv
#define cublasRdot cublasSdot
#define EPSG 1e-37f
#define EPSF 1e-37f
#define EPSX 1e-37f
#define MAXITS 1000
#endif

#define SYNC_LEVEL 2

namespace lbfgsbcuda {
	inline void debugSync() {
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
#endif
	}
	inline int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}
	inline int iDivUp2(int a, int b)
	{
		int c = a >> b;
		return (a > (c << b)) ? (c + 1) : c;
	}
	inline int log2Up(int n) {
		real lnb = log10((real)n);
		real nker = ceil(lnb * inv9l2);
		int m = ceil(lnb * invl2 / nker);
		m = __max(6, m);
		return m;
	}
	inline void CheckBuffer(const real* q, int stride, int total) {
#ifdef _DEBUG
		if(stride <= 0 || total <= 0)
			return;
		int h = iDivUp(total, stride);
		int wh = h * stride;
		real* hq = new real[wh];
		memset(hq, 0, sizeof(real) * wh);
		cutilSafeCall(cudaMemcpy(hq, q, total * sizeof(real), cudaMemcpyDeviceToHost));

/*
		char* pBufStr = new char[30 * wh];
		pBufStr[0] = '\0';
		char pbuft[30];

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < stride; j++) {
				sprintf(pbuft, "%.9lf ", hq[i * stride + j]);
				strcat(pBufStr, pbuft);
			}
			strcat(pBufStr, "\n");
		}

		printf(pBufStr);*/
		delete[] hq;
/*		delete[] pBufStr;*/
#endif
	}

	inline void CheckBuffer_int(int* q, int stride, int total) {
#ifdef _DEBUG
		if(stride <= 0 || total <= 0)
			return;
		int h = iDivUp(total, stride);
		int wh = h * stride;
		int* hq = new int[wh];
		memset(hq, 0, sizeof(int) * wh);
		cutilSafeCall(cudaMemcpy(hq, q, total * sizeof(int), cudaMemcpyDeviceToHost));
/*

		char* pBufStr = new char[30 * wh];
		pBufStr[0] = '\0';
		char pbuft[30];

		for(int i = 0; i < h; i++) {
			for(int j = 0; j < stride; j++) {
				sprintf(pbuft, "%d ", hq[i * stride + j]);
				strcat(pBufStr, pbuft);
			}
			strcat(pBufStr, "\n");
		}

		printf(pBufStr);*/
		delete[] hq;
/*		delete[] pBufStr;*/
#endif
	}

	namespace minimize {
		void vdot_vv(
			int n,
			const real* g,
			const real* d,
			real& gd,
			const cudaStream_t& stream = NULL
			);
		void vmul_v(
			const int n,
			real* d,
			const real stp,
			const cudaStream_t& stream = NULL
			);
		void vsub_v(
			const int n,
			const real* a, const real* b, real* c, const cudaStream_t& stream = NULL);
		void vdiffxchg_v(
			const int n,
			real* xdiff, real* xold, const real* x,
			const cudaStream_t& stream = NULL
			);
	};
	namespace active {
		void prog0(
			const int& n,
			const real* l,
			const real* u,
			const int* nbd,
			real* x
			);
	};
	namespace projgr {
		void prog0(const int& n,
			const real* l,
			const real* u,
			const int* nbd,
			const real* x,
			const real* g,
			real* buf_n,
			real* sbgnrm,
			real* sbgnrm_dev,
			const cudaStream_t& stream);
	};
	namespace cauchy {
		void prog0
			(const int& n,
			const real* x,
			const real* l,
			const real* u,
			const int* nbd,
			const real* g,
			real* t,
			real* xcp,
			real* xcpb,
			const int& m,
			const real* wy,
			const real* ws,
			const real* sy,
			const int iPitch,
			real* wt,
			const real& theta,
			const int& col,
			const int& head,
			real* p,
			real* c,
			real* v,
			int& nint,
			const real& sbgnrm,
			real* buf_s_r,
			real* buf_array_p,
			const cudaStream_t* streamPool
			);
	};
	namespace formk {
		void prog0(
			real* wn1,
			int m,
			int iPitch_wn,
			const cudaStream_t* streamPool
			);
		void prog1(
			int n,
			const int ipntr,
			real* wn1,
			real* buf_array_p,
			const real* wy,
			const int head,
			const int m,
			const int col,
			const int iPitch_ws,
			const int iPitch_wn,
			const cudaStream_t* streamPool
			);
		void prog2(
			real* wn1,
			const int col,
			const int m,
			const int iPitch_wn,
			const cudaStream_t* streamPool
			);
		void prog3(
			const int jpntr,
			const int head,
			const int m,
			const int col,
			const int n,
			const int iPitch_ws,
			const int iPitch_wn,
			const int jy,
			const real* ws,
			const real* wy,
			real* buf_array_p,
			real* wn1,
			const cudaStream_t* streamPool);
		void prog4(
			const int col,
			const int iPitch_wn,
			const int iPitch_ws,
			const int m,
			const real* wn1,
			const real theta,
			const real* sy,
			real* wn,
			const cudaStream_t* streamPool);
		void prog5(
			const int col,
			const int iPitch_wn,
			real* wn,
			const cudaStream_t* streamPool);
	};
	namespace cmprlb {
		void prog0(
			int n,
			real* r,
			const real* g,
			const cudaStream_t& stream
			);
		void prog1(
			int nfree,
			const int col,
			const int head,
			const int m,
			const int iPitch,
			const real* wa,
			const real* wy,
			const real* ws,
			const real theta,
			const real* z,
			const real* x,
			const real* g,
			real* r,
			const cudaStream_t& stream
			);
	};
	namespace subsm {
		void prog0(
			const int n,
			const int head,
			const int m,
			const int col,
			const int iPitch_ws,
			real* buf_array_p,
			const real* wy,
			const real* ws,
			const real* d,
			real* wv,
			const real theta,
			const cudaStream_t& stream
			);
		void prog1(
			real* wn,
			int col,
			int iPitch_wn,
			real* wv,
			const cudaStream_t& stream
			);
		void prog2(
			int nsub,
			const int col,
			const int head,
			const int m,
			const int iPitch,
			const real* wv,
			const real* wy,
			const real* ws,
			const real theta,
			real* d,
			const cudaStream_t& stream
			);
		void prog3
			(int nsub,
			real* d,
			const int* nbd,
			real* buf_s_r,
			int* bufi_s_r,
			real* x,
			const real* u,
			const real* l,
			const cudaStream_t& stream
			);
	};
	namespace lnsrlb {
		void prog0(
			int n,
			const real* d,
			const int* nbd,
			const real* u,
			const real* x,
			const real* l,
			real* buf_s_r,
			real* stpmx_host,
			real* stpmx_dev,			
			const cudaStream_t& stream
			);
		void prog2(
			int n,
			real* x,
			real* d,
			const real* t,
			const real stp,
			const cudaStream_t& stream
			);
	};
	namespace matupd {
		void prog0(
			const int& n,
			const int& m,
			real* wy,
			real* sy,
			const real* r,
			const real* d,
			int& itail,
			const int& iupdat,
			int& col,
			int& head,
			const real& dr,
			const int& iPitch0,
			const int& iPitch_i,
			const int& iPitch_j,
			real* buf_array_p,
			cudaStream_t st);
	};
	namespace formt {
		void prog01(
			const int col,
			const real* sy,
			const real* ss,
			real* wt,
			const int iPitch,
			const real theta,
			const cudaStream_t& stream
			);
	};
	namespace bmv {
		void prog0(
			const real* sy,
			const int& col,
			const int& iPitch,
			const real* v,
			real* p,
			const cudaStream_t& st);
		void prog1(
			const real* wt,
			const int& col,
			const int& iPitch,
			const real* v,
			real* p,
			const cudaStream_t& st
			);
		void prog2(
			const real* sy,
			real* wt,
			const int& col,
			const int& iPitch,
			const real* v,
			real* p,
			const cudaStream_t& st);
	};
	namespace dpofa {
		void prog0(real* m, int n, int pitch, int boffset, const cudaStream_t& st);
	};
};