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
#include "lbfgsbcuda.h"
#include <npps.h>

namespace lbfgsbcuda {
	namespace formk {

		__global__
		void kernel0
		(
		real* wn1,
		const int lx,
		const int ly,
		const int iPitch)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int j = blockIdx.y * blockDim.y + threadIdx.y;

			if(j >= ly || i >= lx || i > j) {
				return;
			}
			
			wn1[j * iPitch + i] = wn1[(j + 1) * iPitch + i + 1];
		}

		template<int bx>		
		__global__
		void kernel10
		(
			const int n,
			const int ipntr,
			real* output,
			const real* wy,
			const int head,
			const int m,
			const int iPitch_ws,
			const int oPitch
		)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int j = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];

			real mySum;

			if(i < n) {
				const int jpntr = Modular((head + j), m);
				mySum = wy[i * iPitch_ws + ipntr] * wy[i * iPitch_ws + jpntr];
			} else
				mySum = 0;

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}			
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if (tid == 0) 
				output[j * oPitch + blockIdx.x] = mySum;
		}

		template<int bx>
		__global__
		void kernel11(
			const int n,
			const int iPitch,
			const int oPitch,
			const real* buf_in,
			real* buf_out)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int j = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];

			real mySum;

			if(i < n)
				mySum = buf_in[j * iPitch + i];
			else
				mySum = 0;

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if(tid == 0) {
				buf_out[j * oPitch + blockIdx.x] = mySum;
			}
		}

		template<int bx>				
		__global__
		void kernel30
		(
			const int jpntr,
			const int head,
			const int m,
			const int n,
			const int iPitch_ws,
			const real* ws,
			const real* wy,
			real* output,
			const int oPitch
		)
		{
			const int k = blockIdx.x * blockDim.x + threadIdx.x;
			const int i = blockIdx.y;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];

			real mySum;

			if(k < n) {
				const int ipntr = Modular((head + i), m);
				mySum = ws[k * iPitch_ws + ipntr] * wy[k * iPitch_ws + jpntr];
			} else
				mySum = 0;

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = (mySum + sdata[tid + 512]); } __syncthreads();}			
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = (mySum + sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = (mySum + sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = (mySum + sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = mySum + smem[32];}
				if(bx > 16) {*smem = mySum = mySum + smem[16];}
				if(bx > 8) {*smem = mySum = mySum + smem[8];}
				if(bx > 4) {*smem = mySum = mySum + smem[4];}
				if(bx > 2) {*smem = mySum = mySum + smem[2];}
				if(bx > 1) {*smem = mySum = mySum + smem[1];}
			}

			if (tid == 0) 
				output[i * oPitch + blockIdx.x] = mySum;
		}

		__global__
		void kernel50(
				real* wn)
		{
			wn[1] = wn[1] / wn[0];
		}

		__global__
		void kernel5(
		int col,
		int iPitch_wn,
		real* wn)
		{
			const int iis = blockIdx.x + col;
			const int js = threadIdx.y + col;
			const int i = threadIdx.x;
			
			volatile __shared__ real sdata[64];

			real mySum = 0;
			if(blockIdx.y < col && blockIdx.x < col && js >= iis) {
				mySum = wn[i * iPitch_wn + iis] * wn[i * iPitch_wn + js];
			}

			volatile real* smem = sdata + (threadIdx.y * blockDim.x + i);
			*smem = mySum;
			__syncthreads();

			if(i < 4) {
				*smem = mySum = mySum + smem[4];
				*smem = mySum = mySum + smem[2];
				*smem = mySum = mySum + smem[1];
			}

			if(i == 0)
				wn[iis * iPitch_wn + js] += mySum;
		}

		void prog0(
			real* wn1,
			int m,
			int iPitch_wn,
			const cudaStream_t* streamPool)
		{
			int iblock = iDivUp(m * 2 - 1, 8);

			kernel0<<<dim3(iblock, iblock), dim3(8, 8)>>>
				(wn1, m * 2 - 1, m * 2 - 1, iPitch_wn);
		}
		
		void prog1(
			const int n,
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
			)
		{
			int nblock0 = n;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			real* output = (nblock1 == 1) ? (wn1 + (col - 1) * iPitch_wn) : buf_array_p;
			int op20 = (nblock1 == 1) ? 1 : n;

			dynamicCall(kernel10, mi, nblock1, col, streamPool[0], (n, ipntr, output, wy, head, m, iPitch_ws, op20));

/*
			kernel10<<<dim3(nblock1, col), dim3(512)>>>
				(nsub, ipntr, output, wy, head, m, iPitch_ws, op20);*/
			
			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				real* input = output;

				output = (nblock1 == 1) ? (wn1 + (col - 1) * iPitch_wn) : (output + nblock0);

				int op20 = (nblock1 == 1) ? 1 : n;

				dynamicCall(kernel11, mi, nblock1, col, streamPool[0], (nblock0, n, op20, input, output));

				nblock0 = nblock1;
			}
		}

		void prog2(
			real* wn1,
			const int col,
			const int m,
			const int iPitch_wn,
			const cudaStream_t* streamPool
			)
		{
			int offset = (col + m - 1) * iPitch_wn;

			cudaMemsetAsync(wn1 + offset + m, 0, col * sizeof(real), streamPool[1]);
			cudaMemsetAsync(wn1 + offset, 0, col * sizeof(real), streamPool[1]);
		}

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
			const cudaStream_t* streamPool)
		{
			int nblock0 = n;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			real* output = (nblock1 == 1) ? (wn1 + m * iPitch_wn + jy) : buf_array_p;
			int op20 = (nblock1 == 1) ? iPitch_wn : n;
			
			dynamicCall(kernel30, mi, nblock1, col, streamPool[2], (jpntr, head, m, n, iPitch_ws, ws, wy, output, op20));

			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				real* input = output;

				output = (nblock1 == 1) ? (wn1 + m * iPitch_wn + jy) : (output + nblock0);

				int op20 = (nblock1 == 1) ? iPitch_wn : n;
				dynamicCall(kernel11, mi, nblock1, col, streamPool[2], (nblock0, n, op20, input, output));

				nblock0 = nblock1;
			}
		}

		__global__ void
			kernel4(const int col,
				const int iPitch_wn,
				const int iPitch_ws,
				const int m,
				const real* wn1,
				const real theta,
				const real* sy,
				real* wn)
		{
			const int iy = blockIdx.y * blockDim.y + threadIdx.y;
			const int jy = blockIdx.x * blockDim.x + threadIdx.x;

			if(iy >= col * 2 || jy > iy)
				return;

			if(jy < col && jy == iy) {
				wn[iy * iPitch_wn + iy] = wn1[iy * iPitch_wn + iy] / theta + sy[iy * iPitch_ws + iy];
			} else if(jy < col - 1 && iy < col && iy > 0) {
				wn[jy * iPitch_wn + iy] = wn1[iy * iPitch_wn + jy] / theta;
			} else if(jy >= col && iy >= col) {
				wn[jy * iPitch_wn + iy] = wn1[(m - col + iy) * iPitch_wn + (m - col + jy)] * theta;
			} else if(jy < col - 1 && jy + col < iy && iy >= col + 1) {
				wn[jy * iPitch_wn + iy] = -wn1[(m - col + iy) * iPitch_wn + jy];
			} else if(jy < col && jy + col >= iy && iy >= col) {
				wn[jy * iPitch_wn + iy] = wn1[(m - col + iy) * iPitch_wn + jy];
			}
		}


		void prog4(
			const int col,
			const int iPitch_wn,
			const int iPitch_ws,
			const int m,
			const real* wn1,
			const real theta,
			const real* sy,
			real* wn,
			const cudaStream_t* streamPool)
		{
			int nblock = iDivUp(col * 2, 8);
			kernel4<<<dim3(nblock, nblock), dim3(8, 8), 0, streamPool[2]>>>
				(col, iPitch_wn, iPitch_ws, m, wn1, theta, sy, wn);
		}

		void prog5(
			const int col,
			const int iPitch_wn,
			real* wn,
			const cudaStream_t* streamPool)
		{

			real alpha = 1;
			if(col == 1)
			{
				kernel50<<<1,1, 0, streamPool[2]>>>(wn);
			} else {
				cublasSetStream(cublasHd, streamPool[2]);
			cublasRtrsm(cublasHd, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
				CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, col, col, &alpha, wn, iPitch_wn, wn + col, iPitch_wn);
			cublasSetStream(cublasHd, NULL);
			CheckBuffer(wn, 16, 14);
			
			}
			kernel5<<<dim3(col), dim3(8, col), 0, streamPool[2]>>>
				(col, iPitch_wn, wn);


		}
	};
};