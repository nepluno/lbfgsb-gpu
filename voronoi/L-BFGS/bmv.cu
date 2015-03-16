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

#include "cutil_inline.h"
#include "lbfgsbcuda.h"
#include <cublas_v2.h>

namespace lbfgsbcuda {
	namespace bmv 
	{
		__global__ void
		kernel0(
			const real* sy,
			const int col,
			const real* v,
			const int iPitch,
			real* p
		)
		{
			const int i = blockIdx.x * blockDim.y + threadIdx.y;
			const int k = threadIdx.x;
			const int i2 = col + i;

			volatile __shared__ real sdata[4][9];

			real mySum = 0;
			if(k < i && i < col) {
				mySum = sy[i * iPitch + k] * v[k] / sy[k * iPitch + k];
			}

			sdata[threadIdx.y][k] = mySum;

			__syncthreads();

			if (k < 4)
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata[threadIdx.y] + k;
				*smem = mySum = mySum + smem[4];
				*smem = mySum = mySum + smem[2];
				*smem = mySum = mySum + smem[1];
			}

			if(k == 0 && i < col) {
				p[i2] = v[i2] + mySum;
			}
		}

		__global__ void
		kernel1(
			const real* sy,
			const int col,
			const real* v,
			const int iPitch,
			real* p
		)
		{
			const int i = blockIdx.x * blockDim.y + threadIdx.y;
			const int k = threadIdx.x;

			volatile __shared__ real sdata[4][9];

			real mySum = 0;
			real pre = 0;

			if(i < col) {
				real syii = 1.0 / sy[i * iPitch + i];
				pre = -v[i] * syii;

				if(k > i && k < col) {
					mySum = sy[k * iPitch + i] * p[col + k] * syii;
				}
			}

			sdata[threadIdx.y][k] = mySum;

			__syncthreads();

			if (k < 4)
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata[threadIdx.y] + k;
				*smem = mySum = mySum + smem[4];
				*smem = mySum = mySum + smem[2];
				*smem = mySum = mySum + smem[1];
			}

			if(k == 0 && i < col) {
				p[i] = pre + mySum;
			}

		}

		void prog0(
			 const real* sy,
			 const int& col,
			 const int& iPitch,
			 const real* v,
			 real* p,
			 const cudaStream_t& st)
		{
			int nblocks = iDivUp(col, 4);
			
			if(col <= 1) {
				if(!st) {
					cudaMemcpy(p + col, v + col, sizeof(real), cudaMemcpyDeviceToDevice);
				} else {
					cudaMemcpyAsync(p + col, v + col, sizeof(real), cudaMemcpyDeviceToDevice, st);
				}
				return;
			}

			if(!st) {
				kernel0<<<nblocks, dim3(8, 4)>>>
					(sy, col, v, iPitch, p);
			} else {
				kernel0<<<nblocks, dim3(8, 4), 0, st>>>
					(sy, col, v, iPitch, p);
			}
		}

		void prog1(
			 const real* wt,
			 const int& col,
			 const int& iPitch,
			 const real* v,
			 real* p,
			 const cudaStream_t& st
			)
		{
			if(st)
				cublasSetStream(cublasHd, st);

			cublasRtrsv(cublasHd, 
				CUBLAS_FILL_MODE_LOWER,
				CUBLAS_OP_N,
				CUBLAS_DIAG_NON_UNIT,
				col,
				wt,
				iPitch,
				p + col,
				1);
			cublasRtrsv(cublasHd, 
				CUBLAS_FILL_MODE_LOWER,
				CUBLAS_OP_T,
				CUBLAS_DIAG_NON_UNIT,
				col,
				wt,
				iPitch,
				p + col,
				1);

		}

		void prog2(
			 const real* sy,
			 real* wt,
			 const int& col,
			 const int& iPitch,
			 const real* v,
			 real* p,
			 const cudaStream_t& st)
		{
			int nblocks = iDivUp(col, 4);

			if(!st) {
				kernel1<<<nblocks, dim3(8, 4)>>>
					(sy, col, v, iPitch, p);
			} else {
				kernel1<<<nblocks, dim3(8, 4), 0, st>>>
					(sy, col, v, iPitch, p);
			}
		}
	};
};