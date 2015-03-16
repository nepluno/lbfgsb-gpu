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

namespace lbfgsbcuda {
	namespace formt {
		__global__
		void kernel0(
			const int col,
			real* wt,
			const real* ss,
			const real theta)
		{
			const int j = threadIdx.x;
			wt[j] = theta * ss[j];
		}

		__global__
		void kernel1(
			const int col,
			const real* sy,
			const real* ss,
			real* wt,
			const int iPitch,
			const real theta
			)
		{
			const int i = blockIdx.y + 1;
			const int j = blockIdx.x * blockDim.y + threadIdx.y;

			if(j < i || j >= col)
				return;

			const int k1 = min(i, j);
			const int k = threadIdx.x;
			
			volatile __shared__ real sdata[4][9];

			real mySum = 0;
			if(k < k1) {
				mySum = sy[i * iPitch + k] * sy[j * iPitch + k] / sy[k * iPitch + k];
			}

			sdata[threadIdx.y][k] = mySum;
			__syncthreads();

			if(k < 4) {
				volatile real* smem = sdata[threadIdx.y] + k;
				*smem = mySum = mySum + smem[4];
				*smem = mySum = mySum + smem[2];
				*smem = mySum = mySum + smem[1];
			}

			if(k == 0) {
				wt[i * iPitch + j] = mySum + theta * ss[i * iPitch + j];
			}
		}

		void prog01(
			const int col,
			const real* sy,
			const real* ss,
			real* wt,
			const int iPitch,
			const real theta,
			const cudaStream_t& stream
			)
		{
			kernel0<<<1, col, 0, stream>>>
				(col, wt, ss, theta);
			CheckBuffer(wt, col, col);
			if(col > 1) {
				kernel1<<<dim3(iDivUp(col, 4), col - 1), dim3(8, 4), 0, stream>>>
					(col, sy, ss, wt, iPitch, theta);
			}
		}


	};
};