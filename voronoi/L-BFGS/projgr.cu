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
	namespace projgr {
		template<int bx>
		__global__ 
		void kernel0(const int n,
			 const real* l,
			 const real* u,
			 const int* nbd,
			 const real* x,
			 const real* g,
			 real* buf_n) 
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int tid = threadIdx.x;
			volatile __shared__ real sdata[bx];

			real mySum;

			if(i >= n) {
				mySum = sdata[tid] = 0;
			} else {
				real gi = g[i];
				int nbdi = nbd[i];
				if( nbdi != 0 )
				{
					if( gi < 0 )
					{
						if( nbdi >= 2 )
						{
							gi = maxr(x[i] - u[i], gi);
						}
					}
					else
					{
						if( nbdi <= 2 )
						{
							gi = minr(x[i] - l[i], gi);
						}
					}
				}
				mySum = sdata[tid] = absr(gi);
			}
			__syncthreads();

			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = maxr(mySum, sdata[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = maxr(mySum, sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = maxr(mySum, sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = maxr(mySum, sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = maxr(mySum, smem[32]);}
				if(bx > 16) {*smem = mySum = maxr(mySum, smem[16]);}
				if(bx > 8) {*smem = mySum = maxr(mySum, smem[8]);}
				if(bx > 4) {*smem = mySum = maxr(mySum, smem[4]);}
				if(bx > 2) {*smem = mySum = maxr(mySum, smem[2]);}
				if(bx > 1) {*smem = mySum = maxr(mySum, smem[1]);}
			}

			if (tid == 0) 
				buf_n[blockIdx.x] = mySum;

		}

		template<int bx>
		__global__
		void kernel01(
			const int n,
			const real* buf_in,
			real* buf_out)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			const int tid = threadIdx.x;
			
			volatile __shared__ real sdata[bx];

			real mySum;

			if(i < n)
				mySum = buf_in[i];
			else
				mySum = -machinemaximum;

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = maxr(mySum, sdata[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = maxr(mySum, sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = maxr(mySum, sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = maxr(mySum, sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = maxr(mySum, smem[32]);}
				if(bx > 16) {*smem = mySum = maxr(mySum, smem[16]);}
				if(bx > 8) {*smem = mySum = maxr(mySum, smem[8]);}
				if(bx > 4) {*smem = mySum = maxr(mySum, smem[4]);}
				if(bx > 2) {*smem = mySum = maxr(mySum, smem[2]);}
				if(bx > 1) {*smem = mySum = maxr(mySum, smem[1]);}
			}

			if(tid == 0) {
				buf_out[blockIdx.x] = mySum;
			}
		}

		void prog0(const int& n,
			 const real* l,
			 const real* u,
			 const int* nbd,
			 const real* x,
			 const real* g,
			 real* buf_n,
			real* sbgnrm,
			real* sbgnrm_dev,
			const cudaStream_t& stream)
		{
			int nblock0 = n;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			real* output = (nblock1 == 1) ? sbgnrm_dev : buf_n;

			dynamicCall(kernel0, mi, nblock1, 1, stream, (n, l, u, nbd, x, g, output));

			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				real* input = output;

				output = (nblock1 == 1) ? sbgnrm_dev : (output + nblock0);

				dynamicCall(kernel01, mi, nblock1, 1, stream, (nblock0, input, output));

				nblock0 = nblock1;
			}
		}
	};
};