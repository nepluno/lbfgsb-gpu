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
	namespace lnsrlb {

		const static __constant__ real big = 1.0E10;
		template<int bx>		
		__global__ void
		kernel00(
			int n,
			const real* d,
			const int* nbd,
			const real* u,
			const real* x,
			const real* l,
			real* output
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;

			const int tid = threadIdx.x;
			volatile __shared__ real sdata[bx];

			real mySum = big;
			if(i < n) {
				real a1 = d[i];
				int nbdi = nbd[i];
				if(nbdi != 0) 
				{
					real xi = x[i];
					real a2;
					if(a1 > 0) {
						a2 = u[i] - xi;
					} else {
						a2 = xi - l[i];
					}
					a2 = maxr(0.0, a2);

					mySum = absr(a2 / a1);
				}
			}
			
			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = minr(mySum, sdata[tid + 512]); } __syncthreads();}						
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = minr(mySum, sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = minr(mySum, sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = minr(mySum, sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = minr(mySum, smem[32]);}
				if(bx > 16) {*smem = mySum = minr(mySum, smem[16]);}
				if(bx > 8) {*smem = mySum = minr(mySum, smem[8]);}
				if(bx > 4) {*smem = mySum = minr(mySum, smem[4]);}
				if(bx > 2) {*smem = mySum = minr(mySum, smem[2]);}
				if(bx > 1) {*smem = mySum = minr(mySum, smem[1]);}
			}

			if (tid == 0) 
				output[blockIdx.x] = mySum;
		}
		template<int bx>	
		__global__
		void kernel01(
			int n,
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
				mySum = big;

			sdata[tid] = mySum;
			__syncthreads();
			if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = minr(mySum, sdata[tid + 512]); } __syncthreads();}
			if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = minr(mySum, sdata[tid + 256]); } __syncthreads();}
			if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = minr(mySum, sdata[tid + 128]); } __syncthreads();}
			if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = minr(mySum, sdata[tid +  64]); } __syncthreads();}
    
			if (tid < __min(bx / 2, 32))
			{
				// now that we are using warp-synchronous programming (below)
				// we need to declare our shared memory volatile so that the compiler
				// doesn't reorder stores to it and induce incorrect behavior.
				volatile real* smem = sdata + tid;
				if(bx > 32) {*smem = mySum = minr(mySum, smem[32]);}
				if(bx > 16) {*smem = mySum = minr(mySum, smem[16]);}
				if(bx > 8) {*smem = mySum = minr(mySum, smem[8]);}
				if(bx > 4) {*smem = mySum = minr(mySum, smem[4]);}
				if(bx > 2) {*smem = mySum = minr(mySum, smem[2]);}
				if(bx > 1) {*smem = mySum = minr(mySum, smem[1]);}
			}

			if(tid == 0) {
				buf_out[blockIdx.x] = mySum;
			}
		}

		__global__
		void kernel2(
			int n,
			real* x,
			real* d,
			const real* t,
			const real stp
			)
		{
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i >= n) 
				return;

			real u = stp * d[i];
			x[i] = t[i] + u;
			d[i] = u;
		}

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
			)
		{
			int nblock0 = n;
			int mi = log2Up(nblock0);
			int nblock1 = iDivUp2(nblock0, mi);

			real* output = (nblock1 == 1) ? stpmx_dev : buf_s_r;
			dynamicCall(kernel00, mi, nblock1, 1, stream, (n, d, nbd, u, x, l, output));
						
			nblock0 = nblock1;
			while(nblock0 > 1) {

				nblock1 = iDivUp2(nblock0, mi);

				real* input = output;

				output = (nblock1 == 1) ? stpmx_dev : (output + nblock0);
				dynamicCall(kernel01, mi, nblock1, 1, stream, (nblock0, input, output));
				
				nblock0 = nblock1;
			}
		}

		void prog2(
			int n,
			real* x,
			real* d,
			const real* t,
			const real stp,
			const cudaStream_t& stream
			)
		{
			kernel2<<<dim3(iDivUp(n, 512)), dim3(512), 0, stream>>>
				(n, x, d, t, stp);
		}
	};
};