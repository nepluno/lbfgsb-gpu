#include "lbfgsbcuda.h"

namespace lbfgsbcuda {
	namespace minimize {
		__global__ void
		kernel0(int n, real* d, const real stp) {
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i >= n)
				return;
			
			d[i] *= stp; 
		}

		__global__ void
		kernel1(int n, const real* a, const real* b, real* c) {
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i >= n)
				return;
			
			c[i] = a[i] - b[i];
		}

		__global__ void
		kernel2(int n, real* xdiff, real* xold, const real* x) {
			const int i = blockIdx.x * blockDim.x + threadIdx.x;
			if(i >= n)
				return;
			real xi = x[i];
			xdiff[i] = xold[i] - xi;
			xold[i] = xi;
		}
		void vsub_v(
			const int n,
			const real* a, const real* b, real* c,
			const cudaStream_t& stream)
		{
			kernel1<<<iDivUp(n, 512), 512, 0, stream>>>
				(n, a, b, c);
		}

		void vdiffxchg_v(
			const int n,
			real* xdiff, real* xold, const real* x,
			const cudaStream_t& stream)
		{
			kernel2<<<iDivUp(n, 512), 512, 0, stream>>>
				(n, xdiff, xold, x);
		}

		void vmul_v(
			const int n,
			real* d,
			const real stp,
			const cudaStream_t& stream)
		{
			kernel0<<<iDivUp(n, 512), 512, 0, stream>>>
				(n, d, stp);
		}

		void vdot_vv(
			const int n,
			const real* g,
			const real* d,
			real& gd,
			const cudaStream_t& stream
			)
		{
			cublasSetStream(cublasHd, stream);
			cublasRdot(cublasHd, n, g, 1, d, 1, &gd);
			cublasSetStream(cublasHd, NULL);
		}
	};
};