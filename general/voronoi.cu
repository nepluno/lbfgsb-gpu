#include "L-BFGS/cutil_inline.h"
#include "L-BFGS/lbfgsbcuda.h"


texture<float4, 2, cudaReadModeElementType> inTex;

template<int bx>
__global__ void
	EnergyfKer(real* g, real* pf, int w, int nsite)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
			
	volatile __shared__ real sdata[bx];

	real mySum;
	
	float4 c;
	c.x = c.y = 0;
	if(i >= nsite) {
		mySum = 0;
	} else {
		const int iy = i / w;
		const int ix = i - iy * w;

		c = tex2D(inTex, ix + 1, iy + 1);
		mySum = c.z;
	}
	
	sdata[tid] = mySum;
	__syncthreads();
	if(bx > 512) {if (tid < 512) { sdata[tid] = mySum = mySum + sdata[tid + 512]; } __syncthreads();}
	if(bx > 256) {if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads();}
	if(bx > 128) {if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();}
	if(bx > 64) {if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid + 64]; } __syncthreads();}
    
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
		pf[blockIdx.x] = mySum;
	}

	if(i < nsite) {
		g[i * 2] = c.x * 2.0f;
		g[i * 2 + 1] = c.y * 2.0f;
	}
}

template<int bx>
__global__
void EnergyfKer1(
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
		buf_out[blockIdx.x] = mySum;
	}
}

extern void Energyf(cudaGraphicsResource_t grSite, real* g, real* f, int w, int h, int nsite, const cudaStream_t& stream) 
{
	cutilSafeCall(cudaGraphicsMapResources(1, &grSite, stream));
	cudaArray *in_array; 
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, grSite, 0, 0));
	
	cutilSafeCall(cudaBindTextureToArray(inTex, in_array));
    inTex.addressMode[0] = cudaAddressModeClamp;
    inTex.addressMode[1] = cudaAddressModeClamp;
    inTex.filterMode = cudaFilterModePoint;
    inTex.normalized = false;

// 	cutilSafeCall(cudaMemcpy2DFromArray(pReadBackValues, screenwidth * sizeof(float) * 4, in_array, sizeof(float) * 4, 1, screenwidth * sizeof(float) * 4, iSiteTextureHeight, cudaMemcpyDeviceToHost));

	int nblock0 = nsite;

	real* buf_s_r;
	cudaMalloc(&buf_s_r, nsite * sizeof(real));

	int m = lbfgsbcuda::log2Up(nblock0);
	int nblock1 = lbfgsbcuda::iDivUp2(nblock0, m);

	real* output = (nblock1 == 1) ? f : buf_s_r;

	dynamicCall(EnergyfKer, m, nblock1, 1, stream, (g, output, w, nsite));
// 	EnergyfKer<<<dim3(nblock1), dim3(512)>>>
// 		(g, output, w, nsite);

/*	float test[8] = {0};*/
/*	cudaMemcpy(test, output, 8 * sizeof(float), cudaMemcpyDeviceToHost);*/

						
	nblock0 = nblock1;
	while(nblock0 > 1) {

		nblock1 = lbfgsbcuda::iDivUp2(nblock0, m);

		real* input = output;

		output = (nblock1 == 1) ? f : (output + nblock0);
		dynamicCall(EnergyfKer1, m, nblock1, 1, stream, (nblock0, input, output));

		nblock0 = nblock1;
	}
/*#if (SYNC_LEVEL > 0)*/
/*	cutilSafeCall(cudaThreadSynchronize());*/
/*#endif*/
	cudaFree(buf_s_r);

	cutilSafeCall(cudaGraphicsUnmapResources(1, &grSite, stream));
}

__global__ void convertSiteKer(real* input, float* output, int nsite, real screenwidth_105, real screenwidth_1051) 
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= nsite)
		return;

	output[i] = input[i] * screenwidth_105 + screenwidth_1051;
}

__global__ void initSiteKer(float* input, int stride, real* output, int* nbd, real* l, real* u, int nsite, real screenwidth_105, real screenwidth_1051) 
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i >= nsite)
		return;

	const int i2 = (i >> 1) * stride + (i & 1);

	output[i] = input[i2] * screenwidth_105 + screenwidth_1051;
	nbd[i] = 2;
	l[i] = -1.0;
	u[i] = 1.0;
}

extern void ConvertSites(real* x, cudaGraphicsResource_t gr, int nsite, int screenw, const cudaStream_t& stream) 
{
	float *dptr;

	cutilSafeCall(cudaGraphicsMapResources(1, &gr, stream));
    
	size_t num_bytes; 
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, gr));

	real a1 = (screenw - 1.0) * 0.5;
	real a2 = a1 + 1.0;

	convertSiteKer<<<lbfgsbcuda::iDivUp(nsite, 512), 512, 0, stream>>>
		(x, dptr, nsite, a1, a2);

	cutilSafeCall(cudaGraphicsUnmapResources(1, &gr, 0));
}

extern void InitSites(real* x, float* init_sites, int stride, int* nbd, real* l, real* u, int nsite, int screenw) 
{
	float *dptr;
    
	real a1 = 1 / real(screenw - 1) * 2.0;
	real a2 = -a1 - 1.0;

	initSiteKer<<<lbfgsbcuda::iDivUp(nsite, 512), 512>>>
		(init_sites, stride, x, nbd, l, u, nsite, a1, a2);
}