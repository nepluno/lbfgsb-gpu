/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CULBFGSB_LBFGSBCUDA_H_
#define CULBFGSB_LBFGSBCUDA_H_

#include <cublas_v2.h>

#include <algorithm>
#include <functional>

#include "culbfgsb/cutil_inline.h"

#define MODU8

#ifdef MODU8
#define Modular(a, b) (a & 7)
#else
#define Modular(a, b) (a % b)
#endif

#define dynamicCall3D(f, bx, type, nblockx, nblocky, nblockz, st, var)         \
  {                                                                            \
    switch (bx) {                                                              \
      case 9:                                                                  \
        f<512, type>                                                           \
            <<<dim3(nblockx, nblocky, nblockz), dim3(512), 0, st>>> var;       \
        break;                                                                 \
      case 8:                                                                  \
        f<256, type>                                                           \
            <<<dim3(nblockx, nblocky, nblockz), dim3(256), 0, st>>> var;       \
        break;                                                                 \
      case 7:                                                                  \
        f<128, type>                                                           \
            <<<dim3(nblockx, nblocky, nblockz), dim3(128), 0, st>>> var;       \
        break;                                                                 \
      default:                                                                 \
        f<64, type><<<dim3(nblockx, nblocky, nblockz), dim3(64), 0, st>>> var; \
        break;                                                                 \
    }                                                                          \
  }

#define dynamicCall(f, bx, type, nblockx, nblocky, st, var) \
  dynamicCall3D(f, bx, type, nblockx, nblocky, 1, st, var)

#define inv9l2 0.36910312165415137198559104772104
#define invl2 3.3219280948873623478703194294894

#define SYNC_LEVEL 2

#define maxr(a, b) fmax(a, b)
#define minr(a, b) fmin(a, b)
#define absr(a) fabs(a)
#define sqrtr(a) sqrt(a)
#define rsqrtr(a) rsqrt(a)

template <typename real>
inline cublasStatus_t cublasRtrsm(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const real* alpha, const real* A, int lda,
                                  real* B, int ldb) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <>
inline cublasStatus_t cublasRtrsm<double>(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const double* alpha, const double* A, int lda, double* B, int ldb) {
  return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

template <>
inline cublasStatus_t cublasRtrsm<float>(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const float* alpha, const float* A, int lda, float* B, int ldb) {
  return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

template <typename real>
inline cublasStatus_t cublasRtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n, const real* A,
    int lda, real* x, int incx) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <>
inline cublasStatus_t cublasRtrsv<float>(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx) {
  return cublasStrsv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

template <>
inline cublasStatus_t cublasRtrsv<double>(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, int n,
                                          const double* A, int lda, double* x,
                                          int incx) {
  return cublasDtrsv(handle, uplo, trans, diag, n, A, lda, x, incx);
}

template <typename real>
inline cublasStatus_t cublasRdot(cublasHandle_t handle, int n, const real* x,
                                 int incx, const real* y, int incy,
                                 real* result) {
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

template <>
inline cublasStatus_t cublasRdot<float>(cublasHandle_t handle, int n,
                                        const float* x, int incx,
                                        const float* y, int incy,
                                        float* result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}

template <>
inline cublasStatus_t cublasRdot<double>(cublasHandle_t handle, int n,
                                         const double* x, int incx,
                                         const double* y, int incy,
                                         double* result) {
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

#ifdef _DEBUG
#define debugSync() cutilSafeCall(cudaDeviceSynchronize())
#else
#define debugSync()
#endif

#include "culbfgsb.h"

namespace lbfgsbcuda {

inline int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

inline int iDivUp2(int a, int b) {
  int c = a >> b;
  return (a > (c << b)) ? (c + 1) : c;
}

inline int log2Up(int n) {
  double lnb = log10(static_cast<double>(n));
  double nker = ceil(lnb * inv9l2);
  int m = static_cast<int>(ceil(lnb * invl2 / nker));
  m = std::max(6, m);
  return m;
}

inline void memSet(void* p1, int _val, size_t length) {
  memset(p1, _val, length);
}

inline int fmaxf(int a, int b) { return a > b ? a : b; }

inline int fminf(int a, int b) { return a > b ? b : a; }

template <typename real>
inline real fmaxf(real a, real b) { return a > b ? a : b; }

template <typename real>
inline real fminf(real a, real b) { return a > b ? b : a; }

template <typename real>
inline void vmove_vm(real* dst, const real* src, int ds, int de, int sr,
                     int scs, int spitch) {
  int n = de - ds + 1;
  for (int i = 0; i < n; i++) {
    dst[ds + i] = src[(scs + i) * spitch + sr];
  }
}

template <typename real>
inline void vmove_mv(real* dst, const real* src, int dr, int dcs, int dce,
                     int ss, int dpitch) {
  int n = dce - dcs + 1;
  for (int i = 0; i < n; i++) {
    dst[(dcs + i) * dpitch + dr] = src[ss + i];
  }
}

template <typename real>
inline void vadd_vm(real* dst, const real* src, int ds, int de, int sr, int scs,
                    int spitch, real& alpha) {
  int n = de - ds + 1;
  for (int i = 0; i < n; i++) {
    dst[ds + i] += src[(scs + i) * spitch + sr] * alpha;
  }
}

template <typename real>
inline void vmul_v(real* dst, int ds, int de, const real& alpha) {
  int n = de - ds + 1;
  for (int i = 0; i < n; i++) {
    dst[ds + i] *= alpha;
  }
}

template <typename real>
inline void vadd_vv(real* dst, const real* src, int ds, int de, int sr,
                    real& alpha) {
  vadd_vm(dst, src, ds, de, sr, 0, 1, alpha);
}

template <typename real>
inline real vdot_vm(const real* dst, const real* src, int ds, int de, int sr,
                    int scs, int spitch) {
  int n = de - ds + 1;
  real r = 0;
  for (int i = 0; i < n; i++) {
    r += dst[ds + i] * src[(scs + i) * spitch + sr];
  }
  return r;
}

template <typename real>
inline real vdot_mm(const real* dst, const real* src, int dr, int dcs, int dce,
                    int dpitch, int sr, int scs, int spitch) {
  int n = dce - dcs + 1;
  real r = 0;
  for (int i = 0; i < n; i++) {
    r += dst[(dcs + i) * dpitch + dr] * src[(scs + i) * spitch + sr];
  }
  return r;
}

template <typename real>
inline real vdot_vv(const real* dst, const real* src, int ds, int de, int sr) {
  return vdot_vm(dst, src, ds, de, sr, 0, 1);
}

namespace cuda {
template <typename real>
bool checkAvailabilty(const LBFGSB_CUDA_STATE<real>& state);

template <typename real>
inline void CheckBuffer(const real* q, int stride, int total) {
#ifdef _DEBUG
  if (stride <= 0 || total <= 0) return;
  int h = iDivUp(total, stride);
  int wh = h * stride;
  real* hq = new real[wh];
  memset(hq, 0, sizeof(real) * wh);
  cutilSafeCall(
      cudaMemcpy(hq, q, total * sizeof(real), cudaMemcpyDeviceToHost));
  delete[] hq;
#endif
}

inline void CheckBuffer_int(const int* q, int stride, int total) {
#ifdef _DEBUG
  if (stride <= 0 || total <= 0) return;
  int h = iDivUp(total, stride);
  int wh = h * stride;
  int* hq = new int[wh];
  memset(hq, 0, sizeof(int) * wh);
  cutilSafeCall(cudaMemcpy(hq, q, total * sizeof(int), cudaMemcpyDeviceToHost));
  delete[] hq;
#endif
}

template <typename real>
void lbfgsbminimize(const int& n, const LBFGSB_CUDA_STATE<real>& state,
                    const LBFGSB_CUDA_OPTION<real>& option, real* x, const int* nbd,
                    const real* l, const real* u, LBFGSB_CUDA_SUMMARY<real>& summary);

template <typename real>
void lbfgsbactive(const int& n, const real* l, const real* u, const int* nbd,
                  real* x, int* iwhere);

template <typename real>
void lbfgsbbmv(const int& m, const real* sy, real* wt, const int& col,
               const int& iPitch, const real* v, real* p,
               cublasHandle_t cublas_handle, const cudaStream_t& stream,
               int& info);

template <typename real>
void lbfgsbcauchy(const int& n, const real* x, const real* l, const real* u,
                  const int* nbd, const real* g, real* t, real* xcp, real* xcpb,
                  const int& m, const real* wy, const real* ws, const real* sy,
                  const int iPitch, real* wt, const real& theta, const int& col,
                  const int& head, real* p, real* c, real* v, int& nint,
                  const real& sbgnrm, real* buf_s_r, real* buf_array_p,
                  int* iwhere, const int& iPitch_normal,
                  const real& machinemaximum, cublasHandle_t cublas_handle,
                  const cudaStream_t* streamPool, int& info);

template <typename real>
void lbfgsbcmprlb(const int& n, const int& m, const real* x, const real* g,
                  const real* ws, const real* wy, const real* sy, real* wt,
                  const real* z, real* r, real* wa, const int* index,
                  const real& theta, const int& col, const int& head,
                  const int& nfree, const bool& cnstnd, int& info,
                  real* workvec, real* workvec2, const int& iPitch,
                  cublasHandle_t cublas_handle, const cudaStream_t& stream);

template <typename real>
void lbfgsbformk(const int& n, const int& nsub, const int* ind,
                 const int& nenter, const int& ileave, const int* indx2,
                 const int& iupdat, const bool& updatd, real* wn, real* wn1,
                 const int& m, const real* ws, const real* wy, const real* sy,
                 const real& theta, const int& col, const int& head, int& info,
                 real* workvec, real* workmat, real* buf_array_p,
                 real* buf_array_super, const int& iPitch_wn,
                 const int& iPitch_ws, const int& iPitch_super,
                 const int& iPitch_normal, const real& machineepsilon,
                 cublasHandle_t cublas_handle, const cudaStream_t* streamPool);

template <typename real>
void lbfgsbformt(const int& m, real* wt, const real* sy, const real* ss,
                 const int& col, const real& theta, int& info,
                 const int& iPitch, const real& machineepsilon,
                 const cudaStream_t* streamPool);

template <typename real>
void lbfgsblnsrlb(const int& n, const real* l, const real* u, const int* nbd,
                  real* x, const real& f, real& fold, real& gd, real& gdold,
                  const real* g, real* d, real* r, real* t, const real* z,
                  real& stp, real& dnrm, real& dtd, real& xstep, real& stpmx,
                  const real& stpscaling, const int& iter, int& ifun,
                  int& iback, int& nfgv, int& info, int& task, int& csave,
                  int* isave, real* dsave, real* buf_s_r,
                  cublasHandle_t cublas_handle, const cudaStream_t* streamPool);

template <typename real>
void lbfgsbmatupdsub(const int& n, const int& m, real* wy, real* sy,
                     const real* r, const real* d, int& itail,
                     const int& iupdat, int& col, int& head, const real& dr,
                     const int& iPitch0, const int& iPitch_i,
                     const int& iPitch_j);

template <typename real>
void lbfgsbmatupd(const int& n, const int& m, real* ws, real* wy, real* sy,
                  real* ss, const real* d, const real* r, int& itail,
                  const int& iupdat, int& col, int& head, real& theta,
                  const real& rr, const real& dr, const real& stp,
                  const real& dtd, const int& iPitch, real* buf_array_p,
                  const int& iPitch_normal, const cudaStream_t* streamPool);

template <typename real>
void lbfgsbprojgr(const int& n, const real* l, const real* u, const int* nbd,
                  const real* x, const real* g, real* buf_n, real* sbgnrm_h,
                  real* sbgnrm_d, const real& machinemaximum,
                  const cudaStream_t& stream);

template <typename real>
void lbfgsbsubsm(const int& n, const int& m, const int& nsub, const int* ind,
                 const real* l, const real* u, const int* nbd, real* x, real* d,
                 real* xp, const real* ws, const real* wy, const real& theta,
                 const real* xx, const real* gg, const int& col,
                 const int& head, real* wv, real* wn, int& info,
                 const int& iPitch_wn, const int& iPitch_ws, real* buf_array_p,
                 real* buf_s_r, int* bufi_s_r, const int& iPitch_normal,
                 cublasHandle_t cublas_handle, const cudaStream_t& stream);

template <typename real>
void lbfgsbdcsrch(const real& f, const real& g, real& stp, const real& ftol,
                  const real& gtol, const real& xtol, const real& stpmin,
                  const real& stpmax, const real& stpscaling, int& task,
                  int* isave, real* dsave);

template <typename real>
void lbfgsbdcstep(real& stx, real& fx, real& dx, real& sty, real& fy, real& dy,
                  real& stp, const real& fp, const real& dp, bool& brackt,
                  const real& stpmin, const real& stpmax);

template <typename real>
bool lbfgsbdpofa(real* a, const int& n, const int& iPitch);

template <typename real>
void lbfgsbdtrsl(real* t, const int& n, const int& iPitch, real* b,
                 const int& job, int& info);

namespace minimize {
template <typename real>
void vdot_vv(int n, const real* g, const real* d, real& gd,
             cublasHandle_t cublas_handle, const cudaStream_t& stream = NULL);

template <typename real>
void vmul_v(const int n, real* d, const real stp,
            const cudaStream_t& stream = NULL);

template <typename real>
void vsub_v(const int n, const real* a, const real* b, real* c,
            const cudaStream_t& stream = NULL);

template <typename real>
void vdiffxchg_v(const int n, real* xdiff, real* xold, const real* x,
                 const cudaStream_t& stream = NULL);
};  // namespace minimize
namespace active {
template <typename real>
void prog0(const int& n, const real* l, const real* u, const int* nbd, real* x,
           int* iwhere);
};
namespace projgr {
template <typename real>
void prog0(const int& n, const real* l, const real* u, const int* nbd,
           const real* x, const real* g, real* buf_n, real* sbgnrm,
           real* sbgnrm_dev, const real machinemaximum,
           const cudaStream_t& stream);
};
namespace cauchy {
template <typename real>
void prog0(const int& n, const real* x, const real* l, const real* u,
           const int* nbd, const real* g, real* t, real* xcp, real* xcpb,
           const int& m, const real* wy, const real* ws, const real* sy,
           const int iPitch, real* wt, const real& theta, const int& col,
           const int& head, real* p, real* c, real* v, int& nint,
           const real& sbgnrm, real* buf_s_r, real* buf_array_p, int* iwhere,
           const int& iPitch_normal, const real& machinemaximum,
           cublasHandle_t cublas_handle, const cudaStream_t* streamPool);
};
namespace freev {
void prog0(const int& n, int& nfree, int* index, int& nenter, int& ileave,
           int* indx2, const int* iwhere, bool& wrk, const bool& updatd,
           const bool& cnstnd, const int& iter, int* temp_ind1, int* temp_ind2,
           int* temp_ind3, int* temp_ind4);

};
namespace formk {
template <typename real>
void prog0(real* wn1, int m, int iPitch_wn, const cudaStream_t* streamPool);

template <typename real>
void prog1(const int n, const int nsub, const int ipntr, const int* ind,
           real* wn1, real* buf_array_p, const real* ws, const real* wy,
           const int head, const int m, const int col, const int iPitch_ws,
           const int iPitch_wn, const int iPitch_normal,
           const cudaStream_t* streamPool);

template <typename real>
void prog2(real* wn1, const int col, const int m, const int iPitch_wn,
           const cudaStream_t* streamPool);

template <typename real>
void prog3(const int* ind, const int jpntr, const int head, const int m,
           const int col, const int n, const int nsub, const int iPitch_ws,
           const int iPitch_wn, const int jy, const real* ws, const real* wy,
           real* buf_array_p, real* wn1, const int iPitch_normal,
           const cudaStream_t* streamPool);

template <typename real>
void prog31(const int* indx2, const int head, const int m, const int upcl,
            const int col, const int nenter, const int ileave, const int n,
            const int iPitch_ws, const int iPitch_wn, const real* wy,
            real* buf_array_sup, real* wn1, const real scal,
            const int iPitch_super, const cudaStream_t* streamPool);

template <typename real>
void prog32(const int* indx2, const int head, const int m, const int upcl,
            const int nenter, const int ileave, const int n,
            const int iPitch_ws, const int iPitch_wn, const real* wy,
            const real* ws, real* buf_array_sup, real* wn1,
            const int iPitch_super, const cudaStream_t* streamPool);

template <typename real>
void prog4(const int col, const int iPitch_wn, const int iPitch_ws, const int m,
           const real* wn1, const real theta, const real* sy, real* wn,
           const cudaStream_t* streamPool);

template <typename real>
void prog5(const int col, const int iPitch_wn, real* wn,
           cublasHandle_t cublas_handle, const cudaStream_t* streamPool);
};  // namespace formk
namespace cmprlb {
template <typename real>
void prog0(int n, real* r, const real* g, const cudaStream_t& stream);

template <typename real>
void prog1(int nfree, const int* index, const int col, const int head,
           const int m, const int iPitch, const real* wa, const real* wy,
           const real* ws, const real theta, const real* z, const real* x,
           const real* g, real* r, const cudaStream_t& stream);
};  // namespace cmprlb
namespace subsm {
template <typename real>
void prog0(const int n, const int* ind, const int head, const int m,
           const int col, const int iPitch_ws, real* buf_array_p,
           const real* wy, const real* ws, const real* d, real* wv,
           const real theta, const int iPitch_normal,
           const cudaStream_t& stream);

template <typename real>
void prog1(real* wn, int col, int iPitch_wn, real* wv,
           cublasHandle_t cublas_handle, const cudaStream_t& stream);

template <typename real>
void prog2(int nsub, const int* ind, const int col, const int head, const int m,
           const int iPitch, const real* wv, const real* wy, const real* ws,
           const real theta, real* d, const cudaStream_t& stream);

template <typename real>
void prog21(int n, int nsub, const int* ind, const real* d, real* x,
            const real* l, const real* u, const int* nbd, const real* xx,
            const real* gg, real* buf_n_r, real* pddp,
            const cudaStream_t& stream);

template <typename real>
void prog3(int nsub, const int* ind, real* d, const int* nbd, real* buf_s_r,
           int* bufi_s_r, real* x, const real* u, const real* l,
           const cudaStream_t& stream);
};  // namespace subsm
namespace lnsrlb {
template <typename real>
void prog0(int n, const real* d, const int* nbd, const real* u, const real* x,
           const real* l, real* buf_s_r, real* stpmx_host, real* stpmx_dev,
           const cudaStream_t& stream);

template <typename real>
void prog2(int n, real* x, real* d, const real* t, const real stp,
           const cudaStream_t& stream);
};  // namespace lnsrlb
namespace matupd {
template <typename real>
void prog0(const int& n, const int& m, real* wy, real* sy, const real* r,
           const real* d, int& itail, const int& iupdat, int& col, int& head,
           const real& dr, const int& iPitch0, const int& iPitch_i,
           const int& iPitch_j, real* buf_array_p, const int& iPitch_normal,
           cudaStream_t st);
};
namespace formt {
template <typename real>
void prog01(const int col, const real* sy, const real* ss, real* wt,
            const int iPitch, const real theta, const cudaStream_t& stream);
};
namespace bmv {
template <typename real>
void prog0(const real* sy, const int& col, const int& iPitch, const real* v,
           real* p, const cudaStream_t& st);

template <typename real>
void prog1(const real* wt, const int& col, const int& iPitch, const real* v,
           real* p, cublasContext* cublas_handle, const cudaStream_t& st);

template <typename real>
void prog2(const real* sy, real* wt, const int& col, const int& iPitch,
           const real* v, real* p, const cudaStream_t& st);
};  // namespace bmv
namespace dpofa {
template <typename real>
void prog0(real* m, int n, int pitch, int boffset, const real machineepsilon,
           const cudaStream_t& st);
};

template <class T>
inline int memAlloc(T** ptr, size_t length) {
  void* ptr_;
  cudaMalloc(&ptr_, length * sizeof(T));
  *ptr = (T*)ptr_;
  if (ptr_ != NULL)
    return 0;
  else
    return -1;
}

template <class T>
inline int memAllocHost(T** ptr, T** ptr_dev, size_t length) {
  void* ptr_;
  cudaHostAlloc(&ptr_, length * sizeof(T), cudaHostAllocMapped);
  if (ptr_dev) cudaHostGetDevicePointer((void**)ptr_dev, ptr_, 0);
  *ptr = (T*)ptr_;

  memset(*ptr, 0, length);

  if (ptr_ != NULL)
    return 0;
  else
    return -1;
}

template <class T>
inline int memAllocPitch(T** ptr, size_t lx, size_t ly, size_t* pitch) {
  int bl = lx * sizeof(T);
  void* ptr_;
  bl = iAlignUp(bl, 32);
  if (pitch) *pitch = bl / sizeof(T);
  int length = bl * ly;
  cudaMalloc(&ptr_, length);
  *ptr = (T*)ptr_;

  cudaMemset(*ptr, 0, length);

  if (ptr_ != NULL)
    return 0;
  else
    return -1;
}

inline void memFree(void* ptr) { cudaFree(ptr); }

inline void memFreeHost(void* ptr) { cudaFreeHost(ptr); }

inline void memCopy(void* p1, const void* p2, size_t length, int type = 0) {
  // memcpy(p1, p2, length);
  cudaMemcpy(p1, p2, length, (cudaMemcpyKind)type);
}

inline void memCopyAsync(void* p1, const void* p2, size_t length, int type = 0,
                         const cudaStream_t stream = NULL) {
  // memcpy(p1, p2, length);
  cudaMemcpyAsync(p1, p2, length, (cudaMemcpyKind)type, stream);
}
};      // namespace cuda
};      // namespace lbfgsbcuda
#endif  // CULBFGSB_LBFGSBCUDA_H_
