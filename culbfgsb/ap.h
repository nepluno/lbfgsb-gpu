/********************************************************************
AP Library version 1.2.1

Copyright (c) 2003-2007, Sergey Bochkanov (ALGLIB project).
See www.alglib.net or alglib.sources.ru for details.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef CULBFGSB_AP_H_
#define CULBFGSB_AP_H_

#include <math.h>
#include <stdlib.h>

#include <string>

/********************************************************************
Array bounds check
********************************************************************/
#define AP_ASSERT

#ifndef AP_ASSERT     //
#define NO_AP_ASSERT  // This code avoids definition of the
#endif                // both AP_ASSERT and NO_AP_ASSERT symbols
#ifdef NO_AP_ASSERT   //
#ifdef AP_ASSERT      //
#undef NO_AP_ASSERT   //
#endif                //
#endif                //

/********************************************************************
Current environment.
********************************************************************/
#ifndef AP_WIN32
#ifndef AP_UNKNOWN
#define AP_UNKNOWN
#endif
#endif
#ifdef AP_WIN32
#ifdef AP_UNKNOWN
#error Multiple environments are declared!
#endif
#endif

/********************************************************************
This symbol is used for debugging. Do not define it and do not remove
comments.
********************************************************************/
//#define UNSAFE_MEM_COPY

/********************************************************************
Namespace of a standard library AlgoPascal.
********************************************************************/
namespace ap {

/********************************************************************
Service routines:
    amalloc - allocates an aligned block of size bytes
    afree - frees block allocated by amalloc
    vlen - just alias for n2-n1+1
********************************************************************/
void* amalloc(size_t size, size_t alignment);
void afree(void* block);
int vlen(int n1, int n2);

/********************************************************************
Exception class.
********************************************************************/
class ap_error {
 public:
  ap_error(){};
  ap_error(const char* s) { msg = s; };

  std::string msg;

  static void make_assertion(bool bClause) {
    if (!bClause) throw ap_error();
  };
  static void make_assertion(bool bClause, const char* msg) {
    if (!bClause) throw ap_error(msg);
  };

 private:
};

/********************************************************************
Templates for vector operations
********************************************************************/
#include "apvt.h"

/********************************************************************
BLAS functions
********************************************************************/
template <typename real>
real vdotproduct(const real* v1, const real* v2, int N) {
  return _vdotproduct<real>(v1, v2, N);
}

template <typename real>
void vmove(real* vdst, const real* vsrc, int N) {
  _vmove<real>(vdst, vsrc, N);
}

template <typename real>
void vmoveneg(real* vdst, const real* vsrc, int N) {
  _vmoveneg<real>(vdst, vsrc, N);
}

template <typename real>
void vmove(real* vdst, const real* vsrc, int N, real alpha) {
  _vmove<real, real>(vdst, vsrc, N, alpha);
}

template <typename real>
void vadd(real* vdst, const real* vsrc, int N) {
  _vadd<real>(vdst, vsrc, N);
}

template <typename real>
void vadd(real* vdst, const real* vsrc, int N, real alpha) {
  _vadd<real, real>(vdst, vsrc, N, alpha);
}

template <typename real>
void vsub(real* vdst, const real* vsrc, int N) {
  _vsub<real>(vdst, vsrc, N);
}

template <typename real>
void vsub(real* vdst, const real* vsrc, int N, real alpha) {
  _vsub<real, real>(vdst, vsrc, N, alpha);
}

template <typename real>
void vmul(real* vdst, int N, real alpha) {
  _vmul<real, real>(vdst, N, alpha);
}

/********************************************************************
Template of a dynamical one-dimensional array
********************************************************************/
template <class T, bool Aligned = false>
class template_1d_array {
 public:
  template_1d_array() {
    m_Vec = 0;
    m_iVecSize = 0;
    m_iLow = 0;
    m_iHigh = -1;
  };

  ~template_1d_array() {
    if (m_Vec) {
      if (Aligned)
        ap::afree(m_Vec);
      else
        delete[] m_Vec;
    }
  };

  template_1d_array(const template_1d_array& rhs) {
    m_Vec = 0;
    m_iVecSize = 0;
    m_iLow = 0;
    m_iHigh = -1;
    if (rhs.m_iVecSize != 0)
      setcontent(rhs.m_iLow, rhs.m_iHigh, rhs.getcontent());
  };

  template_1d_array(T* rhs, int iLow, int iHigh) {
    m_iLow = iLow;
    m_iHigh = iHigh;
    m_iVecSize = iHigh - iLow + 1;
    m_Vec = rhs;
  }

  template_1d_array(const T* rhs, int iLow, int iHigh) : template_1d_array() {
    setbounds(iLow, iHigh);
    memcpy(m_Vec, rhs, m_iVecSize);
  }

  const template_1d_array& operator=(const template_1d_array& rhs) {
    if (this == &rhs) return *this;

    if (rhs.m_iVecSize != 0)
      setcontent(rhs.m_iLow, rhs.m_iHigh, rhs.getcontent());
    else {
      m_Vec = 0;
      m_iVecSize = 0;
      m_iLow = 0;
      m_iHigh = -1;
    }
    return *this;
  };

  const T& operator()(int i) const {
#ifndef NO_AP_ASSERT
    ap_error::make_assertion(i >= m_iLow && i <= m_iHigh);
#endif
    return m_Vec[i - m_iLow];
  };

  T& operator()(int i) {
#ifndef NO_AP_ASSERT
    ap_error::make_assertion(i >= m_iLow && i <= m_iHigh);
#endif
    return m_Vec[i - m_iLow];
  };

  void setbounds(int iLow, int iHigh) {
    if (m_Vec) {
      if (Aligned)
        ap::afree(m_Vec);
      else
        delete[] m_Vec;
    }
    m_iLow = iLow;
    m_iHigh = iHigh;
    m_iVecSize = iHigh - iLow + 1;
    if (Aligned)
      m_Vec = (T*)ap::amalloc(m_iVecSize * sizeof(T), 16);
    else
      m_Vec = new T[m_iVecSize];
  };

  void setcontent(int iLow, int iHigh, const T* pContent) {
    setbounds(iLow, iHigh);
    for (int i = 0; i < m_iVecSize; i++) m_Vec[i] = pContent[i];
  };

  T* getcontent() { return m_Vec; };

  const T* getcontent() const { return m_Vec; };

  int getlowbound(int iBoundNum = 0) const { return m_iLow; };

  int gethighbound(int iBoundNum = 0) const { return m_iHigh; };

  raw_vector<T> getvector(int iStart, int iEnd) {
    if (iStart > iEnd || wrongIdx(iStart) || wrongIdx(iEnd))
      return raw_vector<T>(0, 0, 1);
    else
      return raw_vector<T>(m_Vec + iStart - m_iLow, iEnd - iStart + 1, 1);
  };

  const_raw_vector<T> getvector(int iStart, int iEnd) const {
    if (iStart > iEnd || wrongIdx(iStart) || wrongIdx(iEnd))
      return const_raw_vector<T>(0, 0, 1);
    else
      return const_raw_vector<T>(m_Vec + iStart - m_iLow, iEnd - iStart + 1, 1);
  };

 private:
  bool wrongIdx(int i) const { return i < m_iLow || i > m_iHigh; };

  T* m_Vec;
  long m_iVecSize;
  long m_iLow, m_iHigh;
};

/********************************************************************
Template of a dynamical two-dimensional array
********************************************************************/
template <class T, bool Aligned = false>
class template_2d_array {
 public:
  template_2d_array() {
    m_Vec = 0;
    m_iVecSize = 0;
    m_iLow1 = 0;
    m_iHigh1 = -1;
    m_iLow2 = 0;
    m_iHigh2 = -1;
  };

  ~template_2d_array() {
    if (m_Vec) {
      if (Aligned)
        ap::afree(m_Vec);
      else
        delete[] m_Vec;
    }
  };

  template_2d_array(const template_2d_array& rhs) {
    m_Vec = 0;
    m_iVecSize = 0;
    m_iLow1 = 0;
    m_iHigh1 = -1;
    m_iLow2 = 0;
    m_iHigh2 = -1;
    if (rhs.m_iVecSize != 0) {
      setbounds(rhs.m_iLow1, rhs.m_iHigh1, rhs.m_iLow2, rhs.m_iHigh2);
      for (int i = m_iLow1; i <= m_iHigh1; i++)
        vmove(&(operator()(i, m_iLow2)), &(rhs(i, m_iLow2)),
              m_iHigh2 - m_iLow2 + 1);
    }
  };
  const template_2d_array& operator=(const template_2d_array& rhs) {
    if (this == &rhs) return *this;

    if (rhs.m_iVecSize != 0) {
      setbounds(rhs.m_iLow1, rhs.m_iHigh1, rhs.m_iLow2, rhs.m_iHigh2);
      for (int i = m_iLow1; i <= m_iHigh1; i++)
        vmove(&(operator()(i, m_iLow2)), &(rhs(i, m_iLow2)),
              m_iHigh2 - m_iLow2 + 1);
    } else {
      m_Vec = 0;
      m_iVecSize = 0;
      m_iLow1 = 0;
      m_iHigh1 = -1;
      m_iLow2 = 0;
      m_iHigh2 = -1;
    }
    return *this;
  };

  const T& operator()(int i1, int i2) const {
#ifndef NO_AP_ASSERT
    ap_error::make_assertion(i1 >= m_iLow1 && i1 <= m_iHigh1);
    ap_error::make_assertion(i2 >= m_iLow2 && i2 <= m_iHigh2);
#endif
    return m_Vec[m_iConstOffset + i2 + i1 * m_iLinearMember];
  };

  T& operator()(int i1, int i2) {
#ifndef NO_AP_ASSERT
    ap_error::make_assertion(i1 >= m_iLow1 && i1 <= m_iHigh1);
    ap_error::make_assertion(i2 >= m_iLow2 && i2 <= m_iHigh2);
#endif
    return m_Vec[m_iConstOffset + i2 + i1 * m_iLinearMember];
  };

  void setbounds(int iLow1, int iHigh1, int iLow2, int iHigh2) {
    if (m_Vec) {
      if (Aligned)
        ap::afree(m_Vec);
      else
        delete[] m_Vec;
    }
    int n1 = iHigh1 - iLow1 + 1;
    int n2 = iHigh2 - iLow2 + 1;
    m_iVecSize = n1 * n2;
    if (Aligned) {
      // if( n2%2!=0 )
      while ((n2 * sizeof(T)) % 16 != 0) {
        n2++;
        m_iVecSize += n1;
      }
      m_Vec = (T*)ap::amalloc(m_iVecSize * sizeof(T), 16);
    } else
      m_Vec = new T[m_iVecSize];
    m_iLow1 = iLow1;
    m_iHigh1 = iHigh1;
    m_iLow2 = iLow2;
    m_iHigh2 = iHigh2;
    m_iConstOffset = -m_iLow2 - m_iLow1 * n2;
    m_iLinearMember = n2;
  };

  void setcontent(int iLow1, int iHigh1, int iLow2, int iHigh2,
                  const T* pContent) {
    setbounds(iLow1, iHigh1, iLow2, iHigh2);
    for (int i = m_iLow1; i <= m_iHigh1;
         i++, pContent += m_iHigh2 - m_iLow2 + 1)
      vmove(&(operator()(i, m_iLow2)), pContent, m_iHigh2 - m_iLow2 + 1);
  };

  int getlowbound(int iBoundNum) const {
    return iBoundNum == 1 ? m_iLow1 : m_iLow2;
  };

  int gethighbound(int iBoundNum) const {
    return iBoundNum == 1 ? m_iHigh1 : m_iHigh2;
  };

  raw_vector<T> getcolumn(int iColumn, int iRowStart, int iRowEnd) {
    if ((iRowStart > iRowEnd) || wrongColumn(iColumn) || wrongRow(iRowStart) ||
        wrongRow(iRowEnd))
      return raw_vector<T>(0, 0, 1);
    else
      return raw_vector<T>(&((*this)(iRowStart, iColumn)),
                           iRowEnd - iRowStart + 1, m_iLinearMember);
  };

  raw_vector<T> getrow(int iRow, int iColumnStart, int iColumnEnd) {
    if ((iColumnStart > iColumnEnd) || wrongRow(iRow) ||
        wrongColumn(iColumnStart) || wrongColumn(iColumnEnd))
      return raw_vector<T>(0, 0, 1);
    else
      return raw_vector<T>(&((*this)(iRow, iColumnStart)),
                           iColumnEnd - iColumnStart + 1, 1);
  };

  const_raw_vector<T> getcolumn(int iColumn, int iRowStart, int iRowEnd) const {
    if ((iRowStart > iRowEnd) || wrongColumn(iColumn) || wrongRow(iRowStart) ||
        wrongRow(iRowEnd))
      return const_raw_vector<T>(0, 0, 1);
    else
      return const_raw_vector<T>(&((*this)(iRowStart, iColumn)),
                                 iRowEnd - iRowStart + 1, m_iLinearMember);
  };

  const_raw_vector<T> getrow(int iRow, int iColumnStart, int iColumnEnd) const {
    if ((iColumnStart > iColumnEnd) || wrongRow(iRow) ||
        wrongColumn(iColumnStart) || wrongColumn(iColumnEnd))
      return const_raw_vector<T>(0, 0, 1);
    else
      return const_raw_vector<T>(&((*this)(iRow, iColumnStart)),
                                 iColumnEnd - iColumnStart + 1, 1);
  };

 private:
  bool wrongRow(int i) const { return i < m_iLow1 || i > m_iHigh1; };
  bool wrongColumn(int j) const { return j < m_iLow2 || j > m_iHigh2; };

  T* m_Vec;
  long m_iVecSize;
  long m_iLow1, m_iLow2, m_iHigh1, m_iHigh2;
  long m_iConstOffset, m_iLinearMember;
};

typedef template_1d_array<int> integer_1d_array;
typedef template_1d_array<double, true> real_1d_array;
typedef template_1d_array<float, true> float_1d_array;
typedef template_1d_array<bool> boolean_1d_array;

typedef template_2d_array<int> integer_2d_array;
typedef template_2d_array<double, true> real_2d_array;
typedef template_2d_array<float, true> float_2d_array;
typedef template_2d_array<bool> boolean_2d_array;

/********************************************************************
Constants and functions introduced for compatibility with AlgoPascal
********************************************************************/
extern const double machineepsilon;
extern const double maxrealnumber;
extern const double minrealnumber;

/********************************************************************
standard functions
********************************************************************/
template <typename real>
int sign(real x) {
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

template <typename real>
real randomreal() {
  int i = rand();
  while (i == RAND_MAX) i = rand();
  return real(i) / real(RAND_MAX);
}

int randominteger(int maxv);

template <typename real>
int round(real x) { return int(floor(x + 0.5)); }

template <typename real>
int trunc(real x) { return int(x > 0 ? floor(x) : ceil(x)); }

template <typename real>
int ifloor(real x) { return int(floor(x)); }

template <typename real>
int iceil(real x) { return int(ceil(x)); }

template <typename real>
real pi() { return 3.14159265358979323846; }

template <typename real>
real sqr(real x) { return x * x; }

int maxint(int m1, int m2);

int minint(int m1, int m2);

template <typename real>
real maxreal(real m1, real m2) {
  return m1 > m2 ? m1 : m2;
}

template <typename real>
real minreal(real m1, real m2) {
  return m1 > m2 ? m2 : m1;
}
};  // namespace ap

#endif // CULBFGSB_AP_H_
