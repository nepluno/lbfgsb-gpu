/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "examples/dsscfg/dsscfg_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

template <typename real>
void dsscfg_cpu(int const& nx, int const& ny, real* x, real& f,
                real* fgrad, real** assist_buffer, int task,
                real const& lambda) {
  real hx = 1.0 / static_cast<real>(nx + 1);
  real hy = 1.0 / static_cast<real>(ny + 1);
  real area = 0.5 * hx * hy;
  //
  //     Compute the standard starting point if task = 'XS'.
  //
  if (task == 'XS') {
    real temp1 = lambda / (lambda + 1.0);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        real temp = static_cast<real>(std::min(j, ny - j)) * hy;
        int k = nx * j + i;
        x[k] =
            temp1 *
            sqrt(std::min(static_cast<real>(std::min(i, nx - i)) * hx, temp));
      }
    }
    *assist_buffer = new real[nx * ny * 4];
    return;
  }
  //
  bool feval = task == 'F' || task == 'FG';
  bool geval = task == 'G' || task == 'FG';
  //
  //     Compute the function if task = 'F', the gradient if task = 'G', or
  //     both if task = 'FG'.
  //
  real fquad = 0.0;
  real fexp = 0.0;

  real* fgrad_r = NULL;
  real* fgrad_t = NULL;
  real* fgrad_l = NULL;
  real* fgrad_b = NULL;
  if (geval) {
    fgrad_r = *assist_buffer;
    fgrad_t = fgrad_r + nx * ny;
    fgrad_l = fgrad_t + nx * ny;
    fgrad_b = fgrad_l + nx * ny;

    memset(fgrad, 0, nx * ny * sizeof(real));
    memset(fgrad_r, 0, nx * ny * sizeof(real));
    memset(fgrad_t, 0, nx * ny * sizeof(real));
    memset(fgrad_l, 0, nx * ny * sizeof(real));
    memset(fgrad_b, 0, nx * ny * sizeof(real));
  }
  //
  //     Computation of the function and the gradient over the lower
  //     triangular elements.  The trapezoidal rule is used to estimate
  //     the integral of the exponential term.
  //
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int k = nx * j + i;
      real v = 0.0;
      real vr = 0.0;
      real vt = 0.0;
      v = x[k];
      if (i != nx - 1) {
        vr = x[k + 1];
      }
      if (j != ny - 1) {
        vt = x[k + nx];
      }
      real dvdx = (vr - v) / hx;
      real dvdy = (vt - v) / hy;
      real expv = exp(v);
      real expvr = exp(vr);
      real expvt = exp(vt);
      if (feval) {
        fquad += dvdx * dvdx + dvdy * dvdy;
        fexp = fexp - lambda * (expv + expvr + expvt) / 3.0;
      }
      if (geval) {
        fgrad[k] += -dvdx / hx - dvdy / hy - lambda * expv / 3.0;
        if (i != nx - 1) {
          fgrad_r[k + 1] = dvdx / hx - lambda * expvr / 3.0;
        }
        if (j != ny - 1) {
          fgrad_t[k + nx] = dvdy / hy - lambda * expvt / 3.0;
        }
      }
      //
      //     Computation of the function and the gradient over the upper
      //     triangular elements.  The trapezoidal rule is used to estimate
      //     the integral of the exponential term.
      //
      real vb = 0.0;
      real vl = 0.0;

      if (j != 0) {
        vb = x[k - nx];
      }
      if (i != 0) {
        vl = x[k - 1];
      }

      dvdx = (v - vl) / hx;
      dvdy = (v - vb) / hy;
      real expvb = exp(vb);
      real expvl = exp(vl);
      expv = exp(v);
      if (feval) {
        fquad += dvdx * dvdx + dvdy * dvdy;
        fexp = fexp - lambda * (expvb + expvl + expv) / 3.0;
      }
      if (geval) {
        if (j != 0) {
          fgrad_b[k - nx] = -dvdy / hy - lambda * expvb / 3.0;
        }
        if (i != 0) {
          fgrad_l[k - 1] = -dvdx / hx - lambda * expvl / 3.0;
        }
        fgrad[k] += dvdx / hx + dvdy / hy - lambda * expv / 3.0;
      }
    }
  }

  //
  //     Scale the result.
  //
  if (feval) {
    f = area * (.5 * fquad + fexp);
  }
  if (geval) {
    for (int k = 0; k < nx * ny; ++k) {
      fgrad[k] =
          (fgrad[k] + fgrad_b[k] + fgrad_l[k] + fgrad_r[k] + fgrad_t[k]) * area;
    }
  }
  //
}

#define INST_HELPER(real)                                                    \
  template void dsscfg_cpu<real>(int const& nx, int const& ny, real* x,      \
                                 real& f, real* fgrad, real** assist_buffer, \
                                 int task, real const& lambda);

INST_HELPER(float);
INST_HELPER(double);
