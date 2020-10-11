/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef CULBFGSB_CUTIL_INLINE_H_
#define CULBFGSB_CUTIL_INLINE_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define cutilSafeCall(x)                                                   \
  {                                                                        \
    if (cudaSuccess != x) {                                                \
      printf("lbfgsb failure: %d, %s, %d\n", cudaGetLastError(), __FILE__, \
             __LINE__);                                                    \
      exit(0);                                                             \
    }                                                                      \
  }

#endif  // CULBFGSB_CUTIL_INLINE_H_