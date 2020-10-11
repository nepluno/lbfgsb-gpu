/********************************************************************
AP Library version 1.2
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

#include "culbfgsb/ap.h"

namespace ap {
const double machineepsilon = 5E-16;
const double maxrealnumber = 1E300;
const double minrealnumber = 1E-300;

/********************************************************************
standard functions
********************************************************************/
int randominteger(int maxv) { return rand() % maxv; }

int maxint(int m1, int m2) { return m1 > m2 ? m1 : m2; }

int minint(int m1, int m2) { return m1 > m2 ? m2 : m1; }

/********************************************************************
Service routines:
********************************************************************/
void *amalloc(size_t size, size_t alignment) {
  if (alignment <= 1) {
    //
    // no alignment, just call malloc
    //
    void *block = malloc(sizeof(void *) + size);
    void **p = (void **)block;
    *p = block;
    return (void *)((char *)block + sizeof(void *));
  } else {
    //
    // align.
    //
    void *block = malloc(alignment - 1 + sizeof(void *) + size);
    char *result = (char *)block + sizeof(void *);
    // if( ((unsigned int)(result))%alignment!=0 )
    //    result += alignment - ((unsigned int)(result))%alignment;
    if ((result - (char *)0) % alignment != 0)
      result += alignment - (result - (char *)0) % alignment;
    *((void **)(result - sizeof(void *))) = block;
    return result;
  }
}

void afree(void *block) {
  void *p = *((void **)((char *)block - sizeof(void *)));
  free(p);
}

int vlen(int n1, int n2) { return n2 - n1 + 1; }
}  // namespace ap
