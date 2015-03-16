#ifndef CUTIL_INLINE_H
#define CUTIL_INLINE_H

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/glut.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#include <cuda_gl_interop.h>

#ifndef __max
#define __max(a, b) {a < b ? b : a;}
#endif

#define cutilSafeCall(x) {if(!cudaSuccess == x) exit(0);}

#endif