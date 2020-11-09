// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#ifndef MATRIXMULT_H
#define MATRIXMULT_H

// Block size. Affects the kernel, so if this value changes, the kernel
// needs to be recompiled.
#define BLOCK_SIZE (64ul) // default value
#define SEGMENT (1024ul)

#define BATCH (1000)

// SCALE and N must have the following relationship: N = 2 ^ SCALE
#define SCALE (15)
// #define N ((unsigned long)pow(2, SCALE))
#define N (32768ul)
#define DEGREE (3)
#define NNZ (DEGREE * N)
#define ITER (10)

typedef unsigned long spMV_data;

#define ARRAY(ptr,i1,i0,d1,d0) (*((ptr)+(i1)*(d0)+(i0)))

#define RANGE (10)

#define EPSILON (1e-4)  // do not change this value
#endif

