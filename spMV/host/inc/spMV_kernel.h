#ifndef SPMV_KERNEL_H
#define SPMV_KERNEL_H

// Block size. Affects the kernel, so if this value changes, the kernel
// needs to be recompiled.
#define BLOCK_SIZE (64ul) // default value
#define SEGMENT (4ul)

#define BATCH (1)

// SCALE and N must have the following relationship: N = 2 ^ SCALE
#define SCALE (4)
// #define N ((unsigned long)pow(2, SCALE))
#define N (16ul)
#define DEGREE (3)
#define NNZ (DEGREE * N)
// #define ITER (10)

#define ARRAY(ptr,i1,i0,d1,d0) (*((ptr)+(i1)*(d0)+(i0)))

#define RANGE (10)

#define EPSILON (1e-4)  // do not change this value

#endif

