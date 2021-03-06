#include "../host/inc/spMV.h"

#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

__kernel void spMV( __global float* restrict V, __global spMV_data* restrict col, __global spMV_data* restrict row, 
                     __global float* restrict x, __global float* restrict y, __global spMV_data* restrict y_idx)
{
    __local float x_seg[SEGMENT];
    __local spMV_data y_local_idx[N];
    __local float y_local[N];
    unsigned long start_block = get_global_id(0);
    unsigned long i, j;
    spMV_data k = start_block * BLOCK_SIZE;
    spMV_data row_start, row_end;

    for (int m = 0; m < BATCH; m++){
        // Stream the first segment of x from DRAM to fast memory
        if (start_block == 0) {
            for (i = 0; i < SEGMENT; i++) {
                x_seg[i] = x[i];
            }

            // Initialize y_local
            for (i = 0; i < N; i++) {
                y_local[i] = 0;
                y_local_idx[i] = 0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // spMV
        for (i = start_block * BLOCK_SIZE; i < (start_block + 1) * BLOCK_SIZE; i++) {
            row_start = row[i];
            row_end = row[i+1];

            //If there's a nonzero element in the block of A
            if (row_end > row_start) {
                for (j = row_start; j < row_end; j++) {
                    y_local[k] += V[j] * x_seg[col[j]];
                }
                if (y_local[k] != 0) {
                    y_local_idx[k] = i;
                    
                    k++;
                }
            }
        }

        // printf("Kernel k = %lu at block %lu\n", k, start_block);

        // Stream y_local back to main memory
        for (i = start_block * BLOCK_SIZE; i < k; i++) {
            y[i] = y_local[i];
            y_idx[i] = y_local_idx[i];
        //     if (start_block == 16) {
        //         printf("Kernel block 16 y_local[%lu]: %f at %lu\n", i, y[i], y_idx[i]);
        //     }
        }
    }
}
