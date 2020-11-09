#include "../host/inc/spMV.h"

#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

// #pragma OPENCL EXTENSION cl_intel_channels : enable
// channel int c0, c1;

// __kernel void load(__global float* restrict V, __global spMV_data* restrict col, __global spMV_data* restrict row, 
//                      __global float* restrict x£¬ __global spMV_data real_NNZ) {
//     bool success = false;

//     for (int m = 0; m < BATCH; m++) {
//         for (int i = 0; i < real_NNZ; i++) {
//             success = write_channel_intel(c0, row[i]);
//             if (!success) {
//                 printf("No success in transfering on row[%lu]\n", i);
//             }

//             success = write_channel_intel(c0, col[i]);
//             if (!success) {
//                 printf("No success in transfering on col[%lu]\n", i);
//             }

//             success = write_channel_intel(c0, V[i]);
//             if (!success) {
//                 printf("No success in transfering on V[%lu]\n", i);
//             }

//             success = write_channel_intel(c0, x[col[i]]);
//             if (!success) {
//                 printf("No success in transfering on x[%lu]\n", col[i]);
//             }
//         }
//     }
// }

// __kernel void execute() {
//     for (int m = 0; m < BATCH; m++) {
//         for (int i = 0; i < real_NNZ; i++) {

//             success = write_channel_intel(c0, row[i]);
//             if (!success) {
//                 printf("No success in transfering on row[%lu]\n", i);
//             }

//             success = write_channel_intel(c0, col[i]);
//             if (!success) {
//                 printf("No success in transfering on col[%lu]\n", i);
//             }

//             success = write_channel_intel(c0, V[i]);
//             if (!success) {
//                 printf("No success in transfering on V[%lu]\n", i);
//             }

//             success = write_channel_intel(c0, x[col[i]]);
//             if (!success) {
//                 printf("No success in transfering on x[%lu]\n", col[i]);
//             }
//         }
//     }
// }

// __kernel void store(__global float* restrict y, __global spMV_data* restrict y_idx) {

// }

__kernel void spMV( __global float* restrict V, __global spMV_data* restrict col, __global spMV_data* restrict row, 
                     __global float* restrict x, __global float* restrict y, __global spMV_data* restrict y_idx, 
                     const spMV_data real_NNZ)
{
    __local float x_seg[SEGMENT];
    __local spMV_data y_local_idx[N];
    __local float y_local[N];
    unsigned long i, j;
    spMV_data k;
    spMV_data row_start, row_end;

    for (int m = 0; m < BATCH; m++){
        k = 0;

        for (i = 0; i < N; i++) {
            y[i] = 0;
            y_idx[i] = 0;
        }

        // Stream the first segment of x from DRAM to fast memory
        for (i = 0; i < SEGMENT; i++) {
            x_seg[i] = x[i];
        }

        // // Initialize y_local
        // for (i = 0; i < N; i++) {
        //     y_local[i] = 0;
        //     y_local_idx[i] = 0;
        // }

        // spMV
        for (i = 0; i < real_NNZ; i++) {
            y[k] += V[i] * x_seg[col[i]];
            if (i == real_NNZ - 1 || row[i+1] != row[i]){
                y_idx[k] = row[i];
                k++;
                if (m == 0 && row[i] == 27){
                    printf("kernel: y[%lu] = %f\n", k, y[k]);
                }
            }
        }

        // printf("Kernel k = %lu\n", k);

        // Stream y_local back to main memory
        // for (i = 0; i < k; i++) {
        //     y[i] = y_local[i];
        //     y_idx[i] = y_local_idx[i];
        // }
    }
}
