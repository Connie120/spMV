#include "../host/inc/spMV.h"

#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel float c0, c1;

__kernel void load(__global float* restrict V, __global spMV_data* restrict col, __global spMV_data* restrict row, 
                     __global float* restrict x, const spMV_data real_NNZ) {
    // printf("kernel load\n");
    __private float x_seg[SEGMENT];

    // Stream the first segment of x from DRAM to fast memory
    for (int i = 0; i < SEGMENT; i++) {
        x_seg[i] = x[i];
    }

    for (int m = 0; m < BATCH; m++) {
        for (int i = 0; i < real_NNZ; i++) {
            write_channel_intel(c0, row[i]);
            // printf("kernel write channel c0 row[%d]: %lu\n", i, row[i]);

            write_channel_intel(c0, col[i]);
            // printf("kernel write channel c0 col[%d]: %lu\n", i, col[i]);

            write_channel_intel(c0, V[i]);
            // printf("kernel write channel c0 V[%d]: %f\n", i, V[i]);

            write_channel_intel(c0, x_seg[col[i]]);
            // printf("kernel write channel c0 x_seg[%lu]: %f\n", col[i], x_seg[col[i]]);
        }
    }
    // printf("kernel load done.\n");
}

__kernel void execute(const spMV_data real_NNZ) {
    // printf("kernel execute\n");
    spMV_data row_prev, row_curr, col_curr;
    float x_curr, V_curr;

    float y_temp = 0.0f;

    spMV_data k = 0;

    row_prev = 0;

    for (int m = 0; m < BATCH; m++) {
        for (int i = 0; i < real_NNZ; i++) {
            row_curr = read_channel_intel(c0);
            // printf("kernel read channel c0 row: %lu\n", row_curr);

            col_curr = read_channel_intel(c0);
            // printf("kernel read channel c0 col: %lu\n", col_curr);

            V_curr = read_channel_intel(c0);
            // printf("kernel read channel c0 V: %f\n", V_curr);

            x_curr = read_channel_intel(c0);
            // printf("kernel read channel c0 x: %f\n", x_curr);

            // spMV
            if (row_prev != row_curr){
                write_channel_intel(c1, y_temp);
                // printf("kernel write channel c1 y: %f\n", y_temp);

                write_channel_intel(c1, row_prev);
                // printf("kernel write channel c1 row: %lu\n", row_prev);

                k++;

                y_temp = 0.0f;
            }
            y_temp += V_curr * x_curr;

            row_prev = row_curr;
        }
        write_channel_intel(c1, y_temp);
        // printf("kernel write channel c1 y: %f\n", y_temp);

        write_channel_intel(c1, row_prev);
        // printf("kernel write channel c1 row: %lu\n", row_prev);

        write_channel_intel(c1, -1);
        // printf("kernel write channel c1 terminate\n");
    }
    // printf("kernel execute done.\n");
}

__kernel void store(__global float* restrict y, __global spMV_data* restrict y_idx) {
    // printf("kernel store\n");
    for (int m = 0; m < BATCH; m++) {
        // Initialize the output
        for (int i = 0; i < N; i++) {
            if (y[i] != 0) {
                y[i] = 0;
                y_idx[i] = 0;
            }
            else {
                break;
            }
        }

        for (int i = 0; i < N; i++) {
            y[i] = read_channel_intel(c1);
            // printf("kernel read channel c1 y[%d]: %f\n", i, y[i]);

            if (y[i] < 0) {
                // printf("kernel store terminate\n");
                break;
            }

            y_idx[i] = read_channel_intel(c1);
            // printf("kernel read channel c1 y_idx[%d]: %lu\n", i, y_idx[i]);

            if (y_idx[i] == -1) {
                printf("Error!!!\n");
                break;
            }
        }
    }
    // printf("kernel store done.\n");
}

// __kernel void spMV( __global float* restrict V, __global spMV_data* restrict col, __global spMV_data* restrict row, 
//                      __global float* restrict x, __global float* restrict y, __global spMV_data* restrict y_idx, 
//                      const spMV_data real_NNZ)
// {
//     __private float x_seg[SEGMENT];

//     unsigned long i, j;
//     spMV_data k;
//     spMV_data row_start, row_end;

//     for (int m = 0; m < BATCH; m++){
//         k = 0;

//         for (i = 0; i < N; i++) {
//             y[i] = 0;
//             y_idx[i] = 0;
//         }

//         // Stream the first segment of x from DRAM to fast memory
//         for (i = 0; i < SEGMENT; i++) {
//             x_seg[i] = x[i];
//         }

//         // spMV
//         for (i = 0; i < real_NNZ; i++) {
//             y[k] += V[i] * x_seg[col[i]];
//             if (i == real_NNZ - 1 || row[i+1] != row[i]){
//                 y_idx[k] = row[i];
//                 k++;
//             }
//         }

//         // printf("Kernel k = %lu\n", k);
//     }
// }
