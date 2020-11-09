#include "../host/inc/spMV.h"

#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

// #pragma OPENCL EXTENSION cl_intel_channels : enable
// channel int c0, c1;

__kernel void load(__global float* restrict V, __global spMV_data* restrict col, __global spMV_data* restrict row, 
                     __global float* restrict x£¬ __global spMV_data real_NNZ) {
    bool success = false;

    __private float x_seg[SEGMENT];

    // Stream the first segment of x from DRAM to fast memory
    for (i = 0; i < SEGMENT; i++) {
        x_seg[i] = x[i];
    }

    for (int m = 0; m < BATCH; m++) {
        for (int i = 0; i < real_NNZ; i++) {
            success = write_channel_intel(c0, row[i]);
            if (!success) {
                printf("No success in transfering on row[%lu]\n", i);
            }

            success = write_channel_intel(c0, col[i]);
            if (!success) {
                printf("No success in transfering on col[%lu]\n", i);
            }

            success = write_channel_intel(c0, V[i]);
            if (!success) {
                printf("No success in transfering on V[%lu]\n", i);
            }

            success = write_channel_intel(c0, x_seg[col[i]]);
            if (!success) {
                printf("No success in transfering on x[%lu]\n", col[i]);
            }
        }
    }
}

__kernel void execute(const spMV_data real_NNZ) {
    spMV_data row_prev, row_curr, col_curr;
    float x_curr, V_curr;

    float y_temp = 0.0f;

    bool success = false;

    spMV_data k = 0;

    row_prev = 0;

    for (int m = 0; m < BATCH; m++) {
        for (int i = 0; i < real_NNZ; i++) {
            row_curr = read_channel_intel(c0);

            col_curr = read_channel_intel(c0);

            V_curr = read_channel_intel(c0);

            x_curr = read_channel_intel(c0);

            // spMV
            if (row_prev != row_curr){
                success = write_channel_intel(c1, row_prev);
                if (!success) {
                    printf("No success in transfering on row %lu\n", row_prev);
                }

                success = write_channel_intel(c1, y_temp);
                if (!success) {
                    printf("No success in transfering on y_temp: %f\n", y_temp);
                }

                k++;

                y_temp = 0.0f;
            }
            y_temp += V_curr * x_curr;

            row_prev = row_curr;
        }
        success = write_channel_intel(c1, -1);
        if (!success) {
            printf("No success in transfering on terminating signal\n");
        }
    }
}

__kernel void store(__global float* restrict y, __global spMV_data* restrict y_idx) {
    for (int m = 0; m < BATCH; m++) {
        // Initialize the output
        for (i = 0; i < N; i++) {
            y[i] = 0;
            y_idx[i] = 0;
        }

        for (int i = 0; i < N; i++) {
            y_idx[i] = read_channel_intel(c1);
            y[i] = read_channel_intel(c1);
            if (y_idx[i] == -1) {
                break;
            }
            if (y[i] == -1) {
                printf("Error!!!\n");
                break;
            }
        }
    }
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
