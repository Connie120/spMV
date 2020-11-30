#include "../host/inc/spMV_kernel.h"

// #ifndef SIMD_WORK_ITEMS
// #define SIMD_WORK_ITEMS 4 // default value
// #endif

typedef ulong spMV_data_kernel;
typedef float spMV_float_kernel;

typedef struct pack_in_kernel {
    spMV_data_kernel row;
    spMV_data_kernel col;
    spMV_float_kernel V;
} pack_in_kernel;

typedef struct pack_out_kernel {
    spMV_float_kernel value;
    spMV_data_kernel idx;
} pack_out;

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel struct pack_in_kernel c0;
channel spMV_float_kernel c0_x;
channel struct pack_out_kernel c1;

__kernel void load(__global struct pack_in_kernel* restrict A, __global spMV_float_kernel* restrict x, const spMV_data_kernel real_NNZ) {
    printf("kernel load\n");
    __private spMV_float_kernel x_seg[SEGMENT];

    // Stream the first segment of x from DRAM to fast memory
    for (int i = 0; i < SEGMENT; i++) {
        x_seg[i] = x[i];
    }

    for (int m = 0; m < BATCH; m++) {
        for (int i = 0; i < real_NNZ; i++) {
            write_channel_intel(c0, A[i]);
            printf("kernel write channel c0 row[%d]: %lu\n", i, A[i].row);
            printf("kernel write channel c0 col[%d]: %lu\n", i, A[i].col);
            printf("kernel write channel c0 V[%d]: %f\n", i, A[i].V);

            write_channel_intel(c0_x, x_seg[A[i].col]);
            printf("kernel write channel c0 x_seg[%lu]: %f\n", A[i].col, x_seg[A[i].col]);
        }
    }
    printf("kernel load done.\n");
}

__kernel void execute(const spMV_data_kernel real_NNZ) {
    printf("kernel execute\n");
    spMV_data_kernel row_prev, row_curr, col_curr;
    spMV_float_kernel x_curr, V_curr;
    struct pack_in_kernel A;
    struct pack_out_kernel y;

    A.row = 0;
    A.col = 0;
    A.V = 0.0f;
    printf("sizeof A: %lu\n", sizeof(pack_in_kernel));
    y.value = 0.0f;

    row_prev = 0;

    for (int m = 0; m < BATCH; m++) {
        row_prev = 0;
        y.value = 0.0f;
        for (int i = 0; i < real_NNZ; i++) {
            A = read_channel_intel(c0);
            printf("kernel read channel c0 row: %lu\n", A.row);
            printf("kernel read channel c0 col: %lu\n", A.col);
            printf("kernel read channel c0 V: %f\n", A.V);

            x_curr = read_channel_intel(c0_x);
            printf("kernel read channel c0 x: %f\n", x_curr);

            // spMV
            if (row_prev != A.row){
                y.idx = row_prev;
                write_channel_intel(c1, y);
                printf("kernel write channel c1 y: %f\n", y.value);
                printf("kernel write channel c1 y_idx: %lu\n", y.idx);

                y.value = 0.0f;
            }
            y.value += A.V * x_curr;

            row_prev = A.row;
        }
        y.idx = A.row;
        write_channel_intel(c1, y);
        printf("kernel write channel c1 y: %f\n", y.value);
        printf("kernel write channel c1 y_idx: %lu\n", y.idx);

        // printf("Number of channel write in execute: %lu\n", k);
    }
    printf("kernel execute done.\n");
}

__kernel void store(__global struct pack_out_kernel* restrict y, const spMV_data_kernel NZR) {
    printf("kernel store\n");
    for (int m = 0; m < BATCH; m++) {
        for (int i = 0; i < NZR; i++) {
            y[i] = read_channel_intel(c1);
            printf("kernel read channel c1 y[%lu]: %f\n", y[i].idx, y[i].value);

            // if (y[i] < 0) {
            //     // printf("kernel store terminate\n");
            //     break;
            // }
            // printf("kernel read channel c1 y_idx[%d]: %lu\n", i, y_idx[i]);

            // if (y_idx[i] == -1) {
            //     printf("Error!!!\n");
            //     break;
            // }
        }
    }
    printf("kernel store done.\n");
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
