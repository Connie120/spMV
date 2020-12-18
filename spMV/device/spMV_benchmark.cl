// #include "../host/inc/spMV_kernel.h"
#include "../host/inc/spMV.h"

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
} pack_out_kernel;

#pragma OPENCL EXTENSION cl_intel_channels : enable
channel struct pack_in c0;
channel struct pack_out c1;

__kernel void load(__global struct pack_in* restrict A, const spMV_data real_NNZ) {
    // printf("kernel load\n");

    for (int m = 0; m < BATCH; m++) {
        for (int i = 0; i < real_NNZ; i++) {
            pack_in A_temp = A[i];
            // write_channel_intel(c0, A[i]);
        }
    }
    // printf("kernel load done.\n");
}

// __kernel void execute(__global spMV_float* restrict x, const spMV_data real_NNZ) {
//     // printf("kernel execute\n");
//     spMV_data row_prev, row_curr, col_curr;
//     spMV_float V_curr;
//     struct pack_in A;
//     struct pack_out y;

//     __private spMV_float x_seg[SEGMENT];

//     // Stream the first segment of x from DRAM to fast memory
//     for (int i = 0; i < SEGMENT; i++) {
//         x_seg[i] = x[i];
//     }

//     y.value = 0.0f;
//     row_prev = 0;

//     for (int m = 0; m < BATCH; m++) {
//         row_prev = 0;
//         y.value = 0.0f;
//         #pragma unroll
//         for (int i = 0; i < real_NNZ; i++) {
//             A = read_channel_intel(c0);

//             y.idx = A.row;
//             y.value = A.V * x_seg[A.col];
//             write_channel_intel(c1, y);
//         }
//     }
//     // printf("kernel execute done.\n");
// }

// __kernel void store(__global struct pack_out* restrict y, const spMV_data real_NNZ) {
//     // printf("kernel store\n");
//     for (int m = 0; m < BATCH; m++) {
//         spMV_data k = 0;

//         for (int i = 0; i < real_NNZ; i++) {
//             pack_out y_temp = read_channel_intel(c1);
//             if (y_temp.idx == y[k].idx) {
//                 y[k].value += y_temp.value;
//             }
//             else {
//                 k++;
//                 y[k].value = y_temp.value;
//                 y[k].idx = y_temp.idx;
//             }
//         }
//     }
//     // printf("kernel store done.\n");
// }
