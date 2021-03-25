#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "spMV.h"
#include <assert.h>
#include "float.h"
#include <algorithm>
#include <typeinfo>
#include <iostream>

using namespace aocl_utils;
using namespace std;

#define ACL_ALIGNMENT 64

void* acl_aligned_malloc (size_t size) {
    void *result = NULL;
    if (posix_memalign(&result, ACL_ALIGNMENT, size) != 0)
        printf("acl_aligned_malloc() failed.\n");
    return result;
}
void acl_aligned_free (void *ptr) {
    free (ptr);
}

// extern CL_API_ENTRY cl_int CL_API_CALL;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
cl_command_queue load_queue;
cl_command_queue exec_queue;
cl_command_queue store_queue;
cl_program program = NULL;
cl_kernel load_kernel;
cl_kernel exec_kernel;
cl_kernel store_kernel;

cl_mem A_buf;
cl_mem x_buf;
cl_mem y_buf;
cl_mem y_idx_buf;

spMV_float* dt_x = (spMV_float*)acl_aligned_malloc(N * sizeof(spMV_float));
pack_out* dt_y;
pack_in* dt_A = (pack_in*)acl_aligned_malloc(NNZ * sizeof(pack_in));

spMV_float* ref_output;

spMV_data real_NNZ = 0;
spMV_data NZR = 0; // Number of non-zero rows

// Function prototypes
void calc_result(pack_in *A, spMV_float *x, spMV_float *y);
int nearlyEqual(float a, float b);
bool init_opencl();
void init_problem();
void run();
void verify();
void verifyZeros();
void cleanup();

bool mySort(spMV_data a, spMV_data b) {
    return(a < b);
}

// Entry point.
int main(int argc, char **argv) {
    printf("N: %lu\n", N);
    printf("BLOCK_SIZE: %lu\n", BLOCK_SIZE);
    printf("Segment size: %lu\n", SEGMENT);
    printf("Batch size: %d\n", BATCH);

    // Initialize OpenCL.
    if(!init_opencl()) {
        return -1;
    }
	  printf("init_opencl done.\n");

    // Initialize the problem data.
    // Requires the number of devices to be known.
    init_problem();
	printf("init_problem done.\n");

    // Run the kernel.
    run();

    // Free the resources allocated
    cleanup();

    return 0;
}

// Initializes the OpenCL objects.
bool init_opencl() {
    cl_int status;

    printf("Initializing OpenCL\n");

    // assert(Tm <= max_Tm && Tr <= max_Tr && Tc <= max_Tc && Tn <= max_Tn);

    if(!setCwdToExeDir()) {
        return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    if(platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device.
    device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    printf("Platform: %s\n", getPlatformName(platform).c_str());
    printf("Using %d device(s)\n", num_devices);

	//To test number of device is equal to 1
	  num_devices = 1;
    printf("  %s\n", getDeviceName(device[0]).c_str());

    // Create the context.
    context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the program for all device. Use the first device as the
    // representative device (assuming all device are of the same type).
    std::string binary_file = getBoardBinaryFile("spMV_pipeline", device[0]);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Command queue.
    load_queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create load command queue");

    exec_queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create execute command queue");

    store_queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create store command queue");

    // Kernel.
    const char *load_kernel_name = "load";
    load_kernel = clCreateKernel(program, load_kernel_name, &status);
    checkError(status, "Failed to create load kernel");

    const char *exec_kernel_name = "execute";
    exec_kernel = clCreateKernel(program, exec_kernel_name, &status);
    checkError(status, "Failed to create execute kernel");

    const char *store_kernel_name = "store";
    store_kernel = clCreateKernel(program, store_kernel_name, &status);
    checkError(status, "Failed to create store kernel");

    // Input buffer.
    x_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    N * sizeof(spMV_float), NULL, &status);
    checkError(status, "Failed to create buffer for input vector");
	  printf("x buffer done\n");

    // Matrix buffers.
    A_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, NNZ * sizeof(pack_in), NULL, &status);
    checkError(status, "Failed to create buffer for A");
	printf("A buffer done\n");

    // Output buffer.
    y_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    NNZ * sizeof(pack_out), NULL, &status);
    checkError(status, "Failed to create buffer for output");
	printf("y buffer done\n");

    return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
    if(num_devices == 0) {
        checkError(-1, "No devices");
    }

    // Generate input and weight matrices.
    printf("Generating inputs\n");

    spMV_data i,j;

    // Generate the input matrix
    printf("The number of expected non-zero elements in the whole matrix is: %lu\n", NNZ);
    // for (i = 0; i < N; i++) {
	// 	dt_A[i] = 0;
    // }

    // for (i = 0; i < NNZ;) {
    //     spMV_data index = (spMV_data) (N * N * ((double) rand() / (RAND_MAX + 1.0)));
    //     if (dt_A[index] != 0) {
    //         continue;
    //     }
    //     dt_A[index] = ((spMV_float)(rand()%RANGE))/RANGE;
    //     if (dt_A[index] == 0) {
    //         continue;
    //     }
    //     i++;
    // }

	// // Changed the input matrix into the CSR form (first block)
	// spMV_data k = 0;
    // spMV_data temp = 0;
	// dt_ROW[0] = 0;
	// for (i = 0; i < N; i++) {
	// 	for (j = 0; j < SEGMENT; j++) {
	// 		if (ARRAY(dt_A, i, j, N, N) != 0) {
	// 			dt_V[k] = ARRAY(dt_A, i, j, N, N);
	// 			dt_COL[k] = j;
	// 			k++;
	// 		}
	// 	}
	// 	dt_ROW[i+1] = k;
	// }

    float a = 0.57;
    float b = 0.19;
    float c = 0.19;

    float ab = a + b;
    float c_norm = c / (1 - ab);
    float a_norm = a / ab;

    spMV_data ii_bit = 0;
    spMV_data jj_bit = 0;
    spMV_data start_node = 0;
    spMV_data end_node = 0;

    std::map<spMV_data, std::vector<spMV_data> > nodes;

    for (int i = 0; i < NNZ; i++) {
        start_node = 0;
        end_node = 0;
        for (int ib = 0; ib < SCALE; ib++) {
            ii_bit = ((double) rand() / (RAND_MAX + 1.0)) > ab;
            jj_bit = ((double) rand() / (RAND_MAX + 1.0)) > (c_norm * ii_bit + a_norm * !ii_bit);
            start_node += pow(2, ib) * ii_bit;
            end_node += pow(2, ib) * jj_bit;
        }
        
        if (end_node < SEGMENT) {
            if (nodes.find(start_node) == nodes.end()) {
                nodes.insert(make_pair(start_node, std::vector<spMV_data>()));
            }
            if (find(nodes[start_node].begin(), nodes[start_node].end(), end_node) == nodes[start_node].end()) {
                nodes[start_node].push_back(end_node);
            }
        }
    }

    spMV_data col_iter = 0;
    // spMV_data row_iter = 1;
    // spMV_data prev_row = 0;
    // dt_ROW[0] = 0;
    for (auto& node : nodes) {
        sort(node.second.begin(), node.second.end(), mySort);
        // if (node.first != prev_row) {
        //     while (row_iter <= node.first) {
        //         dt_ROW[row_iter] = dt_ROW[row_iter-1];
        //         row_iter++;
        //     }
        // }
        // dt_ROW[row_iter] = dt_ROW[row_iter-1] + node.second.size();
        // printf("dt_ROW[%lu]: %lu\n", row_iter, dt_ROW[row_iter]);
        // row_iter++;
        // prev_row = node.first + 1;
        for (int i = 0; i < node.second.size(); i++) {
            dt_A[col_iter].col = node.second[i];
            dt_A[col_iter].row = node.first;
            // if (node.first == 0) {
            //     printf("dt_COL[%lu]: %lu\n", col_iter, dt_COL[col_iter]);
            // }
            col_iter++;
        }
    }
    // while (row_iter <= N) {
    //     dt_ROW[row_iter] = dt_ROW[row_iter-1];
    //     row_iter++;
    // }

    for (i = 0; i < col_iter; i++) {
        dt_A[i].V = (spMV_float) rand() / (RAND_MAX + 1.0);
    }

	printf("The number of actual non-zero elements in the block is: %u\n", col_iter);

    // for (i = 0; i < N * N; i++) {
    //     if (dt_A[i] != 0) {
    //         temp++;
    //     }
    // }
    // printf("The number of actual non-zero elements in the matrix is: %lu\n", temp);

    // Generate the input vector
    for (j = 0; j < N; j++) {
        dt_x[j] = (((spMV_float)(rand()%RANGE))/RANGE);
    }
      
	printf("generating all inputs done\n");
    real_NNZ = col_iter;

    NZR++;
    for (i = 1; i < real_NNZ; i++) {
        if (dt_A[i].row != dt_A[i-1].row) {
            NZR++;
        }
    }

    printf("NZR: %u\n", NZR);

    dt_y = (pack_out*)acl_aligned_malloc(NZR * sizeof(pack_out));
    ref_output = (spMV_float*)acl_aligned_malloc(NZR * sizeof(spMV_float));

    // Set the actual output and reference output matrices to 0.
    for(i = 0; i < NZR; i++) {
        dt_y[i].value = 0;
        dt_y[i].idx = 0;
        ref_output[i] = 0;
    }
}

void run() {
    cl_int status;

    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    status = clEnqueueWriteBuffer(load_queue, A_buf, CL_TRUE,
                                    0, NNZ * sizeof(pack_in), dt_A, 0, NULL, NULL);
    checkError(status, "Failed to transfer A");
    printf("sizeof packin: %lu\n", sizeof(pack_in));

    status = clEnqueueWriteBuffer(exec_queue, x_buf, CL_TRUE,
                                    0, N * sizeof(spMV_float), dt_x, 0, NULL, NULL);
    checkError(status, "Failed to transfer x");

    status = clEnqueueWriteBuffer(store_queue, y_buf, CL_TRUE,
                                   0, NZR * sizeof(pack_out), dt_y, 0, NULL, NULL);
    checkError(status, "Failed to transfer y");

    // Wait for all queues to finish.
    clFinish(load_queue);
    clFinish(exec_queue);
    clFinish(store_queue);

    // Launch kernels.
    // This is the portion of time that we'll be measuring for throughput
    // benchmarking.
    cl_event kernel_event_load;
    cl_event kernel_event_exec;
    cl_event kernel_event_store;

    const double start_time = getCurrentTimestamp();
    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(load_kernel, argi++, sizeof(cl_mem), &A_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(load_kernel, argi++, sizeof(spMV_data), &real_NNZ);
    checkError(status, "Failed to set argument %d", argi - 1);

    argi = 0;

    status = clSetKernelArg(exec_kernel, argi++, sizeof(cl_mem), &x_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(exec_kernel, argi++, sizeof(spMV_data), &real_NNZ);
    checkError(status, "Failed to set argument %d", argi - 1);

    argi = 0;

    status = clSetKernelArg(store_kernel, argi++, sizeof(cl_mem), &y_buf);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(store_kernel, argi++, sizeof(spMV_data), &NZR);
    checkError(status, "Failed to set argument %d", argi - 1);

    // Enqueue kernel.
	const size_t global_work_size[1] = { 1 };
	const size_t local_work_size[1] = { 1 };

    printf("Enqueue Kernel\n");

    status = clEnqueueNDRangeKernel(load_queue, load_kernel, 1, NULL,
                                    global_work_size, NULL, 0, NULL, &kernel_event_load);
    checkError(status, "Failed to launch load kernel");

    status = clEnqueueNDRangeKernel(exec_queue, exec_kernel, 1, NULL,
                                    global_work_size, NULL, 0, NULL, &kernel_event_exec);
    checkError(status, "Failed to launch load kernel");

    status = clEnqueueNDRangeKernel(store_queue, store_kernel, 1, NULL,
                                    global_work_size, NULL, 0, NULL, &kernel_event_store);
    checkError(status, "Failed to launch load kernel");

    // clGetProfileInfoIntelFPGA(kernel_event);

    // Wait for all kernels to finish.
    clWaitForEvents(num_devices, &kernel_event_load);
    clWaitForEvents(num_devices, &kernel_event_exec);
    clWaitForEvents(num_devices, &kernel_event_store);

    const double end_time = getCurrentTimestamp();
    const double total_time = end_time - start_time;

    // Wall-clock time taken.
    printf("\nTime: %0.3f ms\n", total_time * 1e3);

    // Get kernel times using the OpenCL event profiling API.
    cl_ulong load_time_ns = getStartEndTime(kernel_event_load);
    printf("Kernel time (device %d): %0.3f ms\n", 1, double(load_time_ns) * 1e-6);

    cl_ulong exec_time_ns = getStartEndTime(kernel_event_exec);
    printf("Kernel time (device %d): %0.3f ms\n", 1, double(exec_time_ns) * 1e-6);

    cl_ulong store_time_ns = getStartEndTime(kernel_event_store);
    printf("Kernel time (device %d): %0.3f ms\n", 1, double(store_time_ns) * 1e-6);

    // TODO: Compute the throughput (GFLOPS).
    double max_time_ns = double(load_time_ns) > double(exec_time_ns) ? load_time_ns : exec_time_ns;
    max_time_ns = max_time_ns > double(store_time_ns) ? max_time_ns : store_time_ns;
    const float flops = (float)(2.0f * BATCH * real_NNZ / (max_time_ns / 1e9));
    printf("\nThroughput: %0.2f GFLOPS\n\n", flops * 1e-9);

    // Release kernel events.
    clReleaseEvent(kernel_event_load);

    clReleaseEvent(kernel_event_exec);

    clReleaseEvent(kernel_event_store);

    // Read the result.
    status = clEnqueueReadBuffer(store_queue, y_buf, CL_TRUE,
                                    0, NZR * sizeof(pack_out), dt_y, 0, NULL, NULL);
    checkError(status, "Failed to read output vector");

    // for (spMV_data i = 0; i < N; i++) {
    //     if (i == 0 || dt_y_idx[i] != 0) {
    //         printf("dt_y_idx[%lu]: %lu\n", i, dt_y_idx[i]);
    //     }   
    // }

    // printf("First verification\n");
    // Verify results.
    calc_result(dt_A, dt_x, ref_output);
    verify();
}

void calc_result(pack_in *A, spMV_float *x, spMV_float *y) {
    printf("Computing reference output\n");
    spMV_data i,j,k;
    spMV_data row_start, row_end;

	// spMV_float* temp_y = (spMV_float*)malloc(N * sizeof(spMV_float));

    // for (i = 0; i < N; i++) {
	// 	for (j = 0; j < SEGMENT; j++) {
	// 		if (ARRAY(A, i, j, N, N) != 0) {
	// 			y[i] += ARRAY(A, i, j, N, N) * x[j];
	// 		}
	// 	}
	// }

    k = 0;
    y[0] += A[0].V * x[A[0].col];
    for (i = 1; i < real_NNZ; i++) {
        if (A[i].row != A[i-1].row) {
            k++;
        }
        y[k] += A[i].V * x[A[i].col];
        // printf("y[%lu] = %f\n", dt_ROW[i], y[dt_ROW[i]]);
    }

    // k = 0;
    // for (i = 0; i < N; i++) {
    //     y[i] = temp
    // }

    // free(temp_y);
}

void verify() {
    printf("Verifying\n");

    spMV_data i;

    // spMV_float y[N];
    // for (i = 0; i < N; i++) {
    //     y[i] = 0;
    //     // printf("dt_y_idx[%lu]: %lu\n", i, dt_y_idx[i]);
    //     // if (i <= 4) {
    //     //     printf("dt_y_idx[%lu] = %lu\n", i, dt_y_idx[i]);
    //     //     printf("dt_y[%lu] = %f\n", i, dt_y[i]);
    //     // }
    // }
    // for (i = 0; i < N; i++) {
    //     if (dt_y[i] != -1) {
    //         y[dt_y_idx[i]] = dt_y[i];
    //         // printf("dt_y_idx[%lu] = %lu\n", i, dt_y_idx[i]);
    //         // printf("dt_y[%lu] = %f\n", i, dt_y[i]);
    //     }
    //     else {
    //         break;
    //     }
    // }

	for(i = 0; i < NZR ; i++) {
		if (!nearlyEqual((float)dt_y[i].value, (float)ref_output[i])) {
            printf("y[%u]: %f, ref_output[%u]: %f\n", i, dt_y[i].value, i, ref_output[i]);
		}
		assert(nearlyEqual((float)dt_y[i].value, (float)ref_output[i]));
	}

    printf("Results correct.\n\n");
}

int nearlyEqual(float a, float b) {
    float absA = fabs(a);
    float absB = fabs(b);
    float diff = fabs(a - b);

    if (a == b) { // shortcut, handles infinities
        return 1;
    } else if (a == 0 || b == 0 || diff < FLT_MIN) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < (EPSILON * FLT_MIN);
    } else { // use relative error
        return diff / fmin((absA + absB), FLT_MAX) < EPSILON;
    }
}

// Free the resources allocated during initialization
void cleanup() {
    if (load_kernel) {
        clReleaseKernel(load_kernel);
    }
    if (exec_kernel) {
        clReleaseKernel(exec_kernel);
    }
    if (store_kernel) {
        clReleaseKernel(store_kernel);
    }

    if (load_queue) {
        clReleaseCommandQueue(load_queue);
    }
    if (exec_queue) {
        clReleaseCommandQueue(exec_queue);
    }
    if (store_queue) {
        clReleaseCommandQueue(store_queue);
    }

    if(x_buf) {
        clReleaseMemObject(x_buf);
    }
    if(A_buf) {
        clReleaseMemObject(A_buf);
    }
    if(y_buf) {
        clReleaseMemObject(y_buf);
    }

    if(program) {
        clReleaseProgram(program);
    }
    if(context) {
        clReleaseContext(context);
    }
}