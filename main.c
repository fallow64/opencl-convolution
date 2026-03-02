#include <stdio.h>
#include <stdlib.h>
#include "util.h"

// Utility struct for managing common OpenCL objects
typedef struct {
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
} OpenCL;

void init_opencl(OpenCL *opencl) {
    printf("Initializing OpenCL...\n");

    cl_uint num_platforms;
    cl_uint num_devices;
    check_error(clGetPlatformIDs(1, &opencl->platform_id, &num_platforms), "Getting platform");
    check_error(clGetDeviceIDs(opencl->platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &opencl->device_id, &num_devices), "Getting device");

    printf("Found %u platform(s) and %u device(s)\n", num_platforms, num_devices);

    opencl->context = clCreateContext(NULL, 1, &opencl->device_id, NULL, NULL, NULL);
    opencl->command_queue = clCreateCommandQueue(opencl->context, opencl->device_id, 0, NULL);
}

void cleanup_opencl(OpenCL *opencl) {
    clReleaseCommandQueue(opencl->command_queue);
    clReleaseContext(opencl->context);
}

void print_device_info(OpenCL *opencl) {
    char device_name[128];
    check_error(clGetDeviceInfo(opencl->device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL), "Getting device name");
    printf("Using device: %s\n", device_name);
}

int main() {
    // Seed for reproducibility
    srand(0);

    // Read the kernel source code from file
    char *kernelSource = read_file("kernels.cl");
    if (!kernelSource) {
        fprintf(stderr, "Failed to read kernel source\n");
        return 1;
    }

    // Size of the vectors
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int N = WIDTH * HEIGHT;
    
    // Allocate memory for the matrix and the kernel
    float* A = (float*)malloc(sizeof(float) * N);
    float* B = (float*)malloc(sizeof(float) * N);
    float* C = (float*)malloc(sizeof(float) * N);

    // Initialize A and B
    for (int i = 0; i < N; i++) {
        A[i] = rand() / (float)RAND_MAX * 100.0f;
        B[i] = rand() / (float)RAND_MAX * 100.0f;
    }

    // OpenCL initialization
    OpenCL opencl;
    init_opencl(&opencl);
    print_device_info(&opencl);

    // Create a program from the kernel source
    const char* kernelSourceConst = kernelSource; // clCreateProgramWithSource expects a const char**
    cl_program program = clCreateProgramWithSource(opencl.context, 1, &kernelSourceConst, NULL, NULL);

    // Create memory buffers on the device for each vector
    cl_mem bufferA = clCreateBuffer(opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(*A) * N, A, NULL);
    cl_mem bufferB = clCreateBuffer(opencl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(*B) * N, B, NULL);
    cl_mem bufferC = clCreateBuffer(opencl.context, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, NULL);

    // Build the program
    check_error(clBuildProgram(program, 1, &opencl.device_id, NULL, NULL, NULL), "Building program");

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

    // Set the arguments of the kernel
    check_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA), "Setting kernel argument 0");
    check_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB), "Setting kernel argument 1");
    check_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC), "Setting kernel argument 2");

    // Execute the kernel on the device
    size_t global_size = N;
    check_error(clEnqueueNDRangeKernel(opencl.command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL), "Enqueueing kernel");
    // Read the result back to host memory
    check_error(clEnqueueReadBuffer(opencl.command_queue, bufferC, CL_TRUE, 0, sizeof(float) * N, C, 0, NULL, NULL), "Reading back result");

    // Print the results
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    cleanup_opencl(&opencl);
    free(A);
    free(B);
    free(C);
    free(kernelSource);
    return 0;
}
