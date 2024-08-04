//
// Created by joey on 2024/6/20.
//

#ifndef CUDACOMPUTEMATRIX_CUDACOMPUTEMATRIX_CUH
#define CUDACOMPUTEMATRIX_CUDACOMPUTEMATRIX_CUH

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include <device_launch_parameters.h>

using namespace std;

__global__ void MatrixMulKernel(float* d_A, float* d_B, float* d_C, int M, int N, int K);

void MatrixMultiply(float* h_A, float* h_B, float* h_C, int M, int N, int K);



#endif //CUDACOMPUTEMATRIX_CUDACOMPUTEMATRIX_CUH
