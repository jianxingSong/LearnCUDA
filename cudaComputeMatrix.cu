//
// Created by joey on 2024/6/20.
//

#include "cudaComputeMatrix.cuh"

__global__ void MatrixMulKernel(float* d_A, float* d_B, float* d_C, int M, int N, int K){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if((Row < M) && (Col < N)){
        float Cvalue = 0.0;
        for(int k = 0;k < K;++k){
            Cvalue += d_A[Row * K + k] * d_B[k * N * Col];
        }
        d_C[Row * N + Col] = Cvalue;
    }
}

void MatrixMultiply(float* h_A, float* h_B, float* h_C, int M, int N, int K){
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    //分配设备内存
    float* d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    //从cpu拷贝矩阵到gpu
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    //一个核函数分配一个grid，一个grid划分为n个block，每个block里面有很多个线程
    dim3 dimBlock(16, 16); //设置一共有16*16个block
    dim3 dimGrid((N + 16 - 1) / 16, (M + 16 - 1) / 16); //每个block中这么多个线程，要确保每个元素都能对应一个线程

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}