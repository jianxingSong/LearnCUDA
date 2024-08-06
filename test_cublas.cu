//
// Created by joey on 2024/8/6.
//
#include "iostream"
#include <cublas.h>
#include "cublas_v2.h"
#include "thread"

using namespace std;

#define N 1024

int main(){
    float *h_a = (float*)malloc(N*N*sizeof(float));
    float *h_b = (float*)malloc(N*N*sizeof(float));
    float *h_c = (float*)malloc(N*N*sizeof(float));
    float* d_a, *d_b, *d_c;

    // 测速
    cublasInit();
    for(int i = 0;i < N * N;i++){
        h_a[i] = 2;
        h_b[i] = 2;
    }
//    cublasAlloc(N * N, sizeof(float), (void**)&d_a);
//    cublasAlloc(N * N, sizeof(float), (void**)&d_b);
//    cublasAlloc(N * N, sizeof(float), (void**)&d_c);
    cudaMalloc((void**)&d_a, sizeof(float) * N * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N * N);
    cudaMalloc((void**)&d_c, sizeof(float) * N * N);
    size_t size = sizeof(float) * N * N;
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0;i < N;i++){
        for(int j = 0;j < N;j++){
            cout << h_c[i * N + j] << " ";
        }
        cout << endl;
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}
