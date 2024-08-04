//
// Created by joey on 2024/8/4.
//

#include "iostream"
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

__global__ void addKernel(int* a, int* b, int* c, const int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N * N){
        c[idx] = a[idx] + b[idx];
    }
}

void setValue(int* m, int N){
    for(int i = 0;i < N;i++){
        for(int j = 0;j < N;j++){
            m[i * N + j] = 1;
        }
    }
}

int main(){
    const int N = 512;
    int* h_a = new int[N * N];
    int* h_b = new int[N * N];
    int* h_c = new int[N * N];
    setValue(h_a, N);
    setValue(h_b, N);
    setValue(h_c, N);

    int* d_a, *d_b, *d_c;
    size_t size = N * N * sizeof(int);
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((N * N - 1 + 32) / 32);
    addKernel<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    for(int i = 0;i < N;i++){
        for(int j = 0;j < N;j++){
            cout << h_c[i * N + j] << " ";
        }
        cout << endl;
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}