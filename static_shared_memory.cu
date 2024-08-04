//
// Created by joey on 2024/8/2.
//
#include "cstdlib"
#include "iostream"
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

__global__ void kernel(float* d_A, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ float s_arr[32];

    if(n < N){
        s_arr[tid] = d_A[n];
    }
    __syncthreads();

    if(tid == 0){
        for(int i = 0;i < 32;++i){
            printf("kernel: %f, blockIdx: %d\n", s_arr[i], bid);
        }
    }
}


int main(){
    int deID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deID);
    cout << "运行GPU设备：" << deviceProp.name << endl;

    int nElems = 64;
    int nBytes = nElems * sizeof(float);

    float *hA = nullptr;
    hA = (float*)malloc(nBytes);
    for(int i = 0;i < nElems;i++)
        hA[i] = float(i);
    float *dA = nullptr;
    cudaMalloc(&dA, nBytes);
    cudaMemcpy(dA, hA, nBytes, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(2);
    kernel<<<grid, block>>>(dA, 64);
    cudaFree(dA);
    free(hA);


    return 0;
}