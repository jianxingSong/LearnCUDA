//
// Created by joey on 2024/8/2.
//
#include "stdlib.h"
#include "iostream"
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

__device__ int d_x = 1;
__device__ int d_y[2];


__global__ void kernel(void){
    d_y[0] += d_x;
    d_y[1] += d_x;

    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}

int main(){
    int h_y[2] = {10, 20};
    cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2);

    dim3 block (1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2);
    cudaDeviceSynchronize();
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);


    return 0;
}
