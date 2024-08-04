//
// Created by joey on 2024/8/3.
//
#include "iostream"
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void kernel(){
    printf("Constant data c_data = %.2f.\n", c_data);
}

int main(){
    int deID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deID);
    cout << "运行GPU设备：" << deviceProp.name << endl;

    float h_data = 8.8f;
    cudaMemcpyToSymbol(c_data, &h_data, sizeof(float));

    dim3 grid(1);
    dim3 block(1);
    kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float));
    printf("Costant data h_data = %.2f.\n", h_data);

    cudaDeviceReset();


    return 0;
}

