//
// Created by joey on 2024/8/3.
//
#include "iostream"
#include "cuda_runtime.h"
#include "helper_cuda.h"

using namespace std;

__global__ void kernel(void){

}

int main(){
    int deID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deID);
    cout << "运行GPU设备：" << deviceProp.name << endl;


    if(deviceProp.globalL1CacheSupported){
        cout << "支持L1缓存" << endl;
    } else {
        cout << "不支持L1缓存" << endl;
    }
    cout << "L2缓存大小：" << deviceProp.l2CacheSize / (1024 * 1024) << "M" << endl;

    return 0;
}

