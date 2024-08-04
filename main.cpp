#include <iostream>
#include "cudaComputeMatrix.cuh"

using namespace std;

int main() {
    const int M = 3;
    const int K = 3;
    const int N = 3;

    float A[M * K] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; //cuda里面只能传递数组 所以矩阵初始化的时候只能初始化成数组
    float B[K * N] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    float C[M * N];

    MatrixMultiply(A, B, C, M, N, K);

    cout << "result matrix c: " << endl;
    for(int i = 0;i < M;i++){
        for(int j = 0;j < N;j++){
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }


    return 0;
}
