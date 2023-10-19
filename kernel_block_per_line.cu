#include<stdio.h>
#include<cuda_runtime.h>
#include <sm_61_intrinsics.h>

#define THREAD_PRE_BLOCK 32 
#define THREAD_M 32
#define THREAD_N 32
#define THREAD_K 32
#define M 1024
#define N 1024
#define K 1024
#define FETCH_CHAR2(ptr)(reinterpret_cast<char2*>(&(ptr))[0])
#define FETCH_CHAR(ptr)(reinterpret_cast<char*>(&(ptr))[0])
#define FETCH_SH2(ptr)(reinterpret_cast<short2*>(&(ptr))[0])
__global__ void gemm(short*a, short *b, int *c){
 
    int res=0;
    if(blockIdx.x % 2 == 0){
        __shared__ char a_tmp[N];
        __shared__ short a_tmp[N];
        for(int i = 0; i < K/N; i++){
            a_tmp[threadIdx.x] = a[blockIdx.x * K + threadIdx.x + i * N];
            __syncthreads();
            for(int j = 0; j < N; j++){
                res += a_tmp[j] * b[(i * N + j) * N + threadIdx.x] + 0;
            }
            __syncthreads();
        }
        
    }
    else{
        __shared__ char4 a_tmp[N/2];
        short2 b_tmp;
        // char4 a_tmp;
        if(threadIdx.x < N/2){
            a_tmp[threadIdx.x].x =  FETCH_CHAR2(a[blockIdx.x * K + threadIdx.x * 2]).x;
            a_tmp[threadIdx.x].y =  FETCH_CHAR2(a[blockIdx.x * K + threadIdx.x * 2 + 1]).x;
        }
        __syncthreads();
        for(int j = 0; j < K/2; j++){
            b_tmp.x = b[j * N * 2 + threadIdx.x];
            b_tmp.y = b[(j * 2 + 1) * N + threadIdx.x];
            res += __dp2a_lo(b_tmp, a_tmp[j],  0);
        }
        __syncthreads();
        }
    // }
    c[blockIdx.x * N + threadIdx.x] = res;
    // __shared__ char a_tmp_8[THREAD_PRE_BLOCK][THREAD_K];
   
   
}


int main(){
    short *a, *b;
    int *c;
    a = (short*) malloc(M * K * sizeof(short));
    b = (short*) malloc(K * N * sizeof(short));
    c = (int*) malloc(M * N * sizeof(int));
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            a[i*K + j] = 1;
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            b[i*N + j] =  1;
        }
        
    }
    short *a_d, *b_d;
    int *c_d;
    cudaMalloc(&a_d, M * K * sizeof(short));
    cudaMalloc(&b_d, K * N * sizeof(short));
    cudaMalloc(&c_d, M * N * sizeof(int));
    cudaMemcpy(a_d, a, M * K * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, K * N * sizeof(short), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_d, c, M * N * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(N);
    dim3 grid(M, 1);
    gemm<<<grid, block>>>(a_d, b_d, c_d);
    cudaMemcpy(c, c_d, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", c[0]);
    free(a);
    free(b);
    free(c);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}
