#include<stdio.h>
#include<cuda_runtime.h>
#include <sm_61_intrinsics.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define SHORT4(pointer) (reinterpret_cast<short4*>(&(pointer))[0])
#define SHORT2(pointer) (reinterpret_cast<short2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CHAR4(pointer) (reinterpret_cast<char4*>(&(pointer))[0])
__global__ void tiled_qu_int8(
    short * __restrict__ a, short * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ short2 s_a[BM][BK];
    __shared__ char4 s_b[BK][BN];

    float r_c[TM][TN] = {0.0};
    // short r_c0[TM][TN] = {0.0};
    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        s_a[load_a_smem_m][load_a_smem_k] = SHORT2(a[load_a_gmem_addr]);

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        s_b[load_b_smem_k][load_b_smem_n] = CHAR4(b[load_b_gmem_addr]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k+=4) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    // r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                    r_c[m][n] += __dp2a_lo(s_a[comp_a_smem_m][k], s_b[k][comp_b_smem_n],  0);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

__global__ void tiled_qu_int16(
    short * __restrict__ a, short * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ short s_a[BM][BK];
    __shared__ short s_b[BK][BN];

    short r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j += 1) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            c[store_c_gmem_addr] = r_c[i][j];
        }
    }
}

int main(){
    int M = 1024;
    int N = 1024;
    int K = 1024;
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    short *a, *b;
    float *c;
    a = (short*) malloc(M * K * sizeof(short));
    b = (short*) malloc(K * N * sizeof(short));
    c = (float*) malloc(M * N * sizeof(float));
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
    float *c_d;
    cudaMalloc(&a_d, M * K * sizeof(short));
    cudaMalloc(&b_d, K * N * sizeof(short));
    cudaMalloc(&c_d, M * N * sizeof(float));
    cudaMemcpy(a_d, a, M * K * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, K * N * sizeof(short), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_d, c, M * N * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    tiled_qu_int8<<<grid, block>>>(a_d, b_d, c_d,M,N,K);
    cudaMemcpy(c, c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%d\n", c[33]);
    free(a);
    free(b);
    free(c);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}