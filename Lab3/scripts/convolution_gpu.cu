#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void convolveGPU(float *img, float *fil, float *out,
                            int W, int H, int F) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = F / 2;

    if (x < W && y < H) {
        float sum = 0.0f;
        for (int fy = 0; fy < F; fy++)
            for (int fx = 0; fx < F; fx++) {
                int iy = y - half + fy;
                int ix = x - half + fx;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                    sum += img[iy * W + ix] * fil[fy * F + fx];
            }
        out[y * W + x] = sum;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int F = (argc > 2) ? atoi(argv[2]) : 3;

    size_t imgSize = N * N * sizeof(float);
    size_t filSize = F * F * sizeof(float);

    float *h_img = (float*)malloc(imgSize);
    float *h_fil = (float*)malloc(filSize);
    float *h_out = (float*)malloc(imgSize);

    for (int i = 0; i < N * N; i++) h_img[i] = (i % 255) / 255.0f;
    for (int i = 0; i < F * F; i++) h_fil[i] = 1.0f / (F * F);

    float *d_img, *d_fil, *d_out;
    cudaMalloc(&d_img, imgSize);
    cudaMalloc(&d_fil, filSize);
    cudaMalloc(&d_out, imgSize);

    cudaMemcpy(d_img, h_img, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fil, h_fil, filSize, cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((N+15)/16, (N+15)/16);

    cudaEvent_t s,e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);

    convolveGPU<<<grid,block>>>(d_img,d_fil,d_out,N,N,F);
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms,s,e);

    printf("CUDA Convolution N=%d F=%d Time=%.3f ms\n", N, F, ms);

    cudaFree(d_img); cudaFree(d_fil); cudaFree(d_out);
    free(h_img); free(h_fil); free(h_out);
}
