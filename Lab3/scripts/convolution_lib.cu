#include <cuda_runtime.h>
#include <stdio.h>

// 2D Convolution kernel
__global__ void convolveGPU(float *image, float *filter, float *output,
                            int imageWidth, int imageHeight,
                            int filterSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < imageHeight && col < imageWidth) {
        float sum = 0.0f;
        int halfFilter = filterSize / 2;

        for (int fRow = 0; fRow < filterSize; fRow++) {
            for (int fCol = 0; fCol < filterSize; fCol++) {
                int imageRow = row - halfFilter + fRow;
                int imageCol = col - halfFilter + fCol;

                // Handle boundaries (zero padding)
                if (imageRow >= 0 && imageRow < imageHeight &&
                    imageCol >= 0 && imageCol < imageWidth) {
                    sum += image[imageRow * imageWidth + imageCol] *
                           filter[fRow * filterSize + fCol];
                }
            }
        }
        output[row * imageWidth + col] = sum;
    }
}

// Exposed C function for Python
extern "C" void gpu_convolve(float *h_image, float *h_filter, float *h_output,
                             int imageWidth, int imageHeight, int filterSize) {
    size_t imageSize = imageWidth * imageHeight * sizeof(float);
    size_t filterSize_bytes = filterSize * filterSize * sizeof(float);

    float *d_image, *d_filter, *d_output;
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&d_filter, filterSize_bytes);
    cudaMalloc((void**)&d_output, imageSize);

    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize_bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((imageWidth + 15) / 16, (imageHeight + 15) / 16);

    convolveGPU<<<dimGrid, dimBlock>>>(d_image, d_filter, d_output,
                                       imageWidth, imageHeight, filterSize);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_filter);
    cudaFree(d_output);
}
