#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CPU convolution
void convolveCPU(float *image, float *filter, float *output,
                 int imageWidth, int imageHeight, int filterSize) {

    int half = filterSize / 2;

    for (int row = 0; row < imageHeight; row++) {
        for (int col = 0; col < imageWidth; col++) {
            float sum = 0.0f;

            for (int fr = 0; fr < filterSize; fr++) {
                for (int fc = 0; fc < filterSize; fc++) {
                    int r = row - half + fr;
                    int c = col - half + fc;

                    if (r >= 0 && r < imageHeight &&
                        c >= 0 && c < imageWidth) {
                        sum += image[r * imageWidth + c] *
                               filter[fr * filterSize + fc];
                    }
                }
            }
            output[row * imageWidth + col] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int filterSize = (argc > 2) ? atoi(argv[2]) : 3;

    size_t imageBytes = N * N * sizeof(float);
    size_t filterBytes = filterSize * filterSize * sizeof(float);

    float *image = (float *)malloc(imageBytes);
    float *output = (float *)malloc(imageBytes);
    float *filter = (float *)malloc(filterBytes);

    for (int i = 0; i < N * N; i++)
        image[i] = (i % 255) / 255.0f;

    for (int i = 0; i < filterSize * filterSize; i++)
        filter[i] = 1.0f / (filterSize * filterSize);

    clock_t start = clock();
    convolveCPU(image, filter, output, N, N, filterSize);
    clock_t end = clock();

    double timeSec = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU Convolution: Image=%dx%d Filter=%dx%d Time=%.4f sec\n",
           N, N, filterSize, filterSize, timeSec);

    free(image);
    free(output);
    free(filter);
    return 0;
}
