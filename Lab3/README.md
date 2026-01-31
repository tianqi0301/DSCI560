This repository contains the work for Lab 3 of DSCI 560.

The lab focuses on implementing and optimizing matrix multiplication using CPU and GPU programming with CUDA. Different implementations were developed to compare performance and understand the impact of GPU acceleration and optimization techniques.

The lab includes:

- Implementing matrix multiplication on the CPU using C
- Implementing a naïve CUDA version and an optimized CUDA version using shared memory
- Using cuBLAS for highly optimized GPU matrix multiplication
- Comparing performance across CPU, naïve CUDA, optimized CUDA, and cuBLAS
- Creating CUDA shared libraries and calling them from Python using ctypes
- Implementing a CUDA-based convolution function for image processing

Libraries and tools used: C, CUDA, cuBLAS, Python, NumPy, ctypes, and Matplotlib.
