import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt

# Load library
lib = ctypes.cdll.LoadLibrary("./libconvolution.so")

lib.gpu_convolve.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# Edge detection filters
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

blur = np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]], dtype=np.float32) / 9.0

# Create test image (circle)
size = 512
image = np.zeros((size, size), dtype=np.float32)
center = size // 2
radius = size // 4
for i in range(size):
    for j in range(size):
        if (i - center)**2 + (j - center)**2 < radius**2:
            image[i, j] = 1.0

# Apply filters
output_x = np.zeros_like(image)
output_y = np.zeros_like(image)
output_blur = np.zeros_like(image)

print("Applying edge detection filters...")

start = time.time()
lib.gpu_convolve(image.ravel(), sobel_x.ravel(), output_x.ravel(), size, size, 3)
time_x = time.time() - start

start = time.time()
lib.gpu_convolve(image.ravel(), sobel_y.ravel(), output_y.ravel(), size, size, 3)
time_y = time.time() - start

start = time.time()
lib.gpu_convolve(image.ravel(), blur.ravel(), output_blur.ravel(), size, size, 3)
time_blur = time.time() - start

# Combine edge detection
edges = np.sqrt(output_x**2 + output_y**2)

print(f"\nPerformance (512x512 image):")
print(f"Sobel X: {time_x*1000:.3f} ms")
print(f"Sobel Y: {time_y*1000:.3f} ms")
print(f"Blur: {time_blur*1000:.3f} ms")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(output_x, cmap='gray')
axes[0, 1].set_title('Sobel X (Vertical Edges)')
axes[0, 2].imshow(output_y, cmap='gray')
axes[0, 2].set_title('Sobel Y (Horizontal Edges)')
axes[1, 0].imshow(edges, cmap='gray')
axes[1, 0].set_title('Combined Edge Detection')
axes[1, 1].imshow(output_blur, cmap='gray')
axes[1, 1].set_title('Blur Filter')
axes[1, 2].axis('off')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('convolution_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nConvolution demonstration complete!")
