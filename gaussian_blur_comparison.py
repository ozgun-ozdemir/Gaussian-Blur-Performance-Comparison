import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Create Gaussian Kernel
def blur_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Apply Gaussian Blur
def custom_blur(image, kernel_size, sigma):
    kernel = blur_kernel(kernel_size, sigma)
    height, width = image.shape[:2]
    
    # Padding
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2), (0, 0)), mode='constant', constant_values=0)
    blurred_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            
            for k in range(image.shape[2]): 
                blurred_image[i, j, k] = np.sum(region[:, :, k] * kernel)

    return blurred_image

image = cv2.imread('car.jpg')

if image is None:
    raise ValueError("Image could not be loaded!")

# Parameters
kernel_size = 11
sigma = 5

# Measure the time
start_time = time.time()
blurred_image = custom_blur(image, kernel_size, sigma)
custom_blur_time = time.time() - start_time

start_time = time.time()
opencv_blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
opencv_blur_time = time.time() - start_time

# Show results with Matplotlib
plt.figure(figsize=(14, 7))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Custom Gaussian Blur')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(opencv_blurred_image, cv2.COLOR_BGR2RGB))
plt.title('OpenCV Gaussian Blur')
plt.axis('off')

# Display the times 
plt.figtext(0.5, 0.2, f"Custom Gaussian Blur Time: {custom_blur_time:.6f} seconds", ha='center', va='center', fontsize=12, color='blue')
plt.figtext(0.5, 0.15, f"OpenCV Gaussian Blur Time: {opencv_blur_time:.6f} seconds", ha='center', va='center', fontsize=12, color='red')

plt.tight_layout()
plt.show()