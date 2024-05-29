import cv2
import numpy as np
import matplotlib.pyplot as plt


def homomorphic_filter(img, d0=0.1, rL=0.5, rH=2.0, c=1.0):
    # Convert image to float32
    img_float = img.astype(np.float32)

    # Apply log transformation
    img_log = np.log1p(img_float)

    # Perform FFT
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)

    # Create Gaussian high-pass filter
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D_uv = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            mask[u, v] = (rH - rL) * (1 - np.exp(-c * (D_uv ** 2 / (d0 ** 2)))) + rL

    # Ensure mask has the same shape as img_fft_shift
    mask = mask[..., np.newaxis]

    # Apply filter to the FFT image
    img_fft_filt = img_fft_shift * mask

    # Perform inverse FFT
    img_ifft_shift = np.fft.ifftshift(img_fft_filt)
    img_ifft = np.fft.ifft2(img_ifft_shift)
    img_exp = np.expm1(np.real(img_ifft))

    # Normalize the image
    img_output = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return img_output


# Load your image
img_path = '1.jpg'  # Replace with your image path
img = cv2.imread(img_path)

# Split the image into R, G, B channels
b, g, r = cv2.split(img)

# Apply homomorphic filter to each channel
filtered_b = homomorphic_filter(b)
filtered_g = homomorphic_filter(g)
filtered_r = homomorphic_filter(r)

# Merge the filtered channels back into a color image
filtered_img = cv2.merge((filtered_b, filtered_g, filtered_r))

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Homomorphic Filtered Image')
plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
