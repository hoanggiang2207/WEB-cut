import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from matplotlib import pyplot as plt

def create_low_pass_filter(rows, cols, cutoff_frequency):
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 1
    return mask

def create_high_pass_filter(rows, cols, cutoff_frequency):
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-cutoff_frequency:crow+cutoff_frequency, ccol-cutoff_frequency:ccol+cutoff_frequency] = 0
    return mask

def load_image():
    global image_path, original_image, filtered_low_pass, filtered_high_pass
    image_path = filedialog.askopenfilename()
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_image = img
        f_transform = np.fft.fft2(img)
        f_transform_shifted = np.fft.fftshift(f_transform)
        rows, cols = img.shape
        low_pass_filter = create_low_pass_filter(rows, cols, 30)
        high_pass_filter = create_high_pass_filter(rows, cols, 30)
        f_low_pass = f_transform_shifted * low_pass_filter
        f_high_pass = f_transform_shifted * high_pass_filter
        f_inverse_low_pass = np.fft.ifftshift(f_low_pass)
        f_inverse_high_pass = np.fft.ifftshift(f_high_pass)
        filtered_low_pass = np.fft.ifft2(f_inverse_low_pass)
        filtered_high_pass = np.fft.ifft2(f_inverse_high_pass)
        filtered_low_pass = np.abs(filtered_low_pass)
        filtered_high_pass = np.abs(filtered_high_pass)
        display_images()

def display_images():
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 3, 1), plt.imshow(original_image, cmap='gray'), plt.title('Original Image')
    plt.subplot(2, 3, 2), plt.imshow(filtered_low_pass, cmap='gray'), plt.title('Low Pass Filter')
    plt.subplot(2, 3, 3), plt.imshow(filtered_high_pass, cmap='gray'), plt.title('High Pass Filter')
    plt.axis('off')
    plt.show()

def close_app():
    root.destroy()

root = tk.Tk()
root.title("Image Fourier Transform")

load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

close_button = tk.Button(root, text="Close App", command=close_app)
close_button.pack(pady=10)

root.mainloop()
