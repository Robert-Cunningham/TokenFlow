import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def generate_circular_gradient(size):
    # Define a blank array with zeros, now with dtype uint8
    image = np.zeros((size, size, 3), dtype=np.uint8)

    # Define the center of the image
    center = [size // 2, size // 2]

    # Create the gradient
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Hue
            angle = np.arctan2(center[0] - x, center[1] - y)
            h = (angle + np.pi) / (2 * np.pi)  # Adjust range to [0, 1]
            # Saturation: use distance from center
            s = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) / (np.sqrt(2) * size // 2)
            s = np.clip(s, 0, 1)  # Ensure saturation stays within [0, 1]
            # Value / Brightness
            v = 1  # Keep constant
            # Convert HSV to RGB and then scale to [0, 255]
            rgb = matplotlib.colors.hsv_to_rgb([h, s, v])
            image[x, y] = np.array(rgb * 255, dtype=np.uint8)

    return Image.fromarray(image)


from math import ceil


def display_images(images):
    w, h = 6, ceil(len(images) / 6)
    plt.figure(figsize=(w * 8, h * 8))  # specifying the overall grid size

    for i in range(len(images)):
        plt.subplot(h, w, i + 1)  # the number of images in the grid is 5*5 (25)
        plt.axis("off")
        plt.imshow(images[i])

    plt.show()
