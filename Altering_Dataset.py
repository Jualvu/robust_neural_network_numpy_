import numpy as np
from scipy.ndimage import shift, rotate, gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

# MNIST image shape: 28x28
def add_gaussian_noise(image, mean=0, std=0.2):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def shift_image(image, shift_range=2):
    dx = np.random.randint(-shift_range, shift_range + 1)
    dy = np.random.randint(-shift_range, shift_range + 1)
    return shift(image, [dy, dx], mode='constant', cval=0)

def rotate_image(image, angle_range=20):
    angle = np.random.uniform(-angle_range, angle_range)
    return rotate(image, angle, reshape=False, mode='constant', cval=0)

def blur_image(image, sigma=1.0):
    return gaussian_filter(image, sigma=sigma)

def augment_image(image):
    image = add_gaussian_noise(image)
    image = shift_image(image)
    image = rotate_image(image)
    image = blur_image(image)
    return np.clip(image, 0, 1)
