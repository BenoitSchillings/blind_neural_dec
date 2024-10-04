import numpy as np
from PIL import Image
from astropy.io import fits
from scipy.signal import fftconvolve
import os
import glob
from scipy.ndimage import shift
import random
import random
import time

random.seed(time.time())

def add_noise(image, noise_level=0.5):
    """Add Gaussian noise to the image"""
    noise = np.random.normal(0, noise_level, image.shape)
    return image + noise

def random_shift(image, max_shift=2):
    """Apply a random shift to the image"""
    dx, dy = np.random.randint(-max_shift, max_shift+1, size=2)
    return shift(image, (dy, dx), mode='constant', cval=0)

def random_crop(image, crop_size=(256, 256)):
    """Crop a random part of the image"""
    h, w = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    return image[top:bottom, left:right]

def oversize_image(image, padding=100):
    """Add padding around the image"""
    return np.pad(image, pad_width=padding, mode='reflect')

def generate_training_data(ref_dir, psf_dir, output_dir, num_samples=100, crop_size=(256, 256), padding=100):
    # Get list of reference images and PSFs
    ref_images = glob.glob(os.path.join(ref_dir, '*.png'))
    psfs = glob.glob(os.path.join(psf_dir, '*.fits'))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i in range(19000, 21500):
        # Randomly select a reference image
        ref_image_path = random.choice(ref_images)

        # Load reference image (PNG)
        ref_image = np.array(Image.open(ref_image_path).convert('L'), dtype=np.float32)

        # Ensure the reference image is large enough for cropping
        if ref_image.shape[0] < crop_size[0] + 2*padding or ref_image.shape[1] < crop_size[1] + 2*padding:
            print(f"Skipping {ref_image_path} - too small for cropping and padding")
            continue

        # Crop a random part of the reference image (larger than final crop size)
        large_crop_size = (crop_size[0] + 2*padding, crop_size[1] + 2*padding)
        cropped_ref = random_crop(ref_image, large_crop_size)

        # Oversize the cropped reference image
        oversized_ref = oversize_image(cropped_ref, padding)

        # Save sharp (cropped reference) image as FITS
        sharp_image = cropped_ref[padding:-padding, padding:-padding]
        sharp_output_path = os.path.join(output_dir, f'image_{i:04d}_sharp.fits')
        #print(sharp_image.shape)
        fits.writeto(sharp_output_path, sharp_image.astype(np.float32), overwrite=True)

        # Generate 7 blurred images
        for j in range(14):
            # Randomly select a PSF for each frame
            psf_path = random.choice(psfs)
            with fits.open(psf_path) as hdul:
                psf = hdul[0].data.astype(np.float32)

            # Normalize PSF
            psf /= np.sum(psf)

            # Ensure PSF is smaller than the cropped image
            if psf.shape[0] > crop_size[0] or psf.shape[1] > crop_size[1]:
                center = (psf.shape[0] // 2, psf.shape[1] // 2)
                psf = psf[center[0] - crop_size[0]//2 : center[0] + crop_size[0]//2,
                          center[1] - crop_size[1]//2 : center[1] + crop_size[1]//2]

            # Apply random shift to PSF
            shifted_psf = random_shift(psf)

            # Convolve oversized reference image with shifted PSF using FFT convolution
            blurred = fftconvolve(oversized_ref, shifted_psf, mode='same')

            # Crop the blurred image back to the original crop size
            blurred_cropped = blurred[2*padding:-2*padding, 2*padding:-2*padding]
            #print("bc", blurred_cropped.shape)
            # Add noise
            noisy_blurred = add_noise(blurred_cropped)

            # Save blurred image as FITS
            output_path = os.path.join(output_dir, f'image_{i:04d}_{j+1}.fits')
            fits.writeto(output_path, noisy_blurred.astype(np.float32), overwrite=True)

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} sample(s)")

    print(f"Generated {num_samples} samples in total")

# Usage
ref_dir = './ref_images/'
psf_dir = './psfs/'
output_dir = './training_data/'
num_samples = 7100  # Adjust this number as needed
crop_size = (960, 960)  # Adjust this size as needed
padding = 100  # Padding size for oversizing

generate_training_data(ref_dir, psf_dir, output_dir, num_samples, crop_size, padding)