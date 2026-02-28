"""
Image preprocessing for YOLO object detection training.
Supports 5 model types: RGB, REPLACEMENT, HSV_WAVELET, DIRECTIONAL, FUSION
Each type creates a different 3-channel image from the original BGR input.
"""

import cv2
import numpy as np
import pywt


def get_wavelet_energy(img, wavelet='db4'):
    """Compute single-channel wavelet energy map. Used by REPLACEMENT and FUSION."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    LL, (LH, HL, HH) = pywt.dwt2(gray, wavelet)
    energy = np.abs(LH) + np.abs(HL) + np.abs(HH)
    norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    return cv2.resize((norm * 255).astype(np.uint8), (img.shape[1], img.shape[0]))


def create_replacement_image(img):
    """Channel Replacement: [B, G, Wavelet] - Red channel replaced by wavelet energy."""
    wavelet = get_wavelet_energy(img)
    b, g, r = cv2.split(img)
    return cv2.merge([b, g, wavelet])


def create_hsv_wavelet_image(img):
    """HSV Wavelet: [H, S, Wavelet] - Value channel replaced by wavelet energy."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(gray, 'db4')
    energy = np.abs(LH) + np.abs(HL) + np.abs(HH)
    norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    wavelet_uint8 = (norm * 255).astype(np.uint8)
    wavelet_resized = cv2.resize(wavelet_uint8, (img.shape[1], img.shape[0]))

    return cv2.merge([h, s, wavelet_resized])


def create_directional_image(img):
    """Directional Wavelet: [LL, LH, HL] - 3 wavelet sub-bands as channels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    LL1, (LH1, HL1, HH1) = pywt.dwt2(gray, 'db4')

    def normalize(band):
        band = np.abs(band)
        norm = (band - band.min()) / (band.max() - band.min() + 1e-8)
        return cv2.resize((norm * 255).astype(np.uint8), (img.shape[1], img.shape[0]))

    return cv2.merge([normalize(LL1), normalize(LH1), normalize(HL1)])


def create_fusion_image(img):
    """Image Fusion: 70% RGB + 30% Wavelet energy (3-channel blend)."""
    wavelet = get_wavelet_energy(img)
    wavelet_3c = cv2.merge([wavelet, wavelet, wavelet])
    return cv2.addWeighted(img, 0.7, wavelet_3c, 0.3, 0)


def process_image(img_path, model_type):
    """
    Load and process a single image according to the specified model type.

    Args:
        img_path: Path to the image file
        model_type: One of 'RGB', 'REPLACEMENT', 'HSV_WAVELET', 'DIRECTIONAL', 'FUSION'

    Returns:
        Processed BGR image (3 channels) or None if loading fails
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    processors = {
        'RGB': lambda x: x,
        'REPLACEMENT': create_replacement_image,
        'HSV_WAVELET': create_hsv_wavelet_image,
        'DIRECTIONAL': create_directional_image,
        'FUSION': create_fusion_image,
    }

    processor = processors.get(model_type.upper())
    if processor is None:
        raise ValueError(f"Unknown detection model type: {model_type}. "
                         f"Choose from: {list(processors.keys())}")

    return processor(img)