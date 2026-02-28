import pywt
import cv2
import numpy as np

def compute_wavelet_transform(image, wavelet='db4', level=2):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    coeffs = pywt.wavedec2(gray, wavelet, level=level)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]

    energy = np.abs(LH) + np.abs(HL) + np.abs(HH)

    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    energy_uint8 = (energy * 255).astype(np.uint8)

    return cv2.resize(energy_uint8, (320, 320))

def compute_wavelet_transform_dual(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    coeffs1 = pywt.dwt2(gray, 'db4')
    LL1, (_, _, _) = coeffs1
    coeffs2 = pywt.dwt2(LL1, 'db4')
    LL2, (LH2, HL2, HH2) = coeffs2

    target_size = (320, 320)
    LL2_resized = cv2.resize(LL2, target_size)
    LH2_resized = cv2.resize(LH2, target_size)
    HL2_resized = cv2.resize(HL2, target_size)
    HH2_resized = cv2.resize(HH2, target_size)

    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    wavelet_map = np.stack([normalize(LL2_resized), normalize(LH2_resized),
                            normalize(HL2_resized), normalize(HH2_resized)], axis=0)
    return wavelet_map.astype(np.float32)