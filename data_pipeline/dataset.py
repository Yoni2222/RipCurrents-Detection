"""
PyTorch Dataset class for rip current detection
Import this into training script
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from utils.wavelet_utils import compute_wavelet_transform, compute_wavelet_transform_dual
import torchvision.transforms as transforms
import random
import pywt

IMG_SIZE = 320


class ThreeChannelDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        pos_dir = self.root_dir / split / 'positive'
        neg_dir = self.root_dir / split / 'negative'
        self.images = list(pos_dir.glob('*')) + list(neg_dir.glob('*'))
        self.labels = [1] * len(list(pos_dir.glob('*'))) + [0] * len(list(neg_dir.glob('*')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = str(self.images[idx])
        label = self.labels[idx]
        bgr = cv2.imread(path)
        if bgr is None: bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        if self.transform:
            img_pil = self.transform(img_pil)

        return img_pil, torch.tensor(label, dtype=torch.float32)


# 2. RGB + Wavelet Dataset (4 Channels Single Stream)
class RGBWaveletDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        pos_dir = self.root_dir / split / 'positive'
        neg_dir = self.root_dir / split / 'negative'
        self.images = list(pos_dir.glob('*')) + list(neg_dir.glob('*'))
        self.labels = [1] * len(list(pos_dir.glob('*'))) + [0] * len(list(neg_dir.glob('*')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = str(self.images[idx])
        label = self.labels[idx]
        bgr = cv2.imread(path)
        if bgr is None: bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        if self.transform:
            img_aug_pil = self.transform(img_pil)
        else:
            img_aug_pil = img_pil

        rgb_tensor = transforms.ToTensor()(img_aug_pil)

        img_aug_np = np.array(img_aug_pil)
        gray = cv2.cvtColor(img_aug_np, cv2.COLOR_RGB2GRAY)
        coeffs = pywt.wavedec2(gray, 'db2', level=2)
        LL, (LH, HL, HH) = coeffs[0], coeffs[1]
        energy = np.abs(LH) + np.abs(HL) + np.abs(HH)
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        energy = cv2.resize(energy, (img_aug_np.shape[1], img_aug_np.shape[0]))
        wavelet_tensor = torch.tensor(energy, dtype=torch.float32).unsqueeze(0)

        final_tensor = torch.cat([rgb_tensor, wavelet_tensor], dim=0)
        return final_tensor, torch.tensor(label, dtype=torch.float32)


# 3. Four Channel Wavelet Only (Single Stream)
class FourChannelWaveletDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        pos_dir = self.root_dir / split / 'positive'
        neg_dir = self.root_dir / split / 'negative'
        self.images = list(pos_dir.glob('*')) + list(neg_dir.glob('*'))
        self.labels = [1] * len(list(pos_dir.glob('*'))) + [0] * len(list(neg_dir.glob('*')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = str(self.images[idx])
        label = self.labels[idx]
        bgr = cv2.imread(path)
        if bgr is None: bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)

        if self.transform:
            img_pil = self.transform(img_pil)

        aug_img = np.array(img_pil)
        gray = cv2.cvtColor(aug_img, cv2.COLOR_RGB2GRAY)

        coeffs1 = pywt.dwt2(gray, 'db2')
        LL1, (_, _, _) = coeffs1
        coeffs2 = pywt.dwt2(LL1, 'db2')
        LL2, (LH2, HL2, HH2) = coeffs2

        def prep(c):
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            return cv2.resize(c, (IMG_SIZE, IMG_SIZE))

        tensor = torch.stack([
            torch.tensor(prep(LL2), dtype=torch.float32),
            torch.tensor(prep(LH2), dtype=torch.float32),
            torch.tensor(prep(HL2), dtype=torch.float32),
            torch.tensor(prep(HH2), dtype=torch.float32)
        ])
        return tensor, torch.tensor(label, dtype=torch.float32)


class ChannelReplacementDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        pos_dir = self.root_dir / split / 'positive'
        neg_dir = self.root_dir / split / 'negative'

        self.images = list(pos_dir.glob('*')) + list(neg_dir.glob('*'))
        self.labels = [1]*len(list(pos_dir.glob('*'))) + [0]*len(list(neg_dir.glob('*')))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        path = str(self.images[idx])
        label = self.labels[idx]

        bgr = cv2.imread(path)
        if bgr is None: bgr = np.zeros((640, 640, 3), dtype=np.uint8)

        h, w, _ = bgr.shape
        crop_h = int(h * 0.8)
        bgr = bgr[0:crop_h, 0:w]

        wavelet_channel = compute_wavelet_transform(bgr, wavelet='db2')
        #print(f"wavelet: {wavelet_channel.shape}")
        bgr = cv2.resize(bgr, (320, 320))

        b, g, r = cv2.split(bgr)
        #print(f"b: {b.shape}")
        hybrid_img = cv2.merge([wavelet_channel, g, b])

        img_pil = Image.fromarray(hybrid_img)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.float32)

    """
        def __getitem__(self, idx):
        path = str(self.images[idx])
        filename = self.images[idx].name
        label = self.labels[idx]

        # 1. טעינה (רזולוציה מלאה)
        bgr = cv2.imread(path)
        if bgr is None: bgr = np.zeros((640, 640, 3), dtype=np.uint8)

        # 2. חיתוך החול (CROP) - על רזולוציה מלאה!
        h, w, _ = bgr.shape
        crop_h = int(h * 0.8)
        bgr = bgr[0:crop_h, 0:w]

        # 3. חישוב וייבלט (על מים ברזולוציה גבוהה)
        wavelet_channel = compute_wavelet_transform(bgr)

        # 4. הקטנת ה-RGB (Resize)
        bgr = cv2.resize(bgr, (320, 320))

        # 5. פירוק ומיזוג
        b, g, r = cv2.split(bgr)
        hybrid_img = cv2.merge([wavelet_channel, g, b])

        # 6. המרה ל-PIL והפעלת הטרנספורמים (היפוך + טנסור)
        img_pil = Image.fromarray(hybrid_img)

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        return img_tensor, torch.tensor(label, dtype=torch.float32), filename
    """



class DualStreamDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.positive_images = list((self.root_dir / split / 'positive').glob('*'))
        self.negative_images = list((self.root_dir / split / 'negative').glob('*'))
        self.images = self.positive_images + self.negative_images
        self.labels = [1] * len(self.positive_images) + [0] * len(self.negative_images)

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        filename = img_path.name
        label = self.labels[idx]

        bgr = cv2.imread(str(img_path))
        if bgr is None: bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w, _ = rgb.shape
        crop_h = int(h * 0.8)
        rgb = rgb[0:crop_h, 0:w]

        wavelet_np = compute_wavelet_transform_dual(rgb)
        rgb = cv2.resize(rgb, (320, 320))

        if self.split == 'train':
            if random.random() > 0.5:
                rgb = cv2.flip(rgb, 1)
                wavelet_np = np.flip(wavelet_np, axis=2).copy()

        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        wavelet_tensor = torch.from_numpy(wavelet_np).float()

        return rgb_tensor, wavelet_tensor, torch.tensor(label, dtype=torch.float32), filename




def get_dataset_class(model_type):
    if model_type == 'rgb_baseline':
        return ThreeChannelDataset
    elif model_type == 'single_rgb_wavelet':
        return RGBWaveletDataset
    elif model_type == 'single_wavelet_only':
        return FourChannelWaveletDataset
    elif model_type == 'channel_replacement':
        return ChannelReplacementDataset
    elif 'dual_stream' in model_type:
        return DualStreamDataset  # All dual streams use the same dataset loader
    else:
        raise ValueError(f"Unknown model_type: {model_type}")