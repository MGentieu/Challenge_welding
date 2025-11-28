# pt_dataloader.py
import os
from pathlib import Path
from typing import Callable, Optional

import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

# -------------------------
# Ta classe Dataset (inchangée)
# -------------------------
class ImageDataFrameDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path = ".",
        path_col: str = "path",
        label_col: str = "label",
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        channels_first: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.label_transform = label_transform
        self.channels_first = channels_first

        if self.transform is None:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = self.root_dir / row[self.path_col]
        label = row[self.label_col]

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        # Réorganisation éventuelle des canaux
        if isinstance(image, torch.Tensor) and image.ndim == 3:
            if self.channels_first:
                if image.shape[0] != 3 and image.shape[-1] == 3:
                    image = image.permute(2, 0, 1)
            else:
                if image.shape[0] == 3:
                    image = image.permute(1, 2, 0)

        if self.label_transform is not None:
            label = self.label_transform(label)

        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        return image, label

# -------------------------
# Corruptions (train: aléatoire)
# -------------------------
class RandomCorruptions:
    """Applique des corruptions stochastiques avec sévérité choisie aléatoirement (1..3)."""

    def __init__(self):
        pass

    def __call__(self, img: Image.Image) -> Image.Image:
        severity = random.choice([1, 2, 3])

        # Brightness
        if random.random() < 0.7:
            delta = {1: 0.1, 2: 0.2, 3: 0.35}[severity]
            img = F.adjust_brightness(img, 1 + random.uniform(-delta, delta))

        # Contrast
        if random.random() < 0.7:
            low, high = {1: (0.9, 1.1), 2: (0.8, 1.2), 3: (0.6, 1.4)}[severity]
            img = F.adjust_contrast(img, random.uniform(low, high))

        # Blur (Gaussian)
        if random.random() < 0.4:
            radius = {1: 1, 2: 2, 3: 3}[severity]
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Gaussian noise (appliqué en numpy)
        if random.random() < 0.6:
            std = {1: 0.02, 2: 0.05, 3: 0.08}[severity]
            img_np = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0.0, 1.0)
            img = Image.fromarray((img_np * 255).astype(np.uint8))

        # Optionnel: légère teinte / saturation jitter
        if random.random() < 0.3:
            hue_delta = random.uniform(-0.02, 0.02)
            img = F.adjust_hue(img, hue_delta)
        if random.random() < 0.3:
            sat_factor = random.uniform(0.9, 1.1)
            img = F.adjust_saturation(img, sat_factor)

        return img

# -------------------------
# Corruptions déterministes pour test (sévérité 1/2/3)
# -------------------------
class DeterministicCorruption:
    def __init__(self, severity: int = 1):
        assert severity in (1, 2, 3)
        self.s = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        s = self.s

        # Brightness (ajout fixe)
        delta = {1: 0.1, 2: 0.2, 3: 0.35}[s]
        img = F.adjust_brightness(img, 1 + delta)

        # Contrast (valeur moyenne)
        low, high = {1: (0.9, 1.1), 2: (0.8, 1.2), 3: (0.6, 1.4)}[s]
        img = F.adjust_contrast(img, (low + high) / 2.0)

        # Blur
        radius = {1: 1, 2: 2, 3: 3}[s]
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Noise (fixé par seed pour reproducibilité si souhaité)
        std = {1: 0.02, 2: 0.05, 3: 0.08}[s]
        # NOTE: on utilise np.random.default_rng() si on veut reproduire avec seed externe
        img_np = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, std, img_np.shape).astype(np.float32)
        img_np = np.clip(img_np + noise, 0.0, 1.0)
        img = Image.fromarray((img_np * 255).astype(np.uint8))

        return img

# -------------------------
# Helpers: créer transforms pour train / test
# -------------------------
def get_train_transform(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        RandomCorruptions(),
        transforms.ToTensor(),
        # Optionnel: normalisation si ton modèle l'attend
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

def get_test_transform(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        # transforms.Normalize(...)
    ])

def get_test_corrupted_transform(severity: int, img_size=(224,

'''

class ImageDataFrameDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path = ".",
        path_col: str = "path",
        label_col: str = "label",
        transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        channels_first: bool = True,
    ):
        """
        df : DataFrame contenant au moins les colonnes `path` et `label`
        root_dir : dossier racine des images (si `path` est relatif)
        transform : transformations sur l'image (e.g. torchvision.transforms)
                    Si None, on applique par défaut ToTensor() -> (C,H,W).
        label_transform : optionnel, pour encoder les labels (e.g. mapping str->int)
        channels_first : 
            - True  -> retourne les images au format (C, H, W) (PyTorch standard)
            - False -> retourne les images au format (H, W, C)
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform
        self.label_transform = label_transform
        self.channels_first = channels_first

        # Transform par défaut si rien n'est fourni
        if self.transform is None:
            self.transform = transforms.ToTensor()  # PIL -> Tensor (C,H,W), [0,1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = self.root_dir / row[self.path_col]
        label = row[self.label_col]

        # Chargement de l'image avec PIL (H,W,C implicite)
        image = Image.open(img_path).convert("RGB")

        # Application des transforms (souvent -> Tensor (C,H,W) avec ToTensor)
        image = self.transform(image)

        # Réorganisation éventuelle des canaux
        if isinstance(image, torch.Tensor) and image.ndim == 3:
            # image.shape = (C,H,W) ou (H,W,C)
            if self.channels_first:
                # On veut (C,H,W)
                if image.shape[0] != 3 and image.shape[-1] == 3:
                    # Cas où l'image serait (H,W,C) par erreur
                    image = image.permute(2, 0, 1)
            else:
                # On veut (H,W,C)
                if image.shape[0] == 3:
                    # Cas standard torchvision : (C,H,W) -> (H,W,C)
                    image = image.permute(1, 2, 0)

        # Transform label si besoin
        if self.label_transform is not None:
            label = self.label_transform(label)

        # Si label est un entier, on le convertit en tensor long
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        return image, label

'''