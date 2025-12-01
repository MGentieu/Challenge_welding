import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import WeldingAugmentation
from torchvision.transforms import InterpolationMode

def train_augmentations():
    """
    Pipeline d'augmentations pour les images industrielles.
    Utilisé uniquement lorsque is_train=True.
    """
    return transforms.Compose([
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.02
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomPosterize(bits=5, p=0.2),
        transforms.ToTensor(),
    ])


def test_transform():
    """Transform minimal sans augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])

class ImageDataFrameDataset(Dataset):
    """
    Dataset PyTorch qui combine :
      - WeldinAugmentation (base, agressive, rare_zones)
      - PyTorch transforms (jitter, blur, lumière)
      - Sur-échantillonnage KO et zones rares
    """
    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path = ".",
        path_col: str = "path",
        label_col: str = "label",
        label_transform: Optional[Callable] = None,
        channels_first: bool = True,
        is_train: bool = True,
        welding_mode: str = "balanced"
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.label_transform = label_transform
        self.channels_first = channels_first
        self.is_train = is_train

        # Augmenteur Welding
        self.welding_augmentor = WeldingAugmentation(mode=welding_mode)

        if is_train:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                transforms.GaussianBlur(kernel_size=(3,5), sigma=(0.1,2.0)),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomPosterize(bits=5, p=0.2),

                # --- Ajout des transformations demandées ---
                transforms.Resize(
                    size=(224, 224),
                    interpolation=InterpolationMode.BILINEAR,
                    max_size=None,
                    antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Mode validation/test → uniquement resize + normalisation (comme demandé)
            self.transform = transforms.Compose([
                transforms.Resize(
                    size=(224, 224),
                    interpolation=InterpolationMode.BILINEAR,
                    max_size=None,
                    antialias=True
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


        # Préparer indices avec sur-échantillonnage KO et zones rares
        self.indices = self._prepare_balanced_indices()

    def _prepare_balanced_indices(self):
        indices = []
        class_counts = self.df[self.label_col].value_counts()
        n_ok = class_counts.get('OK', 0)
        n_ko = class_counts.get('KO', 0)
        augmentation_factor = max(1, n_ok // n_ko) if n_ko > 0 else 1

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            indices.append((idx, False))  # image originale

            # Sur-échantillonner KO
            if row[self.label_col] == 'KO':
                for _ in range(augmentation_factor - 1):
                    indices.append((idx, True))

            # Sur-échantillonner zones rares
            is_rare = (
                (row.get('blur_level',0) > 3000) or 
                (row.get('luminosity_level',0) < 25) or 
                (row.get('luminosity_level',0) > 55) or
                (row.get('welding-seams','') == 'c102')
            )
            if is_rare:
                for _ in range(2):
                    indices.append((idx, True))

        return indices


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx, should_augment = self.indices[idx]
        row = self.df.iloc[real_idx]

        # Charger l'image
        img_path = self.root_dir / row[self.path_col]
        image = np.array(Image.open(img_path).convert("RGB"))

        # Appliquer WeldingAugmentation si nécessaire
        if self.is_train and should_augment:
            image = self.welding_augmentor.augment_image(
            image,
            class_label=row.get(self.label_col, None),
            blur_level=row.get("blur_level", None),
            luminosity=row.get("luminosity_level", None),
            seam=row.get("welding-seams", None)
        )


        # Convertir en PIL pour PyTorch transforms
        image = Image.fromarray(image)
        image = self.transform(image)

        # Réorganisation des canaux
        if isinstance(image, torch.Tensor) and image.ndim == 3:
            if self.channels_first:
                if image.shape[0] != 3 and image.shape[-1] == 3:
                    image = image.permute(2,0,1)
            else:
                if image.shape[0] == 3:
                    image = image.permute(1,2,0)

        # Label transform
        label = row[self.label_col]
        if self.label_transform:
            label = self.label_transform(label)
        if not torch.is_tensor(label):
            label = torch.tensor(1 if label=="KO" else 0, dtype=torch.long)

        return image, label

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
        is_train: bool = True,   # <---- NOUVEL ARGUMENT
    ):
        """
        df : DataFrame contenant au moins `path` et `label`.
        is_train : Si True -> ajoute les augmentations.
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.path_col = path_col
        self.label_col = label_col
        self.label_transform = label_transform
        self.channels_first = channels_first
        self.is_train = is_train

        # Choix automatique du transform si aucun fourni
        if transform is None:
            self.transform = train_augmentations() if is_train else test_transform()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = self.root_dir / row[self.path_col]
        label = row[self.label_col]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Réorganisation des canaux si nécessaire
        if isinstance(image, torch.Tensor) and image.ndim == 3:
            if self.channels_first:
                if image.shape[0] != 3 and image.shape[-1] == 3:
                    image = image.permute(2, 0, 1)
            else:
                if image.shape[0] == 3:
                    image = image.permute(1, 2, 0)

        # Label transform
        if self.label_transform is not None:
            label = self.label_transform(label)

        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        return image, label
'''
