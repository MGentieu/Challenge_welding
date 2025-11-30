import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


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
