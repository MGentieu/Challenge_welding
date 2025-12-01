import os
import random
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# ✅ CORRECTION : Import avec le bon nom de classe
from augmentation import WeldingDataAugmentor, BalancedAugmentationDataset

    
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
        if self.transform is not None:
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
    

# ========== NOUVEAU : Dataset avec Augmentation Agressive ==========

class WeldingDataset(Dataset):
    """
    Dataset pour soudures avec :
    - Augmentation agressive (équilibrage OK/KO)
    - Corruption de données (Martin)
    - Zones rares ciblées
    """
    
    def __init__(
        self, 
        metadata_path, 
        image_dir, 
        corruption_rate=0.15, 
        balance_classes=True,
        augmentation_mode='balanced'
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir = Path(image_dir)
        self.corruption_rate = corruption_rate
        
        # ✅ Utiliser WeldingDataAugmentor (ton module)
        self.augmentor = WeldingDataAugmentor(mode=augmentation_mode)
        
        # ✅ Préparer indices équilibrés
        if balance_classes:
            self.indices = self._prepare_balanced_indices()
        else:
            self.indices = [(i, False) for i in range(len(self.metadata))]
        
        # Transform final (ToTensor + Normalize)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _prepare_balanced_indices(self):
        """Équilibre OK/KO via sur-échantillonnage"""
        indices = []
        class_counts = self.metadata['class'].value_counts()
        n_ok = class_counts.get('OK', 0)
        n_ko = class_counts.get('KO', 0)
        
        # Facteur d'augmentation pour KO
        augmentation_factor = max(1, n_ok // n_ko) if n_ko > 0 else 1
        
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            indices.append((idx, False))  # Image originale
            
            # Sur-échantillonner KO
            if row['class'] == 'KO':
                for _ in range(augmentation_factor - 1):
                    indices.append((idx, True))  # Version augmentée
        
        return indices
    
    def _load_image(self, img_path):
        """Charge une image depuis le disque"""
        full_path = self.image_dir / img_path
        image = Image.open(full_path).convert("RGB")
        return np.array(image)
    
    def _apply_corruption(self, image):
        """Applique corruption de Martin (15% du temps)"""
        # Exemples de corruption (à adapter selon le code de Martin)
        corruption_type = random.choice(['blur', 'noise', 'brightness'])
        
        if corruption_type == 'blur':
            from scipy.ndimage import gaussian_filter
            image = gaussian_filter(image, sigma=2)
        elif corruption_type == 'noise':
            noise = np.random.normal(0, 25, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
        elif corruption_type == 'brightness':
            factor = random.uniform(0.5, 1.5)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx, should_augment = self.indices[idx]
        row = self.metadata.iloc[real_idx]
        
        # 1. Charger image
        image = self._load_image(row['path'])
        
        # 2. Appliquer augmentation agressive si nécessaire
        if should_augment:
            image = self.augmentor.augment_image(
                image,
                class_label=row['class'],
                blur_level=row.get('blur_level', 1000),
                luminosity=row.get('luminosity_level', 40),
                seam=row.get('welding-seams', 'c20')
            )
        
        # 3. Appliquer corruption de Martin (15% du temps)
        if random.random() < self.corruption_rate:
            image = self._apply_corruption(image)
        
        # 4. Transform final (ToTensor + Normalize)
        image = Image.fromarray(image.astype(np.uint8))
        image = self.transform(image)
        
        # 5. Label
        label = 1 if row['class'] == 'KO' else 0
        
        return image, label


# ========== FONCTION HELPER POUR CRÉER DATALOADERS ==========

def create_dataloaders(
    train_metadata_path,
    val_metadata_path,
    image_dir,
    batch_size=32,
    balance_classes=True,
    augmentation_mode='balanced',
    corruption_rate=0.15,
    num_workers=4
):
    """
    Crée train et val dataloaders avec augmentation
    
    Args:
        train_metadata_path: Chemin vers CSV train
        val_metadata_path: Chemin vers CSV validation
        image_dir: Dossier racine des images
        batch_size: Taille du batch
        balance_classes: Équilibrer OK/KO
        augmentation_mode: 'balanced', 'rare_zones', ou 'aggressive'
        corruption_rate: Taux de corruption (Martin)
        num_workers: Nombre de workers PyTorch
    
    Returns:
        train_loader, val_loader
    """
    
    # Train dataset avec augmentation
    train_dataset = WeldingDataset(
        metadata_path=train_metadata_path,
        image_dir=image_dir,
        corruption_rate=corruption_rate,
        balance_classes=balance_classes,
        augmentation_mode=augmentation_mode
    )
    
    # Validation dataset SANS augmentation
    val_df = pd.read_csv(val_metadata_path)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_dataset = ImageDataFrameDataset(
        df=val_df,
        root_dir=image_dir,
        path_col='path',
        label_col='label',
        transform=val_transform
    )
    
    # Créer dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader