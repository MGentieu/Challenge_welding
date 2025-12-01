"""
Module d'augmentation de données pour le projet Trustworthy AI
Auteur: Sarah/Najlaa
Partie 3 : Augmentation - Équilibrage et zones rares
"""

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from typing import Tuple, List, Dict, Callable
import random
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


# APRÈS
class WeldingDataAugmentor:
    """Pipeline d'augmentation AGRESSIVE pour équilibrage dataset"""
    """
    Stratégies:
        1. Augmentation de base (toutes images)
        2. Augmentation agressive (classes minoritaires - KO)
        3. Augmentation zones rares (blur élevé, luminosité extrême)
    """
    
    def __init__(self, mode='balanced'):
        """
        Args:
            mode: 'balanced' (équilibrer OK/KO), 
                  'rare_zones' (cibler zones rares), 
                  'aggressive' (toujours agressif)
        """
        self.mode = mode
        self.base_transform = self._get_base_transform()
        self.aggressive_transform = self._get_aggressive_transform()
        self.rare_zone_transform = self._get_rare_zone_transform()
        
    def _get_base_transform(self):
        """Augmentation de base (toutes images)"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=10, 
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
        ])
    

    def _get_aggressive_transform(self):
        """Augmentation agressive RÉALISTE (pour zones rares et KO)"""
        return A.Compose([
        # Légères translations
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.05,
            rotate_limit=0,        # ❌ PAS de rotation !
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.4
        ),
        
        # FLOU (focus caméra variable) - PRINCIPAL FACTEUR
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.6),  # 60% de chances d'avoir du flou
        
        # LUMINOSITÉ/CONTRASTE (éclairage industriel variable) - 2e FACTEUR
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.25,   # Variations plus fortes
                contrast_limit=0.25,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),  # Égalisation d'histogramme
        ], p=0.7),  # 70% de chances de variation lumineuse
        
        # BRUIT (capteur caméra)
        A.GaussNoise(var_limit=(10, 50), p=0.3),
    ])
    
    def _get_rare_zone_transform(self):
   
        return A.Compose([
        # FLOU TRÈS FORT (blur_level > 3000)
        A.OneOf([
            A.GaussianBlur(blur_limit=(9, 17), p=1.0),   # Très flou
            A.MotionBlur(blur_limit=15, p=1.0),
        ], p=0.8),  # 80% de chances
        
        # LUMINOSITÉ EXTRÊME (< 25 ou > 55)
        A.OneOf([
            # Très sombre
            A.RandomBrightnessContrast(
                brightness_limit=(-0.35, -0.15),
                contrast_limit=0,
                p=1.0
            ),
            # Très clair
            A.RandomBrightnessContrast(
                brightness_limit=(0.15, 0.35),
                contrast_limit=0,
                p=1.0
            ),
        ], p=0.8),  # 80% de chances
        
        # Bruit fort
        A.GaussNoise(var_limit=(30, 70), p=0.5),
    ])
    
    
    def augment_image(self, image: np.ndarray, class_label: str = None, 
                     blur_level: float = None, luminosity: float = None, 
                     seam: str = None) -> np.ndarray:
        """
        Applique l'augmentation appropriée selon le contexte
        
        Args:
            image: Image numpy (H, W, C) en uint8
            class_label: 'OK' ou 'KO'
            blur_level: Niveau de flou original
            luminosity: Niveau de luminosité original
            seam: Nom de la seam (c20, c3B, c102)
        
        Returns:
            Image augmentée (H, W, C) en uint8
        """
        # Toujours appliquer augmentation de base
        result = self.base_transform(image=image)
        image_aug = result['image']
        
        # Conditions pour augmentation agressive
        is_ko = (class_label == 'KO') if class_label else False
        is_rare_seam = (seam == 'c102') if seam else False
        is_rare_zone = False
        
        if blur_level is not None and luminosity is not None:
            is_rare_zone = (blur_level > 3000) or (luminosity < 25) or (luminosity > 55)
        
        if self.mode == 'balanced' and is_ko:
            # Sur-échantillonner les KO avec augmentation agressive
            result = self.aggressive_transform(image=image_aug)
            image_aug = result['image']
        
        elif self.mode == 'rare_zones' and (is_rare_zone or is_rare_seam):
            # Cibler les zones sous-représentées
            result = self.rare_zone_transform(image=image_aug)
            image_aug = result['image']
        
        elif self.mode == 'aggressive':
            # Toujours agressif
            result = self.aggressive_transform(image=image_aug)
            image_aug = result['image']
        
        return image_aug
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Pour compatibilité avec torchvision transforms"""
        return self.augment_image(image)


class BalancedAugmentationDataset(Dataset):
    """
    Dataset PyTorch avec augmentation automatique pour équilibrer les classes
    """
    
    def __init__(self, metadata_df: pd.DataFrame, image_loader_func: Callable,
                 balance_classes: bool = True, augment_rare_zones: bool = True,
                 transform=None):
        """
        Args:
            metadata_df: DataFrame avec colonnes ['sample_id', 'class', 'blur_level', 
                                                    'luminosity_level', 'welding-seams']
            image_loader_func: Fonction qui prend un sample_id et retourne l'image (np.ndarray)
            balance_classes: Si True, sur-échantillonne KO pour équilibrer avec OK
            augment_rare_zones: Si True, sur-échantillonne les zones rares
            transform: Transformations PyTorch finales (ToTensor, Normalize, etc.)
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_loader_func = image_loader_func
        self.balance_classes = balance_classes
        self.augment_rare_zones = augment_rare_zones
        self.transform = transform
        
        # Créer l'augmenteur
        self.augmentor = WeldingAugmentation(mode='balanced')
        
        # Préparer les indices avec sur-échantillonnage
        self.indices = self._prepare_balanced_indices()
        
        print(f"Dataset créé : {len(self.indices)} samples")
    
    def _prepare_balanced_indices(self):
        """Prépare les indices avec sur-échantillonnage des minoritaires"""
        indices = []
        
        # Compter les classes
        class_counts = self.metadata_df['class'].value_counts()
        n_ok = class_counts.get('OK', 0)
        n_ko = class_counts.get('KO', 0)
        
        print(f"Distribution originale : OK={n_ok}, KO={n_ko}")
        
        # Calculer le facteur d'augmentation pour KO
        if self.balance_classes and n_ko > 0:
            augmentation_factor = max(1, n_ok // n_ko)
        else:
            augmentation_factor = 1
        
        # Ajouter tous les indices
        for idx in range(len(self.metadata_df)):
            row = self.metadata_df.iloc[idx]
            
            # Ajouter l'échantillon original
            indices.append((idx, False))  # (index, is_augmented)
            
            # Sur-échantillonner KO
            if self.balance_classes and row['class'] == 'KO':
                for _ in range(augmentation_factor - 1):
                    indices.append((idx, True))
            
            # Sur-échantillonner zones rares
            if self.augment_rare_zones:
                is_rare = (
                    (row['blur_level'] > 3000) or 
                    (row['luminosity_level'] < 25) or 
                    (row['luminosity_level'] > 55) or
                    (row['welding-seams'] == 'c102')
                )
                
                if is_rare:
                    for _ in range(2):  # Dupliquer 2 fois
                        indices.append((idx, True))
        
        n_ko_final = n_ko * augmentation_factor if self.balance_classes else n_ko
        print(f"Distribution finale : OK={n_ok}, KO≈{n_ko_final}")
        
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor (C, H, W)
            label: int (0=OK, 1=KO)
        """
        # Récupérer l'indice réel et le flag d'augmentation
        real_idx, should_augment = self.indices[idx]
        row = self.metadata_df.iloc[real_idx]
        
        # Charger l'image
        image = self.image_loader_func(row['sample_id'])
        
        # Convertir en uint8 si nécessaire
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Appliquer augmentation si nécessaire
        if should_augment:
            image = self.augmentor.augment_image(
                image,
                class_label=row['class'],
                blur_level=row['blur_level'],
                luminosity=row['luminosity_level'],
                seam=row['welding-seams']
            )
        
        # Convertir en PIL pour transforms PyTorch
        image = Image.fromarray(image)
        
        # Appliquer transformations finales (ToTensor, Normalize, etc.)
        if self.transform:
            image = self.transform(image)
        
        # Label
        label = 1 if row['class'] == 'KO' else 0
        
        return image, label


class MixupCutmix:
    """
    Techniques avancées d'augmentation : Mixup et CutMix
    
    Note: Ces techniques sont optionnelles et peuvent améliorer la régularisation
    """
    
    @staticmethod
    def mixup(image1: np.ndarray, image2: np.ndarray, 
              label1: int, label2: int, alpha: float = 0.2) -> Tuple[np.ndarray, float]:
        """
        Mixup de deux images
        
        Args:
            image1, image2: Images (H, W, C)
            label1, label2: Labels (0 ou 1)
            alpha: Paramètre de la distribution Beta
        
        Returns:
            mixed_image: Image mixée
            mixed_label: Label mixé
        """
        lam = np.random.beta(alpha, alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_image.astype(np.uint8), mixed_label
    
    @staticmethod
    def cutmix(image1: np.ndarray, image2: np.ndarray, 
               label1: int, label2: int) -> Tuple[np.ndarray, float]:
        """
        CutMix de deux images
        
        Args:
            image1, image2: Images (H, W, C)
            label1, label2: Labels (0 ou 1)
        
        Returns:
            mixed_image: Image avec région coupée-collée
            mixed_label: Label mixé selon le ratio de surface
        """
        h, w = image1.shape[:2]
        
        # Choisir une région aléatoire
        cut_ratio = np.random.uniform(0.2, 0.5)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Mixer
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        
        # Calculer le ratio de label
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_image, mixed_label


# ========================================================================
# FONCTIONS UTILITAIRES POUR INTÉGRATION FACILE
# ========================================================================

def create_augmented_dataset(metadata_path: str, 
                            image_dir: str, 
                            output_dir: str, 
                            mode: str = 'balanced',
                            save_images: bool = True):
    """
    Crée un dataset augmenté complet et le sauvegarde sur disque
    
    Args:
        metadata_path: Chemin vers metadata_train.csv
        image_dir: Dossier contenant les images
        output_dir: Dossier de sortie
        mode: 'balanced', 'rare_zones', ou 'aggressive'
        save_images: Si True, sauvegarde les images sur disque
    
    Returns:
        augmented_metadata: DataFrame avec les nouvelles métadonnées
    """
    import os
    
    # Charger métadonnées
    metadata = pd.read_csv(metadata_path)
    
    # Créer l'augmenteur
    augmentor = WeldingAugmentation(mode=mode)
    
    # Fonction de chargement d'image
    def load_image(sample_id):
        # Chercher l'image dans le répertoire
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = Path(image_dir) / f"{sample_id}{ext}"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        raise FileNotFoundError(f"Image not found for sample_id: {sample_id}")
    
    # Calculer combien d'augmentations faire
    class_counts = metadata['class'].value_counts()
    n_ok = class_counts.get('OK', 0)
    n_ko = class_counts.get('KO', 0)
    augmentation_factor = max(1, n_ok // n_ko) if n_ko > 0 else 1
    
    print(f"Distribution initiale : OK={n_ok}, KO={n_ko}")
    print(f"Facteur d'augmentation pour KO : {augmentation_factor}")
    
    # Préparer les données augmentées
    augmented_data = []
    
    # Copier toutes les images originales
    for _, row in metadata.iterrows():
        augmented_data.append(row.to_dict())
    
    # Augmenter les KO
    ko_samples = metadata[metadata['class'] == 'KO']
    
    for _, row in tqdm(ko_samples.iterrows(), total=len(ko_samples), desc="Augmenting KO samples"):
        image = load_image(row['sample_id'])
        
        for aug_idx in range(augmentation_factor - 1):
            # Augmenter l'image
            aug_image = augmentor.augment_image(
                image,
                class_label=row['class'],
                blur_level=row['blur_level'],
                luminosity_level=row['luminosity_level'],
                seam=row['welding-seams']
            )
            
            # Créer nouveau sample_id
            new_sample_id = f"{row['sample_id']}_aug_{aug_idx}"
            
            # Sauvegarder l'image si demandé
            if save_images:
                os.makedirs(output_dir, exist_ok=True)
                save_path = Path(output_dir) / f"{new_sample_id}.jpg"
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), aug_image_bgr)
            
            # Créer nouvelle entrée de métadonnées
            new_row = row.to_dict()
            new_row['sample_id'] = new_sample_id
            new_row['path'] = str(Path(output_dir) / f"{new_sample_id}.jpg")
            augmented_data.append(new_row)
    
    # Augmenter les zones rares
    rare_samples = metadata[
        (metadata['blur_level'] > 3000) |
        (metadata['luminosity_level'] < 25) |
        (metadata['luminosity_level'] > 55) |
        (metadata['welding-seams'] == 'c102')
    ]
    
    print(f"Samples rares identifiés : {len(rare_samples)}")
    
    for _, row in tqdm(rare_samples.iterrows(), total=len(rare_samples), desc="Augmenting rare zones"):
        image = load_image(row['sample_id'])
        
        for aug_idx in range(2):  # 2 copies pour zones rares
            aug_image = augmentor.augment_image(
                image,
                class_label=row['class'],
                blur_level=row['blur_level'],
                luminosity_level=row['luminosity_level'],
                seam=row['welding-seams']
            )
            
            new_sample_id = f"{row['sample_id']}_rare_aug_{aug_idx}"
            
            if save_images:
                os.makedirs(output_dir, exist_ok=True)
                save_path = Path(output_dir) / f"{new_sample_id}.jpg"
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), aug_image_bgr)
            
            new_row = row.to_dict()
            new_row['sample_id'] = new_sample_id
            new_row['path'] = str(Path(output_dir) / f"{new_sample_id}.jpg")
            augmented_data.append(new_row)
    
    # Créer DataFrame final
    augmented_metadata = pd.DataFrame(augmented_data)
    
    # Sauvegarder métadonnées
    if save_images:
        metadata_out_path = Path(output_dir) / 'metadata_augmented.csv'
        augmented_metadata.to_csv(metadata_out_path, index=False)
        print(f"Métadonnées sauvegardées : {metadata_out_path}")
    
    print(f"\nDataset augmenté créé :")
    print(f"   Total samples : {len(augmented_metadata)}")
    print(f"   Distribution finale : {augmented_metadata['class'].value_counts().to_dict()}")
    
    return augmented_metadata


def get_augmentation_transforms(is_training: bool = True, img_size: int = 224):
    """
    Retourne les transformations PyTorch appropriées
    
    Args:
        is_training: Si True, retourne transforms avec augmentation
        img_size: Taille de l'image
    
    Returns:
        transforms: Composition de transformations torchvision
    """
    from torchvision import transforms
    
    if is_training:
        # Training : augmentation + normalisation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # Validation/Test : seulement normalisation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# ========================================================================
# EXEMPLE D'UTILISATION
# ========================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation du module d'augmentation
    """
    
    # Exemple 1 : Créer un dataset augmenté sur disque
    print("="*60)
    print("EXEMPLE 1 : Création dataset augmenté")
    print("="*60)
    
    # augmented_df = create_augmented_dataset(
    #     metadata_path='./data/metadata_train.csv',
    #     image_dir='./data/images',
    #     output_dir='./data/augmented_dataset',
    #     mode='balanced',
    #     save_images=True
    # )
    
    # Exemple 2 : Utiliser avec PyTorch DataLoader
    print("\n" + "="*60)
    print("EXEMPLE 2 : Utilisation avec PyTorch")
    print("="*60)
    
    # Charger métadonnées
    # metadata_df = pd.read_csv('./data/metadata_train.csv')
    
    # Fonction de chargement d'image
    # def load_image(sample_id):
    #     img_path = f'./data/images/{sample_id}.jpg'
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     return img
    
    # Créer le dataset
    # train_dataset = BalancedAugmentationDataset(
    #     metadata_df=metadata_df,
    #     image_loader_func=load_image,
    #     balance_classes=True,
    #     augment_rare_zones=True,
    #     transform=get_augmentation_transforms(is_training=True)
    # )
    
    # Créer le DataLoader
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # print(f"\n✅ DataLoader créé : {len(train_dataset)} samples")
    # print(f"   Nombre de batches : {len(train_loader)}")
    
    print("\n✅ Module d'augmentation prêt à l'emploi !")