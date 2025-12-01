import numpy as np
import torch
from pathlib import Path

import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from AIComponent import MyAIComponent
import df_utils as dm
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch_dataloader import ImageDataFrameDataset

# ========== AJOUT : Import augmentation ==========
from augmentation import WeldingDataAugmentor

# Transform basique
transform = transforms.Compose([
    transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ========== PROTECTION MULTIPROCESSING WINDOWS ==========
if __name__ == '__main__':
    
    # ========== MODIFICATION : Utiliser ton dataset au lieu de notebooks_cache ==========
    df_data = dm.explore_csv_hierarchy(
        '../datasets/welding-detection-challenge-dataset',  # ✅ TON DATASET
        depth_name_list=['seam', 'decision', 'type_label'],  # ✅ Adapté à ton arborescence
        allowed_ext='.jpeg'  # ✅ Tes images sont en .jpeg
    )
    
    mapping = {'OK': 0, 'KO': 1}
    df_data['label'] = df_data['decision'].map(mapping)
    
    # Split train/val stratifié
    df_train, df_val = dm.stratified_train_val_split(
        df_data, 
        ['seam', 'decision'], 
        alpha=0.95, 
        random_state=42
    )
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🔧 Using CUDA")
    else:
        device = torch.device('cpu')
        print(f"🔧 Using CPU")
    
    # ========== AJOUT : Créer augmenteur pour équilibrage OK/KO ==========
    augmentor = WeldingDataAugmentor(mode='balanced')
    
    def augmentation_fn(image):
        """
    Fonction d'augmentation appelée par train_model
    Applique augmentation agressive (sans métadonnées pour simplifier)
    """
    # Valeurs par défaut pour les métadonnées
        return augmentor.augment_image(
        image,
        class_label='KO',  # On applique augmentation agressive par défaut
        blur_level=1000,
        luminosity=40,
        seam='c20'
    )
    

    # Datasets
    Train_Dataset = ImageDataFrameDataset(
    df=df_train,
    root_dir="",  # ✅ Chemin vide - utilise directement le path du DataFrame
    path_col="path",
    label_col="label",
    transform=transform,
    channels_first=True
    )

    Val_Dataset = ImageDataFrameDataset(
    df=df_val,
    root_dir="",  # ✅ Chemin vide
    path_col="path",
    label_col="label",
    transform=transform,
    channels_first=True
    )
    # Entraînement
    ai_component = MyAIComponent()
    ai_component.init_model()
    ai_component.load_model()
    
    ai_component.train_model(
        Train_Dataset,
        Val_Dataset,
        device=device,
        save_path="best_model.pth",
        augmentation_fn=None,  
        preprocess_fn=None,
        epochs=50,
        batch_size=128,
        lr=3e-4
    )