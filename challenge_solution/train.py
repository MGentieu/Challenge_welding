'''
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

# Exemple de transform basique
transform = transforms.Compose([transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


df_data = dm.explore_csv_hierarchy('../../notebooks_cache',depth_name_list=['folder_1','folder_2','folder_3','seam','decision','type_label'],allowed_ext='.jpeg')
mapping = {'OK': 0, 'KO': 1}
df_data['label'] = df_data['decision'].map(mapping)
df_train,df_val = dm.stratified_train_val_split(df_data, ['seam','decision'], alpha=0.95, random_state=42)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🔧 Using CUDA")
else:
    device = torch.device('cpu')
    print(f"🔧 Using CPU")

Train_Dataset = ImageDataFrameDataset(df=df_train,root_dir="../Challenge-Welding-Reference-Solution-1/",path_col="path",label_col="label",transform=transform,channels_first=True)
Val_Dataset = ImageDataFrameDataset(df=df_train,root_dir="../Challenge-Welding-Reference-Solution-1/",path_col="path",label_col="label",transform=transform,channels_first=True)

ai_component = MyAIComponent()
ai_component.init_model()
ai_component.load_model()
ai_component.train_model(Train_Dataset,
                         Val_Dataset,
                         device=device,
                         save_path="best_model.pth",
                         augmentation_fn=None,
                         preprocess_fn=None,
                         epochs=50,
                         batch_size=128,
                         lr=3e-4)
'''

import numpy as np
import cv2
import time
import os
import sys
from pathlib import Path

# Add the challenge_solution to path
#sys.path.insert(0, 'challenge_solution')

from torchvision import transforms
from torchvision.transforms import InterpolationMode
sys.path.insert(0, '/home/kevin.pasini/projet_explo/kevin/uqmodels/abench/')
import df_utils as dm
from torch_dataloader import ImageDataFrameDataset
from AIComponent import MyAIComponent


# Exemple de transform basique Redimensionne et normalise
transform = transforms.Compose([transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#Genere un meta dataframe utilisé pour acceder au données.
dossier_des_images = "./datasets/welding-detection-challenge-dataset/"
df_data = dm.explore_csv_hierarchy(dossier_des_images,depth_name_list=['seam','decision','type_label'],allowed_ext='.jpeg')
df_data['path'] = df_data['path'].apply(lambda p: os.path.relpath(p, dossier_des_images))

# Mapping des labels
mapping = {'OK': 0, 'KO': 1}
df_data['label'] = df_data['decision'].map(mapping)

print("Types de décisions :", df_data['decision'].unique())
print("Valeurs manquantes :", df_data['label'].isna().sum())
print("Types de soudure :", df_data['type_label'].unique())

welding_types = ["c20", "c33", "c102"]

for wt in welding_types:
    print(f"\n=== Entraînement modèle pour {wt} ===")
    
    # Filtrer le dataset sur le type de soudure (seam)
    df_subset = df_data[df_data['seam'] == wt]
    
    if df_subset.empty:
        print(f"⚠️  Aucun échantillon trouvé pour {wt}, vérifie tes chemins et extensions.")
        continue

    # Stratified split
    df_train, df_val = dm.stratified_train_val_split(df_subset, ['seam','decision'], alpha=0.95, random_state=42)
    
    # Créer les datasets
    Train_Dataset = ImageDataFrameDataset(
        df=df_train,
        root_dir="./datasets/welding-detection-challenge-dataset/",  # chemin racine
        path_col="path",
        label_col="label",
        transform=transform,
        channels_first=True
    )
    
    Val_Dataset = ImageDataFrameDataset(
        df=df_val,
        root_dir="./datasets/welding-detection-challenge-dataset/",
        path_col="path",
        label_col="label",
        transform=transform,
        channels_first=True
    )

    # Initialiser et entraîner
    ai_component = MyAIComponent()
    ai_component.init_model()
    ai_component.train_model(
        Train_Dataset,
        Val_Dataset,
        device='cpu',  # ou 'cuda' si disponible
        save_path=f"best_model_{wt}.pth",
        augmentation_fn=None,
        preprocess_fn=None,
        epochs=1,
        batch_size=64,
        lr=3e-4
    )


print("\n🎉 Entraînement des trois modèles terminé !")
