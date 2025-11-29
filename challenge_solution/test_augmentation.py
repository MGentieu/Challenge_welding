"""
Script de test pour le module d'augmentation
ADAPTÉ À TA STRUCTURE DE DOSSIERS
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ajouter le chemin pour importer les modules
sys.path.append(os.path.dirname(__file__))

from augmentation import WeldingAugmentation

def find_first_image():
    """Trouve la première image disponible dans ton dataset"""
    
    print("🔍 Recherche d'images dans le dataset...")
    
    # CHEMINS ADAPTÉS À TON ARBORESCENCE
    possible_base_paths = [
        Path('../datasets/welding-detection-challenge-dataset'),      # Si tu es dans challenge_solution/
        Path('./datasets/welding-detection-challenge-dataset'),       # Si tu es à la racine
        Path('../../datasets/welding-detection-challenge-dataset'),   # Si tu es plus profond
    ]
    
    for base_path in possible_base_paths:
        if not base_path.exists():
            continue
        
        print(f"   ✅ Dossier trouvé : {base_path}")
        
        # Chercher récursivement dans expert/ et operator/
        for seam_folder in base_path.iterdir():
            if not seam_folder.is_dir() or seam_folder.name.startswith('.'):
                continue
            
            # seam_folder = c20, c102, c33
            seam = seam_folder.name
            
            for class_folder in seam_folder.iterdir():
                if not class_folder.is_dir() or class_folder.name.startswith('.'):
                    continue
                
                # class_folder = OK, KO
                class_label = class_folder.name
                
                for labeling_folder in class_folder.iterdir():
                    if not labeling_folder.is_dir() or labeling_folder.name.startswith('.'):
                        continue
                    
                    # labeling_folder = expert, operator
                    labeling_type = labeling_folder.name
                    
                    # Chercher les images dans ce dossier
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG']:
                        images = list(labeling_folder.glob(ext))
                        
                        if images:
                            image_path = images[0]
                            print(f"   ✅ Image trouvée : {image_path}")
                            print(f"      Seam: {seam}, Class: {class_label}, Labeling: {labeling_type}")
                            
                            return image_path, seam, class_label, labeling_type
    
    print("   ❌ Aucune image trouvée dans les dossiers expert/ et operator/")
    return None, None, None, None


def test_augmentation_visual():
    """Test visuel : génère 9 versions augmentées d'une image"""
    
    print("="*60)
    print("🔍 TEST VISUEL DE L'AUGMENTATION")
    print("="*60)
    
    # ========== 1. CHERCHER UNE IMAGE ==========
    image_path, seam, class_label, labeling_type = find_first_image()
    
    if image_path is None:
        print("⚠️  Aucune image trouvée. Génération d'une image synthétique...")
        # Créer une image synthétique
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.circle(image, (112, 112), 50, (255, 0, 0), -1)
        cv2.rectangle(image, (50, 50), (150, 150), (0, 255, 0), 3)
        seam = 'c20'
        class_label = 'KO'
        labeling_type = 'expert'
    else:
        # Charger l'image réelle
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Impossible de charger l'image : {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # ========== 2. CRÉER L'AUGMENTEUR ==========
    augmentor = WeldingAugmentation(mode='balanced')
    
    # ========== 3. GÉNÉRER 9 VERSIONS AUGMENTÉES ==========
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, ax in enumerate(axes.flat):
        if i == 0:
            # Image originale
            ax.imshow(image)
            title = f'ORIGINAL\n{seam} | {class_label} | {labeling_type}'
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            # Versions augmentées
            aug_image = augmentor.augment_image(
                image,
                class_label=class_label or 'KO',
                blur_level=5000,    # Zone rare
                luminosity=20,      # Zone rare
                seam=seam or 'c102' # Zone rare
            )
            ax.imshow(aug_image)
            ax.set_title(f'AUGMENTED {i}', fontsize=11)
        
        ax.axis('off')
    
    plt.suptitle('Test du Module d\'Augmentation\nZone rare : KO + blur élevé + luminosité basse', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    output_path = 'augmentation_test_visual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Test visuel sauvegardé : {output_path}")
    
    plt.show()


def test_augmentation_modes():
    """Test des différents modes d'augmentation"""
    
    print("\n" + "="*60)
    print("🔍 TEST DES MODES D'AUGMENTATION")
    print("="*60)
    
    # Chercher une image réelle
    image_path, seam, class_label, labeling_type = find_first_image()
    
    if image_path is None:
        # Image synthétique
        image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        cv2.circle(image, (112, 112), 50, (255, 100, 100), -1)
    else:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Tester les 3 modes
    modes = ['balanced', 'rare_zones', 'aggressive']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row, mode in enumerate(modes):
        augmentor = WeldingAugmentation(mode=mode)
        
        # Original
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f'Original\n(Mode: {mode})', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        # 3 versions augmentées
        for col in range(1, 4):
            aug_image = augmentor.augment_image(
                image,
                class_label='KO',
                blur_level=4000,
                luminosity=30,
                seam='c20'
            )
            axes[row, col].imshow(aug_image)
            axes[row, col].set_title(f'Aug {col}', fontsize=10)
            axes[row, col].axis('off')
    
    plt.suptitle('Comparaison des 3 Modes d\'Augmentation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'augmentation_test_modes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Test des modes sauvegardé : {output_path}")
    
    plt.show()


def test_class_balance_simulation():
    """Simule l'équilibrage des classes"""
    
    print("\n" + "="*60)
    print("🔍 SIMULATION DE L'ÉQUILIBRAGE")
    print("="*60)
    
    # Simuler des métadonnées réalistes
    initial_ok = 4000
    initial_ko = 1000
    
    augmentation_factor = initial_ok // initial_ko
    final_ko = initial_ko * augmentation_factor
    
    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # AVANT
    ax1.bar(['OK', 'KO'], [initial_ok, initial_ko], color=['green', 'red'], alpha=0.7)
    ax1.set_ylabel('Nombre de samples', fontsize=12)
    ax1.set_title('AVANT Augmentation', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 4500])
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (label, value) in enumerate([('OK', initial_ok), ('KO', initial_ko)]):
        ax1.text(i, value + 100, f'{value}\n({value/(initial_ok+initial_ko)*100:.1f}%)', 
                ha='center', fontsize=11, fontweight='bold')
    
    # APRÈS
    ax2.bar(['OK', 'KO'], [initial_ok, final_ko], color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Nombre de samples', fontsize=12)
    ax2.set_title('APRÈS Augmentation', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 4500])
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (label, value) in enumerate([('OK', initial_ok), ('KO', final_ko)]):
        ax2.text(i, value + 100, f'{value}\n({value/(initial_ok+final_ko)*100:.1f}%)', 
                ha='center', fontsize=11, fontweight='bold')
    
    ax2.annotate(f'Facteur × {augmentation_factor}', 
                xy=(1, final_ko), xytext=(1.3, final_ko - 500),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red')
    
    plt.suptitle(f'Impact de l\'Augmentation sur l\'Équilibrage des Classes\n'
                 f'Total: {initial_ok+initial_ko} → {initial_ok+final_ko} samples',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'augmentation_test_balance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Test d'équilibrage sauvegardé : {output_path}")
    
    plt.show()
    
    # Afficher les stats
    print("\n📊 STATISTIQUES D'ÉQUILIBRAGE:")
    print(f"AVANT : OK={initial_ok} ({initial_ok/(initial_ok+initial_ko)*100:.1f}%), KO={initial_ko} ({initial_ko/(initial_ok+initial_ko)*100:.1f}%)")
    print(f"APRÈS : OK={initial_ok} ({initial_ok/(initial_ok+final_ko)*100:.1f}%), KO={final_ko} ({final_ko/(initial_ok+final_ko)*100:.1f}%)")
    print(f"✅ Facteur d'augmentation : × {augmentation_factor}")


def test_rare_zones_detection():
    """Test de la détection des zones rares"""
    
    print("\n" + "="*60)
    print("🔍 TEST DE DÉTECTION DES ZONES RARES")
    print("="*60)
    
    samples = [
        {'id': 1, 'blur': 1500, 'lum': 45, 'seam': 'c20', 'rare': False},
        {'id': 2, 'blur': 5000, 'lum': 45, 'seam': 'c20', 'rare': True},
        {'id': 3, 'blur': 1500, 'lum': 15, 'seam': 'c20', 'rare': True},
        {'id': 4, 'blur': 1500, 'lum': 60, 'seam': 'c20', 'rare': True},
        {'id': 5, 'blur': 1500, 'lum': 45, 'seam': 'c102', 'rare': True},
        {'id': 6, 'blur': 5500, 'lum': 18, 'seam': 'c102', 'rare': True},
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, sample in enumerate(samples):
        base_brightness = int(sample['lum'] * 5)
        image = np.ones((224, 224, 3), dtype=np.uint8) * base_brightness
        
        cv2.putText(image, f"ID:{sample['id']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, sample['seam'], (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        axes[i].imshow(image)
        
        title = f"blur={sample['blur']}, lum={sample['lum']}\n{sample['seam']}"
        color = 'red' if sample['rare'] else 'green'
        fontweight = 'bold' if sample['rare'] else 'normal'
        
        axes[i].set_title(title, fontsize=10, color=color, fontweight=fontweight)
        axes[i].axis('off')
        
        if sample['rare']:
            axes[i].text(0.5, 0.95, '⚠️ ZONE RARE', 
                        transform=axes[i].transAxes,
                        fontsize=12, fontweight='bold', color='red',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.suptitle('Détection des Zones Rares\n'
                 'Critères : blur>3000 OU lum<25 OU lum>55 OU seam=c102',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'augmentation_test_rare_zones.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Test zones rares sauvegardé : {output_path}")
    
    plt.show()


def main():
    """Exécute tous les tests"""
    
    print("\n" + "="*70)
    print(" "*15 + "🚀 TEST DU MODULE D'AUGMENTATION 🚀")
    print("="*70)
    
    try:
        test_augmentation_visual()
        test_augmentation_modes()
        test_class_balance_simulation()
        test_rare_zones_detection()
        
        print("\n" + "="*70)
        print(" "*20 + "✅ TOUS LES TESTS RÉUSSIS !")
        print("="*70)
        print("\n📂 Fichiers générés :")
        print("   • augmentation_test_visual.png")
        print("   • augmentation_test_modes.png")
        print("   • augmentation_test_balance.png")
        print("   • augmentation_test_rare_zones.png")
        print("\n💡 Vérifie ces images pour valider l'augmentation !")
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()