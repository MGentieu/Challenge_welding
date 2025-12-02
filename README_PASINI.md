# Trustworthy AI for Welding Quality Detection

## Pour M. Pasini :

**Auteurs :**
- GENTIEU Martin
- ROUSSELET Philéas
- GUERMOUCH Kenza
- JEBBARI Fatima
- ROBERT Mathias
- ALLIOUI Najlaa
- SHANIN Sarah
**Date :** Décembre 2025  
**Projet :** Challenge de Détection de Qualité de Soudures  
**Composante :** Trustworthy AI (Partie 3)

---

## Table des Matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Problématique](#-problématique)
3. [Architecture](#-architecture)
4. [Module 1 : Augmentation](#-module-1--augmentation-de-données)
5. [Module 2 : Dataloader](#-module-2--dataloader)
6. [Module 2 : Uncertainty](#-module-2--quantification-dincertitude)
7. [Module 3 : OOD Detection](#-module-3--détection-out-of-distribution)
8. [Installation](#-installation)
10. [Références](#-références)

---

##  Vue d'ensemble

Ce projet implémente trois composants d'**Intelligence Artificielle Responsable** (Trustworthy AI) pour améliorer la robustesse et la fiabilité d'un système de détection automatique de défauts de soudure industrielle.

### Contexte

Dans un environnement industriel critique, un système de détection de défauts doit non seulement être précis, mais aussi :
- **Fiable** : Capable d'identifier ses propres limites
- **Robuste** : Performant sur des données rares ou extrêmes
- **Transparent** : Fournir une mesure de confiance pour chaque prédiction

### Objectifs du projet

1. **Améliorer les performances** sur la classe minoritaire (KO - défauts)
2. **Quantifier l'incertitude** des prédictions
3. **Détecter les anomalies** (images jamais vues pendant l'entraînement)

---

## 🚨 Problématique

### Dataset déséquilibré

Le dataset présente un déséquilibre important :
- **OK (bonnes soudures)** : 4000 images (80%)
- **KO (défauts)** : 1000 images (20%)

**Conséquence** : Le modèle apprend mal à détecter les défauts (classe critique).

### Zones rares sous-représentées

Certaines conditions sont très peu présentes :
- **Seam c102** : 3% du dataset (vs 48% pour c20)
- **Blur élevé** (>3000) : 15% du dataset
- **Luminosité extrême** (<25 ou >55) : 12% du dataset

**Conséquence** : Mauvaise performance dans ces conditions critiques.

### Absence de garanties

Le modèle baseline ne fournit aucune information sur :
- Sa **confiance** dans chaque prédiction
- La **détection d'images anormales** (hors distribution d'entraînement)

**Conséquence** : Impossible de savoir quand le modèle est fiable.

---

## 🏗️ Architecture

### Structure globale

```
┌─────────────┐
│   train.py  │  Script d'entraînement principal
└──────┬──────┘
       │
       ├──> AIComponent.py (Modèle EfficientNet-B0)
       │      ├──> uncertainty.py (Module 2)
       │      └──> ood_detection.py (Module 3)
       │
       ├──> torch_dataloader.py
       │      └──> augmentation.py (Module 1)
       │
       └──> df_utils.py (Utilitaires)
```

##  Module 1 : Augmentation de données

**Fichier** : `augmentation.py`  
**Classe** : `WeldingDataAugmentor`  
**Objectif** : Créer des variations réalistes d'images pour équilibrer le dataset et améliorer la robustesse.

### Exploration des données

Une exploration initiale des données :

![Images montrant les répartitions des données](stats_donnes.jpg)

Cette exploration révèle une répartition inéquitable et biaisée des données, ce qui nécessite de prendre cela en compte lors du chargement des données.

Pour y remédier, le module sur **l'augementation du dataset** a été mis en place.

### Analyse du contexte industriel

Contrairement aux techniques d'augmentation génériques (rotation, flip), nous avons adapté les transformations au contexte spécifique des soudures industrielles :

#### Ce que nous avons fait (et pourquoi)

**GaussianBlur (3-17px, p=0.7)** :  
→ Simule défocus caméra (focus mal réglé, vibrations)

**RandomBrightnessContrast (±25-35%, p=0.7)** :  
→ Simule variations d'éclairage industriel (lampes vieillissantes, reflets)

**GaussNoise (var 10-50, p=0.5)** :  
→ Simule bruit capteur (capteur vieillissant, interférences électriques)

**ShiftScaleRotate (shift ±5%, rotation 0°, p=0.3)** :  
→ Simule légères translations (décalage pièce, vibrations machine)

### Méthodologie

#### Librairie utilisée

**Albumentations**  :
- Transformations plus rapides (optimisées C++)
- Contrôle fin des probabilités par transformation
- Support natif numpy arrays
- Compositions complexes avec un seul pipeline

#### 3 Modes d'augmentation

**1. Mode `balanced` (équilibrage OK/KO)**


- **Objectif** : Équilibrer les classes
- **Méthode** : Sur-échantillonnage de la classe KO
- **Facteur** : n_OK / n_KO (typiquement ×4)
- **Transformations** : Modérées (p=0.5-0.7)
- **Résultat** : des classes ok/ko equilibrées (50/50)

**2. Mode `rare_zones` (zones rares)**

- **Objectif** : Sur-représenter conditions rares
- **Critères** :
  * blur_level > 3000
  * luminosity < 25 ou > 55
  * seam == 'c102'
- **Transformations** : Fortes (p=0.8)
- **Sur-échantillonnage** : ×3 pour ces images

**3. Mode `aggressive` (robustesse maximale)**


- **Objectif** : Robustesse extrême
- **Transformations** : Maximales (p=0.9)
- **Combinaisons** : Toutes transformations appliquées
- **Usage** : Tests de stress, validation robustesse

<img src="augmentation_test_modes.png" alt="Description" width="600"/>

## Module 2 : Dataloader :

Le module est représenté par le script **challenge_solution/torch_dataloader.py**
Le dataloader est la partie de pré-processing ayant pour but de charger le dataset et de transformer les données pour qu'elles puissent être utilisées dans des tenseurs pytorch normalisés.

Cela permet au réseau d'étudier les composantes principales, en normalisant les valeurs des tenseurs entre 0 et 1 pour les images.

Dans notre cas, une tentative a été faite pour intégrer les composantes du module **augmentation.py** afin de pouvoir générer automatiquement des données corrompues et enrichies.
Une option a également été mise en place pour générer soit un dataset d'entraînement (shuffled et augmenté) ou un dataset de validation (sans modification)

Cela permet lors de l'implémentation d'avoir uniquement à appelé ces deux fonctions pour obtenir des datasets prêts à l'emploi pour les modèles pytorch.

## Module 3 : Uncertainty

### Méthodologie

#### MC Dropout (Monte Carlo Dropout)

**Principe** : Effectuer T passages forward avec dropout activé pendant l'inférence.

**Interprétation** :
- **Faible variance** → Prédictions stables → Haute confiance
- **Forte variance** → Prédictions variables → Faible confiance

**Avantages** :
- Pas de ré-entraînement nécessaire
- Computationnellement abordable (30 forward passes ≈ 1 seconde CPU)
- Capture incertitude épistémique

#### Temperature Scaling

**Objectif** : Calibrer les probabilités pour qu'elles reflètent la vraie confiance.

**Problème** : Les réseaux de neurones sont sur-confiants :
```
Confidence 90% → Accuracy réelle 75%:  Sur-confiance
```

**Solution** : Appliquer une température T aux logits :
```python
calibrated_probs = softmax(logits / T)
```

**Optimisation** : Minimiser NLL (Negative Log-Likelihood) sur validation set :
```python
T_opt = argmin_T NLL(softmax(logits / T), true_labels)
```


## Module 4 : Détection Out-of-Distribution

**Fichier** : `ood_detection.py`  
**Classe** : `OODDetector`  
**Objectif** : Détecter les images anormales (hors distribution d'entraînement).

### Définition OOD

**In-Distribution (ID)** : Images similaires à celles d'entraînement  
**Out-of-Distribution (OOD)** : Images significativement différentes

**Exemples OOD dans notre contexte** :
- Nouveau type de soudure (seam c999)
- Conditions extrêmes (blur 20000, luminosité 0)
- Objet étranger dans le champ
- Défaut capteur (artefacts, saturation)

### Méthodologie

#### Approche multi-méthodes

Implémentation de **3 méthodes complémentaires** pour robustesse :

**1. Mahalanobis Distance**

**Principe** : Distance statistique dans l'espace des features.

**Interprétation** :
- Distance faible → Image similaire au training (ID)
- Distance élevée → Image anormale (OOD)

**Seuil** : 95e percentile des distances sur train set

**Avantages** :
- Théoriquement fondé (statistiques)
- Capture la structure de covariance

**2. Isolation Forest**

**Principe** : Détection d'outliers par isolation dans arbres de décision.

**Algorithme** :
```
Pour chaque point x:
    - Construire arbres aléatoires
    - Mesurer profondeur moyenne pour isoler x
    - Points anormaux → faible profondeur (faciles à isoler)
```

**Avantages** :
- Sans hypothèse sur distribution
- Efficace en haute dimension
- Capture outliers multivariés

**3. Energy-based Detection**

**Principe** : Images normales ont une énergie élevée (activations fortes).


**Interprétation** :
- Énergie élevée → Modèle confiant → ID
- Énergie faible → Modèle incertain → OOD

**Avantages** :
- Simple à implémenter
- Utilise directement les logits
- Pas besoin d'extraire features



### Structure des fichiers

```
challenge_solution/
├── AIComponent.py              # Modèle principal + intégration
├── augmentation.py             # Module 1 : Augmentation
├── uncertainty.py              # Module 2 : Uncertainty
├── ood_detection.py            # Module 3 : OOD Detection
├── torch_dataloader.py         # DataLoader avec augmentation
├── df_utils.py                 # Utilitaires données
├── train.py                    # Script entraînement
├── test_real_data.py           # Tests avec vraies données
└── README.Sarah.md             # Ce fichier
```

### Installation rapide

```bash
# Cloner repo
git clone <repo_url>
cd challenge_solution

# Installer dépendances
pip install -r requirements.txt


---

##  Utilisation

### Entraînement

```bash
python train.py
```

**Génère** :
- `best_model.pth` : Poids du modèle
- `train_features.npy` : Features pour OOD
- `train_logits.npy` : Logits pour OOD



##  Références

### Augmentation de données

1. **Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A.** (2020). *Albumentations: Fast and Flexible Image Augmentations*. Information, 11(2), 125. [https://doi.org/10.3390/info11020125](https://doi.org/10.3390/info11020125)

2. **Shorten, C., & Khoshgoftaar, T. M.** (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6(1), 1-48.

### Quantification d'incertitude

3. **Gal, Y., & Ghahramani, Z.** (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. In International Conference on Machine Learning (pp. 1050-1059). PMLR.

4. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017). *On Calibration of Modern Neural Networks*. In International Conference on Machine Learning (pp. 1321-1330). PMLR.

5. **Kendall, A., & Gal, Y.** (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*. Advances in Neural Information Processing Systems, 30.

### Détection Out-of-Distribution

6. **Lee, K., Lee, K., Lee, H., & Shin, J.** (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks*. Advances in Neural Information Processing Systems, 31.

7. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). *Isolation Forest*. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.

8. **Hendrycks, D., & Gimpel, K.** (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks*. In International Conference on Learning Representations.

### EfficientNet

9. **Tan, M., & Le, Q.** (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. In International Conference on Machine Learning (pp. 6105-6114). PMLR.
