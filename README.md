# Welding Detection AI

## Description

Ce projet implémente un **système de classification automatique de soudures** à partir d’images.  
Le système distingue trois classes de décisions :  
- ✅ `OK`  
- 🔧 `KO`  
- ❓ `UNKNOWN` (en cas d’incertitude ou OOD – Out-Of-Distribution)  

Chaque type de soudure (`c20`, `c33`, `c102`) possède un **modèle spécialisé** pour améliorer la précision.

---

**Ici nous nous concentrons sur l'utilisation de 3 modèles spécialisés pour cahcunes des 3 types de soudures**

## Structure du script

Le script contient deux parties principales :

1. **Entraînement des modèles spécialisés**
    - Chargement des images depuis le dataset structuré en dossiers.
    - Création d’un dataframe méta contenant le chemin des images, type de soudure et labels.
    - Application de transformations : resize, normalisation, conversion en tenseur.
    - Stratified split train / validation pour chaque type de soudure.
    - Entraînement avec `MyAIComponent` et sauvegarde du modèle : `best_model_{type}.pth`.

2. **Prédiction sur une image**
    - Chargement d’une image RGB via `cv2`.
    - Sélection automatique du modèle spécialisé selon le type de soudure.
    - Inférence avec `MyAIComponent.predict()`.
    - Retour de :
        - `predictions` → classe prédite (`OK`, `KO`, `UNKNOWN`)
        - `probabilities` → probabilités normalisées pour chaque classe
        - `OOD_scores` → score de confiance hors distribution

---