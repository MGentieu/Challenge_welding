"""
Module de détection Out-of-Distribution (OOD)
Auteur: Sarah/Najlaa
Partie 3 : Trustworthy AI - OOD Detection
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class OODDetector:
    """
    Détection multi-méthodes de samples Out-of-Distribution
    
    Méthodes disponibles:
        - Mahalanobis Distance: Distance statistique dans l'espace des features
        - Isolation Forest: Détection d'outliers multivariés
        - Energy-based: Score basé sur l'énergie des logits
        - Softmax: Maximum softmax probability (baseline)
    """
    
    def __init__(self, methods=['mahalanobis', 'isolation_forest', 'energy']):
        """
        Args:
            methods: Liste des méthodes à utiliser
        """
        self.methods = methods
        self.fitted = False
        
        # Stockage des paramètres pour chaque méthode
        self.mahalanobis_params = {}
        self.isolation_forest = None
        self.energy_threshold = None
        
    def fit(self, train_features=None, train_logits=None, contamination=0.05):
        """
        Fit les détecteurs OOD sur les données d'entraînement
        
        Args:
            train_features: Features extraites (N, D) - pour Mahalanobis, Isolation Forest
            train_logits: Logits du modèle (N, C) - pour Energy
            contamination: Proportion attendue de samples OOD
        """
        print("Fitting OOD detectors...")
        
        if 'mahalanobis' in self.methods and train_features is not None:
            self._fit_mahalanobis(train_features)
        
        if 'isolation_forest' in self.methods and train_features is not None:
            self._fit_isolation_forest(train_features, contamination)
        
        if 'energy' in self.methods and train_logits is not None:
            self._fit_energy(train_logits)
        
        self.fitted = True
        print("OOD detectors fitted")
    
    def _fit_mahalanobis(self, train_features):
        """Fit Mahalanobis distance detector"""
        print("Fitting Mahalanobis...")
        
        self.mahalanobis_params['mean'] = train_features.mean(axis=0)
        
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(train_features)
        self.mahalanobis_params['cov_inv'] = np.linalg.inv(cov_estimator.covariance_)
        
        # Calculer le seuil (95e percentile)
        train_distances = np.array([
            mahalanobis(x, self.mahalanobis_params['mean'], 
                       self.mahalanobis_params['cov_inv'])
            for x in train_features
        ])
        self.mahalanobis_params['threshold'] = np.percentile(train_distances, 95)
    
    def _fit_isolation_forest(self, train_features, contamination):
        """Fit Isolation Forest detector"""
        print("Fitting Isolation Forest...")
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(train_features)
    
    def _fit_energy(self, train_logits, temperature=1.0):
        """Fit Energy-based detector"""
        print("Fitting Energy detector...")
        
        train_energies = self._compute_energy(train_logits, temperature)
        self.energy_threshold = np.percentile(train_energies, 5)  # 5e percentile
    
    @staticmethod
    def _compute_energy(logits, temperature=1.0):
        """Calcule l'énergie des logits"""
        # Energy = -T * log(sum(exp(logits / T)))
        scaled_logits = logits / temperature
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        energy = -temperature * (np.log(np.sum(exp_logits, axis=1)) + max_logits.squeeze())
        return energy
    
    def detect(self, test_features=None, test_logits=None, test_probs=None):
        """
        Détecte les samples OOD
        
        Args:
            test_features: Features du test set (N, D)
            test_logits: Logits du test set (N, C)
            test_probs: Probabilités du test set (N, C)
        
        Returns:
            dict avec pour chaque méthode:
                - scores: Scores OOD (plus haut = plus OOD)
                - is_ood: Prédictions binaires (True = OOD)
        """
        if not self.fitted:
            raise ValueError("OOD detectors not fitted. Call fit() first.")
        
        results = {}
        
        if 'mahalanobis' in self.methods and test_features is not None:
            results['mahalanobis'] = self._detect_mahalanobis(test_features)
        
        if 'isolation_forest' in self.methods and test_features is not None:
            results['isolation_forest'] = self._detect_isolation_forest(test_features)
        
        if 'energy' in self.methods and test_logits is not None:
            results['energy'] = self._detect_energy(test_logits)
        
        if 'softmax' in self.methods and test_probs is not None:
            results['softmax'] = self._detect_softmax(test_probs)
        
        # Combiner les scores (ensemble)
        if len(results) > 1:
            results['ensemble'] = self._ensemble_scores(results)
        
        return results
    
    def _detect_mahalanobis(self, test_features):
        """Détection par distance de Mahalanobis"""
        distances = np.array([
            mahalanobis(x, self.mahalanobis_params['mean'], 
                       self.mahalanobis_params['cov_inv'])
            for x in test_features
        ])
        
        is_ood = distances > self.mahalanobis_params['threshold']
        
        return {
            'scores': distances,
            'is_ood': is_ood,
            'threshold': self.mahalanobis_params['threshold']
        }
    
    def _detect_isolation_forest(self, test_features):
        """Détection par Isolation Forest"""
        predictions = self.isolation_forest.predict(test_features)  # 1 = inlier, -1 = outlier
        scores = -self.isolation_forest.score_samples(test_features)  # Plus haut = plus OOD
        
        is_ood = (predictions == -1)
        
        return {
            'scores': scores,
            'is_ood': is_ood,
            'threshold': 0
        }
    
    def _detect_energy(self, test_logits):
        """Détection par énergie"""
        energies = self._compute_energy(test_logits)
        
        is_ood = energies < self.energy_threshold
        
        return {
            'scores': -energies,  # Inverser pour que plus haut = plus OOD
            'is_ood': is_ood,
            'threshold': -self.energy_threshold
        }
    
    def _detect_softmax(self, test_probs, threshold=0.7):
        """Détection par maximum softmax probability"""
        max_probs = np.max(test_probs, axis=1)
        
        is_ood = max_probs < threshold
        
        return {
            'scores': 1 - max_probs,  # Plus haut = plus OOD
            'is_ood': is_ood,
            'threshold': 1 - threshold
        }
    
    def _ensemble_scores(self, results):
        """Combine les scores de plusieurs méthodes"""
        # Normaliser les scores entre 0 et 1
        normalized_scores = []
        
        for method, result in results.items():
            scores = result['scores']
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            normalized_scores.append(scores_norm)
        
        # Moyenne des scores
        ensemble_scores = np.mean(normalized_scores, axis=0)
        
        # Vote majoritaire pour is_ood
        votes = np.array([result['is_ood'] for result in results.values()])
        ensemble_is_ood = (votes.sum(axis=0) > len(votes) / 2)
        
        return {
            'scores': ensemble_scores,
            'is_ood': ensemble_is_ood,
            'threshold': 0.5
        }


def evaluate_ood_detection(y_true_ood, scores):
    """
    Évalue la performance de détection OOD
    
    Args:
        y_true_ood: Ground truth (0 = in-dist, 1 = OOD)
        scores: Scores OOD prédits (plus haut = plus OOD)
    
    Returns:
        dict avec métriques: AUROC, AUPR, FPR@95TPR
    """
    # AUROC
    auroc = roc_auc_score(y_true_ood, scores)
    
    # AUPR (Average Precision)
    aupr = average_precision_score(y_true_ood, scores)
    
    # FPR @ 95% TPR
    fpr, tpr, thresholds = roc_curve(y_true_ood, scores)
    fpr_at_95tpr = fpr[np.argmax(tpr >= 0.95)] if np.any(tpr >= 0.95) else 1.0
    
    return {
        'AUROC': auroc,
        'AUPR': aupr,
        'FPR@95TPR': fpr_at_95tpr
    }


def plot_ood_roc_curve(y_true_ood, scores_dict, save_path=None):
    """
    Plot ROC curves pour plusieurs méthodes OOD
    """
    plt.figure(figsize=(10, 8))
    
    for method, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true_ood, scores)
        auroc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{method} (AUROC={auroc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - OOD Detection', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve sauvegardée : {save_path}")
    
    plt.show()


def plot_ood_score_distribution(in_dist_scores, ood_scores, method_name='OOD', save_path=None):
    """
    Histogrammes des scores In-Dist vs OOD
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(in_dist_scores, bins=50, alpha=0.5, label='In-Distribution', color='blue')
    plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD', color='red')
    
    plt.xlabel(f'{method_name} Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(f'{method_name} Score Distribution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot sauvegardée : {save_path}")
    
    plt.show()