"""
Module de quantification de l'incertitude
Auteur: Sarah et Najlaa
Partie 3 : Trustworthy AI - Incertitude
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class MCDropoutModel(nn.Module):
    """Wrapper pour activer dropout pendant l'inférence"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x):
        # Forcer le mode training pour activer dropout
        return self.base_model(x)
    
    def enable_dropout(self):
        """Active dropout dans tous les modules"""
        for m in self.base_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class UncertaintyQuantifier:
    """
    Quantification complète de l'incertitude
    
    Méthodes:
        - MC Dropout: Multiple forward passes avec dropout actif
        - Temperature Scaling: Calibration des probabilités
        - Calcul d'incertitude épistémique et aléatoire
    """
    
    def __init__(self, model, device='cuda', method='mc_dropout', n_iterations=30):
        """
        Args:
            model: Modèle PyTorch
            device: 'cuda' ou 'cpu'
            method: 'mc_dropout' (recommandé)
            n_iterations: Nombre d'itérations MC Dropout
        """
        self.device = device
        self.method = method
        self.n_iterations = n_iterations
        self.temperature = 1.0
        
        if method == 'mc_dropout':
            self.model = MCDropoutModel(model).to(device)
        else:
            self.model = model.to(device)
    
    def calibrate_temperature(self, val_loader):
        """
        Calibre la température sur un validation set
        
        Args:
            val_loader: DataLoader PyTorch pour validation
        
        Returns:
            optimal_temperature: Température optimale
        """
        print("Calibration de la température...")
        
        # Collecter logits et labels
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                
                # Obtenir les logits
                if hasattr(self.model, 'base_model'):
                    outputs = self.model.base_model(images)
                else:
                    outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Optimiser la température
        def nll_loss(T):
            scaled_logits = all_logits / T
            probs = self._softmax(scaled_logits)
            nll = -np.mean(np.log(probs[np.arange(len(all_labels)), all_labels] + 1e-10))
            return nll
        
        result = minimize(nll_loss, x0=[1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        
        print(f"Température optimale : {self.temperature:.3f}")
        
        return self.temperature
    
    @staticmethod
    def _softmax(logits):
        """Softmax stable numériquement"""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def predict_with_uncertainty(self, images):
        """
        Prédictions avec quantification d'incertitude
        
        Args:
            images: Tensor PyTorch (B, C, H, W)
        
        Returns:
            dict contenant:
                - predictions: Prédictions moyennes (B, num_classes)
                - epistemic_uncertainty: Incertitude épistémique (B,)
                - aleatoric_uncertainty: Incertitude aléatoire (B,)
                - total_uncertainty: Incertitude totale (B,)
        """
        images = images.to(self.device)
        all_predictions = []
        
        if self.method == 'mc_dropout':
            # MC Dropout : T passes avec dropout actif
            self.model.enable_dropout()
            
            with torch.no_grad():
                for t in range(self.n_iterations):
                    outputs = self.model(images)
                    
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    # Appliquer température
                    probs = F.softmax(logits / self.temperature, dim=1)
                    all_predictions.append(probs.cpu().numpy())
            
            all_predictions = np.array(all_predictions)  # (T, B, C)
        
        # Calculer les statistiques
        mean_pred = all_predictions.mean(axis=0)  # (B, C)
        
        # Incertitude épistémique (variance des prédictions)
        epistemic = all_predictions.var(axis=0).mean(axis=1)  # (B,)
        
        # Incertitude totale (entropie de la prédiction moyenne)
        total = self._predictive_entropy(mean_pred)  # (B,)
        
        # Incertitude aléatoire (différence)
        aleatoric = total - epistemic
        aleatoric = np.maximum(aleatoric, 0)  # Éviter valeurs négatives
        
        return {
            'predictions': mean_pred,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total,
        }
    
    @staticmethod
    def _predictive_entropy(probs):
        """Entropie des prédictions"""
        return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calcule l'Expected Calibration Error (ECE)
    
    Args:
        y_true: Labels vrais (N,)
        y_prob: Probabilités prédites (N, C)
        n_bins: Nombre de bins
    
    Returns:
        ece: Expected Calibration Error
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_reliability_diagram(y_true, y_prob, n_bins=10, title='Reliability Diagram', save_path=None):
    """
    Plot du diagramme de fiabilité (calibration)
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_acc = []
    bin_conf = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_acc.append(accuracy_in_bin)
            bin_conf.append(avg_confidence_in_bin)
        else:
            bin_acc.append(0)
            bin_conf.append((bin_lower + bin_upper) / 2)
    
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.bar(bin_conf, bin_acc, width=0.1, alpha=0.3, edgecolor='black', 
            label=f'Model (ECE={ece:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram sauvegardé : {save_path}")
    
    plt.show()
    
    return ece


def plot_uncertainty_histogram(epistemic, aleatoric, save_path=None):
    """
    Histogrammes des incertitudes épistémique et aléatoire
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(epistemic, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Epistemic Uncertainty', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Epistemic Uncertainty', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(aleatoric, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('Aleatoric Uncertainty', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Aleatoric Uncertainty', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogrammes d'incertitude sauvegardés : {save_path}")
    
    plt.show()