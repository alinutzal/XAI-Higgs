"""Visualization utilities for XAI-Higgs experiments."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores/probabilities
        save_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return roc_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Background', 'Signal'],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_distributions(
    X_signal: np.ndarray,
    X_background: np.ndarray,
    feature_names: List[str],
    n_features: int = 9,
    save_path: Optional[str] = None
):
    """
    Plot feature distributions for signal vs background.
    
    Args:
        X_signal: Signal samples
        X_background: Background samples
        feature_names: List of feature names
        n_features: Number of features to plot
        save_path: Path to save figure
    """
    n_rows = (n_features + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i in range(min(n_features, len(feature_names))):
        ax = axes[i]
        
        ax.hist(X_background[:, i], bins=50, alpha=0.5, label='Background', 
                color='blue', density=True)
        ax.hist(X_signal[:, i], bins=50, alpha=0.5, label='Signal', 
                color='red', density=True)
        ax.set_xlabel(feature_names[i], fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions: Signal vs Background', fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot distribution of prediction scores for signal and background.
    
    Args:
        y_true: True labels
        y_scores: Predicted scores
        save_path: Path to save figure
    """
    signal_scores = y_scores[y_true == 1]
    background_scores = y_scores[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(background_scores, bins=50, alpha=0.5, label='Background', 
             color='blue', density=True)
    plt.hist(signal_scores, bins=50, alpha=0.5, label='Signal', 
             color='red', density=True)
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Prediction Scores', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_correlation(
    X: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot feature correlation matrix.
    
    Args:
        X: Feature array
        feature_names: List of feature names
        save_path: Path to save figure
    """
    # Compute correlation matrix
    correlation = np.corrcoef(X.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    save_path: Optional[str] = None
):
    """
    Plot comparison of different models.
    
    Args:
        results: Dictionary with results for each model
        metrics: List of metrics to compare
        save_path: Path to save figure
    """
    model_names = list(results.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        
        axes[idx].bar(range(len(model_names)), values, color='steelblue')
        axes[idx].set_xticks(range(len(model_names)))
        axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].set_title(f'{metric.capitalize()} Comparison', fontsize=12)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_attribution_heatmap(
    attributions: np.ndarray,
    feature_names: List[str],
    n_samples: int = 20,
    save_path: Optional[str] = None
):
    """
    Plot heatmap of attributions for multiple samples.
    
    Args:
        attributions: Attribution values (n_samples, n_features)
        feature_names: List of feature names
        n_samples: Number of samples to display
        save_path: Path to save figure
    """
    # Select subset of samples
    n_samples = min(n_samples, attributions.shape[0])
    sample_indices = np.linspace(0, attributions.shape[0]-1, n_samples, dtype=int)
    attr_subset = attributions[sample_indices]
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(attr_subset.T, 
                yticklabels=feature_names,
                xticklabels=[f'Sample {i+1}' for i in range(n_samples)],
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Attribution'})
    plt.title('Feature Attributions Across Samples', fontsize=14)
    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
