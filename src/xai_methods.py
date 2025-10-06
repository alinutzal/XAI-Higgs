"""Explainable AI methods for Higgs classification models."""

import torch
import torch.nn as nn
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from captum.attr import IntegratedGradients, DeepLift, GradientShap
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class XAIAnalyzer:
    """Analyzer for applying various XAI methods to Higgs models."""
    
    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str],
        device: str = 'cpu'
    ):
        """
        Args:
            model: Trained PyTorch model
            feature_names: List of feature names
            device: Device to run analysis on
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_names = feature_names
        self.device = device
        
    def compute_shap_values(
        self,
        X_background: np.ndarray,
        X_test: np.ndarray,
        n_background: int = 100
    ) -> Tuple[np.ndarray, shap.Explainer]:
        """
        Compute SHAP values using DeepExplainer.
        
        Args:
            X_background: Background data for SHAP
            X_test: Test data to explain
            n_background: Number of background samples to use
            
        Returns:
            SHAP values and explainer object
        """
        # Sample background data
        if len(X_background) > n_background:
            indices = np.random.choice(len(X_background), n_background, replace=False)
            X_background = X_background[indices]
        
        # Convert to tensors
        background = torch.FloatTensor(X_background).to(self.device)
        test = torch.FloatTensor(X_test).to(self.device)
        
        # Create SHAP explainer
        explainer = shap.DeepExplainer(self.model, background)
        
        # Calculate SHAP values
        with torch.no_grad():
            shap_values = explainer.shap_values(test)
        
        # Convert to numpy if needed
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()
        
        return shap_values, explainer
    
    def compute_lime_explanation(
        self,
        X_train: np.ndarray,
        X_instance: np.ndarray,
        num_features: int = 10
    ) -> Dict:
        """
        Compute LIME explanation for a single instance.
        
        Args:
            X_train: Training data for building explainer
            X_instance: Single instance to explain
            num_features: Number of top features to show
            
        Returns:
            Dictionary with LIME explanation
        """
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=['Background', 'Signal'],
            mode='classification'
        )
        
        # Define prediction function
        def predict_fn(X):
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
            probs = outputs.cpu().numpy()
            return np.hstack([1 - probs, probs])
        
        # Get explanation
        exp = explainer.explain_instance(
            X_instance,
            predict_fn,
            num_features=num_features
        )
        
        # Extract feature importances
        feature_weights = exp.as_list()
        
        return {
            'explanation': exp,
            'feature_weights': feature_weights
        }
    
    def compute_integrated_gradients(
        self,
        X_test: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attributions.
        
        Args:
            X_test: Test data to explain
            baseline: Baseline for integration (zeros if None)
            n_steps: Number of integration steps
            
        Returns:
            Attribution values
        """
        # Create Integrated Gradients object
        ig = IntegratedGradients(self.model)
        
        # Convert to tensors
        test = torch.FloatTensor(X_test).to(self.device)
        test.requires_grad = True
        
        if baseline is None:
            baseline = torch.zeros_like(test)
        else:
            baseline = torch.FloatTensor(baseline).to(self.device)
        
        # Compute attributions
        attributions = ig.attribute(
            test,
            baseline,
            n_steps=n_steps
        )
        
        return attributions.cpu().detach().numpy()
    
    def compute_deeplift(
        self,
        X_test: np.ndarray,
        baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute DeepLIFT attributions.
        
        Args:
            X_test: Test data to explain
            baseline: Baseline for DeepLIFT
            
        Returns:
            Attribution values
        """
        # Create DeepLift object
        dl = DeepLift(self.model)
        
        # Convert to tensors
        test = torch.FloatTensor(X_test).to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(test)
        else:
            baseline = torch.FloatTensor(baseline).to(self.device)
        
        # Compute attributions
        attributions = dl.attribute(test, baseline)
        
        return attributions.cpu().detach().numpy()
    
    def compute_gradient_shap(
        self,
        X_background: np.ndarray,
        X_test: np.ndarray,
        n_samples: int = 50
    ) -> np.ndarray:
        """
        Compute Gradient SHAP attributions.
        
        Args:
            X_background: Background data
            X_test: Test data to explain
            n_samples: Number of samples for stochastic approximation
            
        Returns:
            Attribution values
        """
        # Create GradientShap object
        gs = GradientShap(self.model)
        
        # Sample background
        if len(X_background) > n_samples:
            indices = np.random.choice(len(X_background), n_samples, replace=False)
            X_background = X_background[indices]
        
        # Convert to tensors
        background = torch.FloatTensor(X_background).to(self.device)
        test = torch.FloatTensor(X_test).to(self.device)
        
        # Compute attributions
        attributions = gs.attribute(test, background)
        
        return attributions.cpu().detach().numpy()
    
    def plot_feature_importance(
        self,
        attributions: np.ndarray,
        top_k: int = 15,
        title: str = 'Feature Importance',
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance from attributions.
        
        Args:
            attributions: Attribution values (n_samples, n_features)
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        # Average absolute attributions across samples
        mean_abs_attr = np.abs(attributions).mean(axis=0)
        
        # Get top k features
        top_indices = np.argsort(mean_abs_attr)[-top_k:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_values = mean_abs_attr[top_indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_values)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Mean Absolute Attribution')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_test: np.ndarray,
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            X_test: Test data
            max_display: Maximum number of features to display
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_methods(
        self,
        X_background: np.ndarray,
        X_test: np.ndarray,
        methods: List[str] = ['shap', 'ig', 'deeplift'],
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compare different XAI methods.
        
        Args:
            X_background: Background data
            X_test: Test data
            methods: List of methods to compare
            save_path: Path to save figure
            
        Returns:
            Dictionary with attributions from each method
        """
        results = {}
        
        if 'shap' in methods:
            print("Computing SHAP values...")
            shap_values, _ = self.compute_shap_values(X_background, X_test)
            results['shap'] = shap_values
        
        if 'ig' in methods:
            print("Computing Integrated Gradients...")
            ig_attr = self.compute_integrated_gradients(X_test)
            results['ig'] = ig_attr
        
        if 'deeplift' in methods:
            print("Computing DeepLIFT...")
            dl_attr = self.compute_deeplift(X_test)
            results['deeplift'] = dl_attr
        
        # Plot comparison
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6))
        
        if n_methods == 1:
            axes = [axes]
        
        for idx, (method_name, attributions) in enumerate(results.items()):
            mean_abs_attr = np.abs(attributions).mean(axis=0)
            top_indices = np.argsort(mean_abs_attr)[-15:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_values = mean_abs_attr[top_indices]
            
            axes[idx].barh(range(len(top_features)), top_values)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features)
            axes[idx].set_xlabel('Mean Absolute Attribution')
            axes[idx].set_title(method_name.upper())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return results
