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
            n_background: int = 100,
            check_additivity: bool = False
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

            print(f"Background shape: {X_background.shape}")
            print(f"Test shape: {X_test.shape}")

            # Convert to tensors with gradient tracking enabled
            background = torch.FloatTensor(X_background).to(self.device)
            background.requires_grad = True

            test = torch.FloatTensor(X_test).to(self.device)
            test.requires_grad = True
            self.model.eval()

            # Disable dropout and set batchnorm to eval mode explicitly
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.eval()

            try:
                # Create SHAP explainer
                explainer = shap.DeepExplainer(self.model, background)

                # Calculate SHAP values with additivity check disabled by default
                # This avoids errors with models containing Dropout, BatchNorm, etc.
                shap_values = explainer.shap_values(test, check_additivity=check_additivity)

                # Convert to numpy if needed
                if isinstance(shap_values, torch.Tensor):
                    shap_values = shap_values.cpu().detach().numpy()
                elif isinstance(shap_values, list):
                    # Handle multi-output case
                    shap_values = [sv.cpu().detach().numpy() if isinstance(sv, torch.Tensor) else sv
                                  for sv in shap_values]
                    # For binary classification, take the positive class
                    if len(shap_values) == 2:
                        shap_values = shap_values[1]

                print(f"Raw SHAP values shape: {shap_values.shape}")
                print(f"SHAP values type: {type(shap_values)}")

            except Exception as e:
                print(f"Warning: SHAP computation encountered an issue: {e}")
                print("Trying alternative approach with KernelExplainer...")
                # Fallback to KernelExplainer if DeepExplainer fails
                return self._compute_shap_kernel(X_background, X_test, n_background)

            return shap_values, explainer

    def _compute_shap_kernel(
        self,
        X_background: np.ndarray,
        X_test: np.ndarray,
        n_background: int = 100
    ) -> Tuple[np.ndarray, shap.Explainer]:

        # Sample background data
        if len(X_background) > n_background:
            indices = np.random.choice(len(X_background), n_background, replace=False)
            X_background = X_background[indices]

        # Define prediction function
        def model_predict(X):
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
            return outputs.cpu().numpy()

        # Create KernelExplainer
        explainer = shap.KernelExplainer(model_predict, X_background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)

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
        # Flatten if needed for LIME (it expects 2D)
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_instance_flat = X_instance.flatten()
            feature_names_expanded = [f"{name}_t{t}" for t in range(X_train.shape[1]) for name in self.feature_names]
        else:
            X_train_flat = X_train
            X_instance_flat = X_instance
            feature_names_expanded = self.feature_names

        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X_train_flat,
            feature_names=feature_names_expanded,
            class_names=['Background', 'Signal'],
            mode='classification'
        )

        # Define prediction function
        def predict_fn(X):
            # Reshape back if needed
            if len(self.feature_names) != X.shape[1]:
                original_shape = (-1,) + tuple(X_train.shape[1:])
                X = X.reshape(original_shape)

            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
            probs = outputs.cpu().numpy()
            return np.hstack([1 - probs, probs])

        # Get explanation
        exp = explainer.explain_instance(
            X_instance_flat,
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

    def _prepare_data_for_plotting(
        self,
        shap_values: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Handle 3D SHAP values (e.g., from time series models)
        if len(shap_values.shape) == 3:
            print(f"Detected 3D SHAP values: {shap_values.shape}")
            if shap_values.shape[-1] == 1:
                # Squeeze last dimension: (n_samples, n_timesteps, 1) -> (n_samples, n_timesteps)
                shap_values_2d = shap_values.squeeze(-1)
                print(f"Squeezed to 2D: {shap_values_2d.shape}")
            else:
                # Flatten: (n_samples, n_timesteps, n_features) -> (n_samples, n_timesteps*n_features)
                shap_values_2d = shap_values.reshape(shap_values.shape[0], -1)
                print(f"Flattened to 2D: {shap_values_2d.shape}")
        else:
            shap_values_2d = shap_values

        # Handle 3D test data
        if len(X_test.shape) == 3:
            print(f"Detected 3D test data: {X_test.shape}")
            if X_test.shape[-1] == 1:
                X_test_2d = X_test.squeeze(-1)
                print(f"Squeezed to 2D: {X_test_2d.shape}")
            else:
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                print(f"Flattened to 2D: {X_test_2d.shape}")
        else:
            X_test_2d = X_test

        return shap_values_2d, X_test_2d

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
            attributions: Attribution values (n_samples, n_features) or 3D
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        # Handle 3D attributions
        if len(attributions.shape) == 3:
            attributions, _ = self._prepare_data_for_plotting(attributions, attributions)

        # Average absolute attributions across samples
        mean_abs_attr = np.abs(attributions).mean(axis=0)

        # Get top k features
        top_indices = np.argsort(mean_abs_attr)[-top_k:]

        # Create feature names for time series if needed
        if len(self.feature_names) < mean_abs_attr.shape[0]:
            # Time series case: create names like "feature_t0", "feature_t1", etc.
            n_timesteps = mean_abs_attr.shape[0] // len(self.feature_names)
            feature_names_expanded = [f"{name}_t{t}" for t in range(n_timesteps) for name in self.feature_names]
        else:
            feature_names_expanded = self.feature_names

        top_features = [feature_names_expanded[i] if i < len(feature_names_expanded) else f"Feature {i}"
                       for i in top_indices]
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
            shap_values: SHAP values (2D or 3D)
            X_test: Test data (2D or 3D)
            max_display: Maximum number of features to display
            save_path: Path to save figure
        """
        # Prepare data for plotting
        shap_values_2d, X_test_2d = self._prepare_data_for_plotting(shap_values, X_test)

        # Create feature names for time series if needed
        if len(self.feature_names) < shap_values_2d.shape[1]:
            n_timesteps = shap_values_2d.shape[1] // len(self.feature_names)
            feature_names = [f"{name}_t{t}" for t in range(n_timesteps) for name in self.feature_names]
        else:
            feature_names = self.feature_names

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_2d,
            X_test_2d,
            feature_names=feature_names[:shap_values_2d.shape[1]],
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
            # Handle 3D attributions
            if len(attributions.shape) == 3:
                attributions_2d, _ = self._prepare_data_for_plotting(attributions, attributions)
            else:
                attributions_2d = attributions

            mean_abs_attr = np.abs(attributions_2d).mean(axis=0)
            top_indices = np.argsort(mean_abs_attr)[-15:]

            # Create feature names
            if len(self.feature_names) < mean_abs_attr.shape[0]:
                n_timesteps = mean_abs_attr.shape[0] // len(self.feature_names)
                feature_names_expanded = [f"{name}_t{t}" for t in range(n_timesteps) for name in self.feature_names]
            else:
                feature_names_expanded = self.feature_names

            top_features = [feature_names_expanded[i] if i < len(feature_names_expanded) else f"Feature {i}"
                           for i in top_indices]
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