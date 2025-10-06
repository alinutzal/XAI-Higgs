#!/usr/bin/env python
"""
Simple example script demonstrating the XAI-Higgs workflow.

This script:
1. Loads/generates data
2. Trains a simple model
3. Applies XAI methods
4. Generates visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_higgs_data, get_data_loaders, get_feature_names
from src.models import create_model
from src.trainer import Trainer
from src.xai_methods import XAIAnalyzer
from src.visualization import plot_training_history

# Configuration
CONFIG = {
    'n_samples': 5000,      # Small dataset for quick demo
    'batch_size': 64,
    'epochs': 10,
    'model_type': 'simple'
}

def main():
    print("=" * 60)
    print("XAI-Higgs Example Script")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Step 1: Load Data
    print("\n" + "=" * 60)
    print("Step 1: Loading Data")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = load_higgs_data(
        data_path='data/HIGGS.csv',
        n_samples=CONFIG['n_samples'],
        test_split=0.2,
        random_seed=42
    )
    
    print(f"\nDataset Summary:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Signal ratio (train): {np.mean(y_train):.2%}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        X_train, X_test, y_train, y_test,
        batch_size=CONFIG['batch_size']
    )
    
    # Step 2: Train Model
    print("\n" + "=" * 60)
    print("Step 2: Training Model")
    print("=" * 60)
    
    model = create_model(
        model_type=CONFIG['model_type'],
        input_dim=X_train.shape[1]
    )
    
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001
    )
    
    print(f"\nModel: {CONFIG['model_type']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG['epochs'],
        early_stopping_patience=5,
        verbose=True
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Step 3: Evaluating Model")
    print("=" * 60)
    
    metrics = trainer.evaluate(test_loader)
    
    print("\nTest Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # Step 3: XAI Analysis
    print("\n" + "=" * 60)
    print("Step 4: Applying XAI Methods")
    print("=" * 60)
    
    feature_names = get_feature_names()
    xai_analyzer = XAIAnalyzer(
        model=model,
        feature_names=feature_names,
        device=device
    )
    
    # Use subset for faster XAI computation
    X_test_xai = X_test[:200]
    
    # SHAP
    print("\nComputing SHAP values...")
    shap_values, _ = xai_analyzer.compute_shap_values(
        X_background=X_train,
        X_test=X_test_xai,
        n_background=50
    )
    
    # Integrated Gradients
    print("Computing Integrated Gradients...")
    ig_attr = xai_analyzer.compute_integrated_gradients(X_test_xai)
    
    # Get top features from SHAP
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_importance)[-10:]
    
    print("\nTop 10 Most Important Features (SHAP):")
    for i, idx in enumerate(reversed(top_indices), 1):
        print(f"  {i:2d}. {feature_names[idx]:30s} {shap_importance[idx]:.4f}")
    
    # Compare methods
    print("\nComparing XAI methods...")
    comparison = xai_analyzer.compare_methods(
        X_background=X_train,
        X_test=X_test_xai[:50],  # Small subset for demo
        methods=['shap', 'ig']
    )
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Model trained successfully with reasonable performance")
    print("2. XAI methods reveal important physics features")
    print("3. High-level features (masses) are typically most important")
    print("\nFor detailed analysis, run the Jupyter notebooks:")
    print("  - notebooks/01_data_exploration.ipynb")
    print("  - notebooks/02_model_training.ipynb")
    print("  - notebooks/03_xai_analysis.ipynb")
    

if __name__ == '__main__':
    main()
