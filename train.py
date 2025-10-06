#!/usr/bin/env python
"""
Training script for Higgs boson classification.

Usage:
    python train.py --model simple --epochs 50 --batch-size 128
"""

import argparse
import os
import torch
import numpy as np

from src.data_loader import load_higgs_data, get_data_loaders
from src.models import create_model
from src.trainer import Trainer
from src.visualization import plot_training_history, plot_roc_curve, plot_confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Higgs boson classifier')
    
    parser.add_argument('--data-path', type=str, default='data/HIGGS.csv',
                        help='Path to HIGGS.csv file')
    parser.add_argument('--model', type=str, default='simple',
                        choices=['simple', 'standard', 'deep'],
                        help='Model architecture')
    parser.add_argument('--n-samples', type=int, default=50000,
                        help='Number of samples to use (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for saved models')
    parser.add_argument('--figure-dir', type=str, default='figures',
                        help='Output directory for figures')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    X_train, X_test, y_train, y_test = load_higgs_data(
        data_path=args.data_path,
        n_samples=args.n_samples,
        test_split=0.2,
        random_seed=args.seed
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        X_train, X_test, y_train, y_test,
        batch_size=args.batch_size,
        val_split=0.1
    )
    
    # Create model
    print(f"\nCreating {args.model} model...")
    input_dim = X_train.shape[1]
    model = create_model(
        model_type=args.model,
        input_dim=input_dim
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr
    )
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        verbose=True
    )
    
    # Plot training history
    print("\nSaving training history plot...")
    plot_training_history(
        history,
        save_path=os.path.join(args.figure_dir, f'training_history_{args.model}.png')
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(test_loader)
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # Get predictions
    print("\nGenerating predictions...")
    model.eval()
    y_pred_probs = []
    y_pred_labels = []
    y_true_all = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = model(data)
            probs = outputs.cpu().numpy()
            y_pred_probs.extend(probs)
            y_pred_labels.extend((probs >= 0.5).astype(int))
            y_true_all.extend(target.numpy())
    
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred_labels = np.array(y_pred_labels).flatten()
    y_true_all = np.array(y_true_all).flatten()
    
    # Plot ROC curve
    print("Saving ROC curve...")
    plot_roc_curve(
        y_true_all,
        y_pred_probs,
        save_path=os.path.join(args.figure_dir, f'roc_curve_{args.model}.png')
    )
    
    # Plot confusion matrix
    print("Saving confusion matrix...")
    plot_confusion_matrix(
        y_true_all,
        y_pred_labels,
        save_path=os.path.join(args.figure_dir, f'confusion_matrix_{args.model}.png')
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, f'higgs_classifier_{args.model}.pth')
    trainer.save_model(model_path)
    
    print(f"\nTraining complete! Model saved to {model_path}")


if __name__ == '__main__':
    main()
