"""Training utilities for Higgs classification models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


class Trainer:
    """Trainer class for PyTorch models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Args:
            model: PyTorch model to train
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            target = target.view(-1, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (output >= 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.view(-1, 1)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predicted = (output >= 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                predicted = (output >= 0.5).float()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(output.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = np.mean(all_predictions == all_targets)
        
        # True positives, false positives, true negatives, false negatives
        tp = np.sum((all_predictions == 1) & (all_targets == 1))
        fp = np.sum((all_predictions == 1) & (all_targets == 0))
        tn = np.sum((all_predictions == 0) & (all_targets == 0))
        fn = np.sum((all_predictions == 0) & (all_targets == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
