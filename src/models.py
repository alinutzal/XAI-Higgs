"""Neural network models for Higgs boson classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class HiggsClassifier(nn.Module):
    """
    Deep Neural Network for Higgs boson classification.
    
    Architecture: Fully connected layers with dropout and batch normalization.
    """
    
    def __init__(
        self, 
        input_dim: int = 28,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(HiggsClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Activation
            layers.append(nn.ReLU())
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output probabilities of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Predicted labels (0 or 1)
        """
        with torch.no_grad():
            probs = self.forward(x)
            return (probs >= threshold).float()


class SimpleHiggsClassifier(nn.Module):
    """
    Simple 3-layer neural network for Higgs classification.
    Lighter model suitable for XAI analysis.
    """
    
    def __init__(self, input_dim: int = 28, hidden_dim: int = 64):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
        """
        super(SimpleHiggsClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x


class DeepHiggsClassifier(nn.Module):
    """
    Deep neural network with residual connections for Higgs classification.
    More complex model for higher accuracy.
    """
    
    def __init__(self, input_dim: int = 28):
        """
        Args:
            input_dim: Number of input features
        """
        super(DeepHiggsClassifier, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, 256)
        self.bn_input = nn.BatchNorm1d(256)
        
        # Residual blocks
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.output_layer = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input layer
        x = self.input_layer(x)
        x = self.bn_input(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Residual block 1
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Residual connection
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        
        return x


def create_model(model_type: str = 'simple', input_dim: int = 28, **kwargs) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: Type of model ('simple', 'standard', 'deep')
        input_dim: Number of input features
        **kwargs: Additional arguments for the model
        
    Returns:
        PyTorch model
    """
    if model_type == 'simple':
        return SimpleHiggsClassifier(input_dim, **kwargs)
    elif model_type == 'standard':
        return HiggsClassifier(input_dim, **kwargs)
    elif model_type == 'deep':
        return DeepHiggsClassifier(input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
