"""Data loading and preprocessing for Higgs dataset."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class HiggsDataset(Dataset):
    """PyTorch Dataset for Higgs boson detection."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """
        Args:
            data: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            transform: Optional transform to be applied on features
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


def load_higgs_data(
    data_path: str,
    n_samples: Optional[int] = None,
    test_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess Higgs dataset.
    
    The Higgs dataset typically has 28 features:
    - 21 low-level features
    - 7 high-level physics features
    
    Args:
        data_path: Path to the Higgs CSV file
        n_samples: Number of samples to load (None for all)
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Feature names for Higgs dataset
    feature_names = [
        'lepton_pT', 'lepton_eta', 'lepton_phi',
        'missing_energy_magnitude', 'missing_energy_phi',
        'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
        'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
        'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
        'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
        'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Generating synthetic Higgs-like data for demonstration...")
        return generate_synthetic_higgs_data(n_samples or 10000, test_split, random_seed)
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, nrows=n_samples, header=None)
    
    # First column is the label, rest are features
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Split into train and test
    np.random.seed(random_seed)
    n_test = int(len(X) * test_split)
    indices = np.random.permutation(len(X))
    
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Class distribution - Train: {np.mean(y_train):.2%} signal")
    print(f"Class distribution - Test: {np.mean(y_test):.2%} signal")
    
    return X_train, X_test, y_train, y_test


def generate_synthetic_higgs_data(
    n_samples: int = 10000,
    test_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Higgs-like data for demonstration purposes.
    
    Args:
        n_samples: Total number of samples to generate
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    
    n_features = 28
    n_signal = n_samples // 2
    n_background = n_samples - n_signal
    
    # Generate signal events (label = 1)
    X_signal = np.random.randn(n_signal, n_features)
    # Add some structure to make signal distinguishable
    X_signal[:, 0] += 1.5  # lepton pT higher for signal
    X_signal[:, 5] += 1.0  # jet 1 pT higher for signal
    X_signal[:, 21] += 2.0  # m_jj (dijet mass) characteristic peak
    y_signal = np.ones(n_signal)
    
    # Generate background events (label = 0)
    X_background = np.random.randn(n_background, n_features)
    y_background = np.zeros(n_background)
    
    # Combine and shuffle
    X = np.vstack([X_signal, X_background])
    y = np.concatenate([y_signal, y_background])
    
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    
    # Split into train and test
    n_test = int(n_samples * test_split)
    X_train, X_test = X[n_test:], X[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Generated {n_samples} synthetic Higgs-like samples")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Number of features: {n_features}")
    
    return X_train, X_test, y_train, y_test


def get_data_loaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create training dataset and split into train/val
    train_dataset = HiggsDataset(X_train, y_train)
    n_val = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    
    train_subset, val_subset = random_split(
        train_dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create test dataset
    test_dataset = HiggsDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def get_feature_names() -> list:
    """Return the list of feature names for the Higgs dataset."""
    return [
        'lepton_pT', 'lepton_eta', 'lepton_phi',
        'missing_energy_magnitude', 'missing_energy_phi',
        'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag',
        'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_b-tag',
        'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 'jet_3_b-tag',
        'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_b-tag',
        'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]
