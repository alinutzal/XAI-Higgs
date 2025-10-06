# Implementation Notes

## Overview

This repository implements a complete pipeline for applying Explainable AI (XAI) methods to Higgs boson detection, reproducing experiments from the paper: https://iopscience.iop.org/article/10.1088/1742-6596/2438/1/012082/meta

## What Has Been Implemented

### 1. Data Processing (`src/data_loader.py`)
- **HiggsDataset**: PyTorch Dataset class for Higgs data
- **load_higgs_data()**: Load and preprocess HIGGS.csv with standardization
- **generate_synthetic_higgs_data()**: Generate synthetic data when real dataset unavailable
- **get_data_loaders()**: Create train/val/test PyTorch DataLoaders
- **get_feature_names()**: Return list of 28 Higgs feature names

Features handled:
- 21 low-level kinematic features (lepton, jets, missing energy)
- 7 high-level physics features (invariant masses)

### 2. Neural Network Models (`src/models.py`)
Three model architectures implemented:

- **SimpleHiggsClassifier**: Lightweight 3-layer network
  - Input → 64 → 32 → Output
  - Batch normalization and dropout
  - ~4K parameters

- **HiggsClassifier**: Configurable multi-layer network
  - Customizable hidden layers (default: [128, 64, 32])
  - Batch normalization, ReLU, dropout
  - ~14K parameters (default config)

- **DeepHiggsClassifier**: Deep network with residual connections
  - 256 → 256 (residual) → 128 → 64 → 32 → Output
  - Batch normalization throughout
  - ~118K parameters

All models:
- Binary classification (signal vs background)
- Sigmoid output activation
- Support for PyTorch training pipeline

### 3. Training Infrastructure (`src/trainer.py`)
- **Trainer** class with:
  - Binary cross-entropy loss
  - Adam optimizer with weight decay
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping based on validation loss
  - Training history tracking
  - Comprehensive evaluation metrics (accuracy, precision, recall, F1)
  - Model save/load functionality

### 4. XAI Methods (`src/xai_methods.py`)
- **XAIAnalyzer** class implementing:

  1. **SHAP (SHapley Additive exPlanations)**
     - DeepExplainer for neural networks
     - Global feature importance
     - Individual prediction explanations
     - Summary plots

  2. **LIME (Local Interpretable Model-agnostic Explanations)**
     - Tabular explainer for individual predictions
     - Local surrogate models
     - Feature contribution visualization

  3. **Integrated Gradients**
     - Path integration from baseline
     - Attribution through gradients
     - Configurable number of steps

  4. **DeepLIFT**
     - Activation-based attribution
     - Comparison to reference activations

  5. **Gradient SHAP**
     - Combines gradients with SHAP
     - Stochastic approximation

  All methods support:
  - Feature importance ranking
  - Visualization
  - Comparison across methods

### 5. Visualization Tools (`src/visualization.py`)
Comprehensive plotting functions:
- **plot_training_history()**: Loss and accuracy curves
- **plot_roc_curve()**: ROC curve with AUC score
- **plot_confusion_matrix()**: Classification confusion matrix
- **plot_feature_distributions()**: Signal vs background distributions
- **plot_prediction_distribution()**: Model prediction scores
- **plot_feature_correlation()**: Feature correlation heatmap
- **plot_model_comparison()**: Compare multiple models
- **plot_attribution_heatmap()**: XAI attribution heatmap

### 6. Jupyter Notebooks

#### `01_data_exploration.ipynb`
- Load HIGGS dataset or generate synthetic data
- Visualize feature distributions (signal vs background)
- Compute feature correlations
- Dataset statistics and class balance

#### `02_model_training.ipynb`
- Create PyTorch DataLoaders
- Train neural network models
- Visualize training progress
- Evaluate on test set (accuracy, precision, recall, F1, AUC)
- Generate ROC curves and confusion matrices
- Save trained models

#### `03_xai_analysis.ipynb`
- Load trained models
- Apply SHAP analysis with summary plots
- Compute LIME explanations for individual predictions
- Calculate Integrated Gradients
- Apply DeepLIFT
- Compare all XAI methods side-by-side
- Generate feature importance rankings
- Create attribution heatmaps

### 7. Command-Line Tools

#### `train.py`
Full-featured training script with argparse:
```bash
python train.py --model simple --epochs 50 --n-samples 50000 --batch-size 128
```

Options:
- Model architecture selection
- Data size and batch size
- Learning rate and training epochs
- Early stopping patience
- Output directories

#### `example.py`
Quick demonstration script:
- Loads/generates small dataset
- Trains simple model
- Applies XAI methods
- Shows top important features
- Complete workflow in ~5 minutes

### 8. Configuration

#### `config.yaml`
Centralized configuration for:
- Data paths and parameters
- Model architectures
- Training hyperparameters
- XAI settings
- Output directories
- Feature names

#### `requirements.txt`
Complete dependency list:
- PyTorch and torchvision
- NumPy, pandas, scikit-learn
- SHAP, LIME, Captum (XAI libraries)
- Matplotlib, seaborn, plotly (visualization)
- Jupyter and related tools
- Data processing utilities

#### `setup.py`
Python package setup for installation:
```bash
pip install -e .
```

### 9. Documentation

#### `README.md`
Comprehensive guide covering:
- Project overview and features
- Installation instructions
- Dataset download and preparation
- Usage examples for all components
- Code examples
- Citation information
- References

#### `QUICKSTART.md`
Quick start guide with:
- Minimal installation steps
- Quick demo without dataset
- Step-by-step workflow
- Common issues and solutions
- Example output structure

#### `IMPLEMENTATION_NOTES.md` (this file)
Technical implementation details

## Testing Results

Successfully tested:
1. ✓ Data loading (real and synthetic)
2. ✓ Model creation (all three architectures)
3. ✓ Forward pass and output validation
4. ✓ Training loop (2 epochs, convergence verified)
5. ✓ Evaluation metrics computation
6. ✓ Model save/load functionality
7. ✓ Command-line training script
8. ✓ Figure generation (training history, ROC, confusion matrix)

## Key Design Decisions

### 1. Synthetic Data Generation
- Enables testing without downloading 3.6GB dataset
- Generates 28 features matching real HIGGS dataset
- Adds realistic structure (signal features have higher means)
- Maintains proper class balance

### 2. Model Architecture Choices
- **Simple**: Fast training, good for XAI analysis (fewer parameters)
- **Standard**: Configurable, balanced performance/complexity
- **Deep**: High capacity, residual connections for deeper networks

### 3. XAI Method Selection
- **SHAP**: Most comprehensive, good for global understanding
- **LIME**: Best for explaining individual predictions
- **Integrated Gradients**: Theoretically grounded, gradient-based
- **DeepLIFT**: Good for deep networks, activation-based
- Multiple methods provide consensus on feature importance

### 4. Modular Design
- Separate modules for data, models, training, XAI, visualization
- Each component can be used independently
- Easy to extend with new models or XAI methods
- Clear interfaces between modules

### 5. Output Organization
```
XAI-Higgs/
├── data/          # Dataset files
├── models/        # Saved model weights
├── figures/       # Generated visualizations
├── notebooks/     # Jupyter notebooks
└── src/           # Source code modules
```

## Physics Insights

The XAI analysis typically reveals:

1. **Most Important Features**:
   - High-level physics features: m_jj, m_jjj, m_bb, m_wbb (invariant masses)
   - Jet properties: pT (transverse momentum), b-tagging
   - Lepton kinematics: lepton pT

2. **Physical Interpretation**:
   - Higgs boson decays produce specific invariant mass signatures
   - b-jets are strong indicators (Higgs → bb̄)
   - High pT jets characteristic of signal events
   - Missing energy patterns differ between signal and background

3. **Model Behavior**:
   - Models learn physics-motivated features
   - High-level features capture domain knowledge
   - Low-level features provide additional discrimination
   - Consistent across different XAI methods

## Performance Expectations

With full HIGGS dataset (11M events):
- **Accuracy**: 70-75%
- **AUC**: 0.75-0.80
- **Training time**: 10-30 minutes (GPU), 1-2 hours (CPU)

With synthetic data (demo):
- **Accuracy**: 50-60% (less structured than real data)
- **AUC**: 0.55-0.65
- **Training time**: <1 minute

## Future Enhancements

Potential improvements:
1. Add more model architectures (CNNs, Transformers)
2. Implement additional XAI methods (Attention, GradCAM)
3. Add hyperparameter tuning (Optuna, Ray Tune)
4. Support for distributed training
5. Real-time XAI dashboard
6. Uncertainty quantification
7. Adversarial robustness analysis
8. Model compression and deployment

## Known Limitations

1. Synthetic data is simplified vs real HIGGS data
2. XAI methods can be computationally expensive
3. No GPU-specific optimizations implemented
4. Limited hyperparameter search capabilities
5. No distributed training support

## References

1. HIGGS Dataset: https://archive.ics.uci.edu/ml/datasets/HIGGS
2. Original Paper: https://iopscience.iop.org/article/10.1088/1742-6596/2438/1/012082/meta
3. SHAP: https://github.com/slundberg/shap
4. LIME: https://github.com/marcotcr/lime
5. Captum: https://captum.ai/

## Contact

For questions or contributions, please open an issue on GitHub.
