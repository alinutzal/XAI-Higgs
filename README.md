# XAI-Higgs: Explainable AI for Higgs Boson Detection

This repository contains Python and PyTorch code to reproduce experiments applying Explainable AI (XAI) methods to Higgs boson classification, as described in the paper: [Explainable AI for Higgs Boson Detection](https://iopscience.iop.org/article/10.1088/1742-6596/2438/1/012082/meta).

## Overview

The project implements deep learning models for classifying Higgs boson signal events from background events, and applies various XAI methods to interpret model predictions:

- **SHAP (SHapley Additive exPlanations)**: Unified measure of feature importance based on game theory
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local surrogate models for individual predictions
- **Integrated Gradients**: Attribution through gradient integration from baseline
- **DeepLIFT**: Activation comparison to reference values
- **Gradient SHAP**: Combining SHAP with gradient-based methods

## Repository Structure

```
XAI-Higgs/
├── data/                   # Dataset directory (HIGGS.csv)
├── models/                 # Saved trained models
├── figures/                # Generated visualizations
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_xai_analysis.ipynb
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── models.py           # Neural network architectures
│   ├── trainer.py          # Training utilities
│   ├── xai_methods.py      # XAI methods implementation
│   └── visualization.py    # Plotting utilities
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/alinutzal/XAI-Higgs.git
cd XAI-Higgs
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the **HIGGS dataset** from the UCI Machine Learning Repository. The dataset contains 11 million events with 28 features:

- **21 low-level features**: kinematic properties measured by particle detectors
- **7 high-level features**: physics-derived features calculated from low-level features

### Download the Dataset

You can download the HIGGS dataset from:
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS)
- Direct link: [HIGGS.csv.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)

Place the downloaded `HIGGS.csv` file in the `data/` directory.

**Note**: If the dataset is not available, the code will automatically generate synthetic Higgs-like data for demonstration purposes.

### Features

The 28 features include:
- Lepton kinematics: pT, eta, phi
- Missing energy: magnitude and phi
- Jet properties: pT, eta, phi, b-tagging for 4 jets
- High-level features: invariant masses (m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)

## Usage

### 1. Data Exploration

Explore the dataset and visualize feature distributions:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook:
- Loads and preprocesses the Higgs dataset
- Visualizes feature distributions for signal vs background
- Computes feature correlations
- Provides dataset statistics

### 2. Model Training

Train neural network classifiers:

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

This notebook:
- Loads and prepares data with PyTorch DataLoaders
- Defines and trains neural network models
- Evaluates model performance (accuracy, precision, recall, F1)
- Generates ROC curves and confusion matrices
- Saves trained models

Available model architectures:
- **Simple**: 3-layer network (lightweight, suitable for XAI)
- **Standard**: Configurable multi-layer network with dropout
- **Deep**: Deep network with residual connections

### 3. XAI Analysis

Apply explainability methods to understand model predictions:

```bash
jupyter notebook notebooks/03_xai_analysis.ipynb
```

This notebook:
- Loads trained models
- Applies SHAP, LIME, Integrated Gradients, DeepLIFT
- Generates feature importance visualizations
- Compares different XAI methods
- Creates attribution heatmaps

## Code Examples

### Loading Data

```python
from src.data_loader import load_higgs_data, get_data_loaders

# Load data
X_train, X_test, y_train, y_test = load_higgs_data(
    data_path='data/HIGGS.csv',
    n_samples=50000,
    test_split=0.2,
    random_seed=42
)

# Create PyTorch DataLoaders
train_loader, val_loader, test_loader = get_data_loaders(
    X_train, X_test, y_train, y_test,
    batch_size=128
)
```

### Training a Model

```python
import torch
from src.models import create_model
from src.trainer import Trainer

# Create model
model = create_model(model_type='simple', input_dim=28)

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(model, device=device, learning_rate=0.001)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    early_stopping_patience=10
)

# Evaluate
metrics = trainer.evaluate(test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

### Applying XAI Methods

```python
from src.xai_methods import XAIAnalyzer
from src.data_loader import get_feature_names

# Initialize analyzer
feature_names = get_feature_names()
xai_analyzer = XAIAnalyzer(model, feature_names, device)

# Compute SHAP values
shap_values, explainer = xai_analyzer.compute_shap_values(
    X_background=X_train,
    X_test=X_test,
    n_background=100
)

# Plot feature importance
xai_analyzer.plot_feature_importance(
    shap_values,
    top_k=15,
    title='SHAP Feature Importance'
)

# Compare multiple methods
results = xai_analyzer.compare_methods(
    X_background=X_train,
    X_test=X_test,
    methods=['shap', 'ig', 'deeplift']
)
```

## Results

The XAI analysis typically reveals that the most important features for Higgs boson classification are:

1. **High-level physics features**: Invariant masses (m_jj, m_jjj, m_bb, m_wbb)
2. **Jet properties**: Transverse momentum and b-tagging information
3. **Lepton kinematics**: Lepton pT and angular coordinates

These findings align with physics intuition about Higgs boson decay signatures.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{higgs_xai_2023,
    title={Explainable AI for Higgs Boson Detection},
    author={...},
    journal={Journal of Physics: Conference Series},
    volume={2438},
    pages={012082},
    year={2023},
    publisher={IOP Publishing},
    doi={10.1088/1742-6596/2438/1/012082}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HIGGS dataset from UCI Machine Learning Repository
- PyTorch deep learning framework
- SHAP, LIME, and Captum XAI libraries
- IOP Publishing for the original research paper

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainer.

## References

1. [HIGGS Dataset - UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/HIGGS)
2. [SHAP Documentation](https://shap.readthedocs.io/)
3. [LIME Documentation](https://lime-ml.readthedocs.io/)
4. [Captum Documentation](https://captum.ai/)
5. [Original Paper](https://iopscience.iop.org/article/10.1088/1742-6596/2438/1/012082/meta)