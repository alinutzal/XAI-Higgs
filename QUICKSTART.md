# Quick Start Guide

This guide will help you get started with XAI-Higgs in just a few steps.

## Installation

```bash
# Clone the repository
git clone https://github.com/alinutzal/XAI-Higgs.git
cd XAI-Higgs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Demo (Without Dataset)

If you don't have the HIGGS dataset yet, you can run a quick demo with synthetic data:

```bash
# Train a model with synthetic data (will auto-generate if HIGGS.csv not found)
python train.py --model simple --epochs 20 --n-samples 10000
```

This will:
1. Generate 10,000 synthetic Higgs-like samples
2. Train a simple neural network for 20 epochs
3. Save the trained model to `models/`
4. Generate evaluation plots in `figures/`

## With HIGGS Dataset

### 1. Download the Dataset

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Download HIGGS dataset (3.6GB compressed)
cd data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
gunzip HIGGS.csv.gz
cd ..
```

### 2. Run Data Exploration

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

This will:
- Load and explore the Higgs dataset
- Visualize feature distributions
- Show correlations between features

### 3. Train a Model

Option A: Using the training script:
```bash
python train.py --model simple --epochs 50 --n-samples 50000
```

Option B: Using the Jupyter notebook:
```bash
jupyter notebook notebooks/02_model_training.ipynb
```

### 4. Run XAI Analysis

```bash
jupyter notebook notebooks/03_xai_analysis.ipynb
```

This will:
- Apply SHAP analysis
- Compute LIME explanations
- Use Integrated Gradients
- Compare different XAI methods

## Understanding the Results

### Model Performance

After training, you should see:
- **Accuracy**: ~70-75% (depending on model and data size)
- **AUC Score**: ~0.75-0.80
- Training and validation curves in `figures/training_history_*.png`

### XAI Insights

The XAI analysis will reveal:
1. **Most Important Features**:
   - High-level physics features (m_jj, m_jjj, m_bb)
   - Jet transverse momenta (jet_*_pt)
   - B-tagging information

2. **Feature Importance Rankings**:
   - Different XAI methods may rank features slightly differently
   - Consensus features are most reliable

3. **Individual Explanations**:
   - LIME shows why individual predictions were made
   - SHAP provides both local and global explanations

## Common Issues

### Out of Memory

If you run out of memory:
- Reduce `--n-samples` (e.g., `--n-samples 10000`)
- Reduce batch size: `--batch-size 64`
- Use simpler model: `--model simple`

### Slow Training

For faster training:
- Use GPU if available (automatically detected)
- Reduce number of epochs: `--epochs 20`
- Use smaller dataset: `--n-samples 20000`

### XAI Takes Too Long

In `notebooks/03_xai_analysis.ipynb`, reduce:
- `n_test_samples`: Number of samples to analyze (default: 500)
- `n_background`: Background samples for SHAP (default: 100)

## Next Steps

1. **Experiment with different models**:
   ```bash
   python train.py --model standard --epochs 50
   python train.py --model deep --epochs 50
   ```

2. **Compare XAI methods**: Run the XAI notebook with different models

3. **Tune hyperparameters**: Edit `config.yaml` and experiment

4. **Add new features**: Extract additional physics features from the data

5. **Try different XAI methods**: Enable GradientShap in `config.yaml`

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Open an issue on GitHub for bugs or questions
- Review the code comments in `src/` for implementation details

## Example Output

After running the full pipeline, you should have:

```
XAI-Higgs/
├── models/
│   └── higgs_classifier_simple.pth
├── figures/
│   ├── training_history_simple.png
│   ├── roc_curve_simple.png
│   ├── confusion_matrix_simple.png
│   ├── feature_distributions_*.png
│   ├── shap_summary.png
│   ├── shap_importance.png
│   ├── ig_importance.png
│   ├── deeplift_importance.png
│   └── xai_comparison.png
└── data/
    └── HIGGS.csv (if downloaded)
```

Enjoy exploring Explainable AI for Higgs physics! 🚀
