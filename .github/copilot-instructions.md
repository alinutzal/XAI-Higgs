# GitHub Copilot Instructions for XAI-Higgs

## Project Overview

This repository focuses on Explainable AI (XAI) techniques applied to Higgs boson physics research. The project aims to provide interpretable machine learning models for particle physics analysis.

## Technology Stack

- **Language**: Python
- **Domain**: Particle Physics, Machine Learning, Explainable AI
- **License**: Apache License 2.0

## Coding Standards

### Python Code Style
- Follow PEP 8 guidelines for Python code
- Use type hints where appropriate for better code clarity
- Write docstrings for functions, classes, and modules using Google or NumPy style
- Keep functions focused and single-purpose

### Dependencies
- Use virtual environments (venv, conda, or similar) for dependency management
- Document all required packages and their versions
- Consider scientific computing best practices (NumPy, pandas, scikit-learn conventions)

### Machine Learning Best Practices
- Always set random seeds for reproducibility
- Document model architectures and hyperparameters clearly
- Include data preprocessing steps with clear documentation
- Validate model performance with appropriate metrics

### Explainable AI Considerations
- Document the XAI techniques being used (e.g., SHAP, LIME, attention mechanisms)
- Provide clear explanations of how interpretability is achieved
- Include visualizations where helpful for understanding model decisions

## Testing
- Write unit tests for utility functions and data processing
- Include integration tests for model pipelines
- Use pytest or unittest framework
- Test edge cases and data validation

## Documentation
- Keep README.md updated with project status and usage instructions
- Document data sources and preprocessing steps
- Include examples and tutorials where applicable
- Add inline comments for complex physics or ML concepts

## Performance and Optimization
- Consider memory efficiency when working with large datasets
- Profile code for performance bottlenecks
- Use vectorized operations (NumPy, pandas) where possible
- Document computational requirements

## Physics Domain Specifics
- Use standard particle physics nomenclature and conventions
- Reference relevant physics papers or documentation
- Ensure units and physical quantities are clearly specified
- Validate results against known physics benchmarks where applicable

## Version Control
- Write clear, descriptive commit messages
- Keep commits focused and atomic
- Reference related issues in commit messages when applicable
