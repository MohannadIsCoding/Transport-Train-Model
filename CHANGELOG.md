# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - December 2025

### Added
- **K-Nearest Neighbors (KNN) Model**: New regression model for delay prediction
  - Implemented in `train_models.py` (lines 155-178)
  - Integrated into Streamlit app (`app.py`)
  - Added to static web frontend (`index.html`, `app.js`)
  - Performance: Test MAE 67.72, R² -0.04
  
- **Comprehensive Documentation**:
  - `README.md`: Complete project overview and feature guide
  - `ARCHITECTURE.md`: System design and component details
  - `INSTALL.md`: Step-by-step installation instructions
  - `CHANGELOG.md`: Version history (this file)

- **Model Comparison Enhancements**:
  - Added 4th color to chart visualizations for KNN
  - Updated model selection dropdown in frontend
  - Extended all prediction comparison tables

### Changed
- Updated `appState.models` in `app.js` to include KNN metrics
- Extended prediction simulation to generate KNN predictions
- Modified chart rendering to accommodate 4 models
- Updated model performance table styling

### Performance
- Total Models: 4 (XGBoost, Random Forest, Linear Regression, KNN)
- Best Model: XGBoost (R² = 0.425)
- Worst Model: KNN (R² = -0.043)

## [1.0.0] - Previous Release

### Initial Features
- XGBoost regression model
- Random Forest regression model
- Linear Regression baseline model
- Feature engineering pipeline
- Streamlit web interface
- Static HTML/JS frontend
- Interactive data visualizations
- Model evaluation metrics
- Jupyter notebook for analysis

### Frontend Capabilities
- Dashboard with key metrics
- Delay distribution analysis
- Prediction interface
- Model comparison
- Raw data viewer
- Statistics and exploratory analysis

---

**Current Version**: 1.1.0  
**Release Date**: December 20, 2025  
**Status**: Stable
