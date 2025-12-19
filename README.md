# Transport Delay Predictor

An AI-powered bus delay analysis and prediction system built with machine learning models. This project provides both backend model training and a modern, interactive frontend for analyzing and predicting transportation delays.

## 📋 Overview

The Transport Delay Predictor system includes:
- **Multiple ML Models**: XGBoost, Random Forest, Linear Regression, and K-Nearest Neighbors (KNN)
- **Interactive Frontend**: Web-based UI with real-time predictions and visualizations
- **Streamlit Dashboard**: Advanced analytics and model performance monitoring
- **Data Processing Pipeline**: Automated feature engineering and preprocessing

## 🎯 Features

### Core Functionality
- **Delay Prediction**: Predict bus delays based on various features (route, weather, passenger count, time, location)
- **Model Comparison**: Compare predictions across multiple ML models simultaneously
- **Performance Metrics**: View MAE (Mean Absolute Error) and R² scores for each model
- **Data Analysis**: Exploratory data analysis with interactive visualizations
- **Historical Data Insights**: Analyze patterns and trends in historical delay data

### Machine Learning Models

| Model | Test MAE | Test R² | Status |
|-------|----------|---------|--------|
| XGBoost | 56.29 | 0.425 | ⭐ Recommended |
| Random Forest | 56.29 | 0.427 | ✓ Good |
| Linear Regression | 62.53 | 0.185 | ✓ Baseline |
| K-Nearest Neighbors | 67.72 | -0.043 | ⚠️ Reference |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js (optional, for running HTTP server)
- Git

### Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd "Transport Train Model"
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download the cleaned dataset** (if not included):
   - Place `cleaned_transport_dataset.csv` in the project root

### Running the Application

#### Option 1: Streamlit Frontend (Advanced)
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

#### Option 2: Static Web Frontend (Recommended for quick start)
```bash
# In PowerShell/Terminal
python -m http.server 8000
# Or use Node.js
npm install -g http-server
http-server
```
Access at: `http://localhost:8000`

## 📁 Project Structure

```
├── app.py                              # Streamlit application (advanced analytics)
├── app.js                              # Frontend JavaScript logic
├── index.html                          # Web frontend UI
├── styles.css                          # Frontend styling
├── train_models.py                     # Model training script
├── transport_delay_analysis.ipynb      # Jupyter notebook for analysis
├── cleaned_transport_dataset.csv       # Processed dataset
├── dirty_transport_dataset.csv         # Raw dataset
├── requirements.txt                    # Python dependencies
├── model_evaluation_summary.csv        # Model performance metrics
├── README.md                           # This file
│
├── models/                             # Trained model artifacts
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── knn.pkl
│   ├── scaler.pkl
│   ├── label_encoder_*.pkl
│   └── metadata.json
│
├── tools/                              # Utility scripts
│   ├── extract_importances.py
│   └── extract_xgb_importances.py
│
└── old/                                # Archived files
    ├── app_old.py
    └── train_models_old.py
```

## 🔧 Model Training

### Retraining Models

To retrain all models with your data:

```bash
python train_models.py
```

This script will:
1. Load and preprocess the cleaned dataset
2. Perform feature engineering
3. Split data into train/test sets
4. Train all four ML models
5. Evaluate model performance
6. Save trained models and metadata
7. Generate performance metrics CSV

### Dataset Features

The models use the following features:
- **Temporal**: Hour, Day of Week, Time of Day, Weekend indicator
- **Location**: Latitude, Longitude
- **Traffic**: Route ID, Passenger Count
- **Weather**: Weather Condition, Weather Severity

### Target Variable
- **Delay (minutes)**: Actual delay from scheduled time

## 💻 Frontend Pages

### 1. Dashboard
- Key metrics (total records, mean/median/max delay)
- Delay distribution histogram
- Delay by route analysis
- Weather impact analysis
- Passenger count correlation
- Dataset preview table

### 2. Predict Delay
- Interactive prediction form with all input parameters
- Model selection dropdown (XGBoost, Random Forest, Linear Regression, KNN)
- Real-time predictions with status badges
- Gauge chart visualization
- All-models comparison table

### 3. Data Analysis
- **Statistics Tab**: Comprehensive dataset statistics
- **Exploratory Tab**: Correlation analysis and visualizations
- **Raw Data Tab**: Searchable, filterable data table

### 4. Model Performance
- MAE comparison chart
- R² score comparison chart
- Model rankings and recommendations
- Detailed performance metrics table

## 📊 API Integration (Streamlit)

The Streamlit app (`app.py`) provides advanced features:
- Real-time model retraining interface
- Cross-validation results
- Feature importance visualization
- Shapley value explanations
- Custom prediction scenarios

## 🎓 Model Explanations

### XGBoost (Recommended)
- Gradient boosting ensemble method
- Best overall performance (R² = 0.425)
- Robust to outliers and non-linear relationships
- Suitable for production use

### Random Forest
- Ensemble of decision trees
- Good generalization (R² = 0.427)
- Provides feature importance scores
- Parallel prediction capability

### Linear Regression
- Baseline statistical model
- Interpretable coefficients
- Moderate performance (R² = 0.185)
- Fast inference

### K-Nearest Neighbors (KNN)
- Instance-based learning
- Reference model for comparison
- Lower performance (-0.043 R²)
- Useful for local pattern analysis

## 📈 Performance Metrics

Models are evaluated using:
- **MAE (Mean Absolute Error)**: Average prediction error in minutes
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
- **R² Score**: Coefficient of determination (0-1 scale)
- **Cross-Validation**: k-fold CV for stability assessment

## 🔐 Data Privacy

- No personal data is collected or stored
- Dataset contains aggregated transportation metrics only
- All model artifacts are saved locally
- No external API calls for predictions

## 🐛 Troubleshooting

### Models not loading
```bash
# Retrain models
python train_models.py
```

### Port already in use
```bash
# Change port for HTTP server
python -m http.server 9000

# For Streamlit
streamlit run app.py --server.port 8502
```

### Missing dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Dataset encoding issues
Ensure CSV files use UTF-8 encoding.

## 📝 Recent Updates

### KNN Model Addition
- Added K-Nearest Neighbors model to the ensemble
- Integrated into all UI components (frontend and Streamlit)
- Added to model comparison visualizations
- Includes performance metrics evaluation

## 🚀 Future Enhancements

- [ ] Deep Learning models (LSTM, Neural Networks)
- [ ] Real-time data ingestion
- [ ] Geographic heat maps
- [ ] Mobile app version
- [ ] REST API for external integrations
- [ ] Automated retraining pipeline
- [ ] Model explainability dashboard
- [ ] Anomaly detection for unusual delays

## 📚 Dependencies

See `requirements.txt` for full list:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms & preprocessing
- **xgboost**: Gradient boosting
- **plotly**: Interactive visualizations
- **streamlit**: Web framework
- **jupyter**: Notebook environment
- **joblib**: Model serialization

## 👨‍💻 Development

### Adding New Models

1. Train and test the model in the Jupyter notebook
2. Add model saving to `train_models.py`
3. Update `app.py` to load the new model
4. Add to `app.js` frontend model selection
5. Update performance comparison in `index.html`
6. Run tests and validate predictions

### Code Style

- Python: PEP 8 compliance
- JavaScript: ES6+ standards
- HTML/CSS: Semantic markup

## 📄 License

This project is provided as-is for educational and operational purposes.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review model training logs
3. Verify dataset format
4. Check browser console for frontend errors

## 📞 Contact

For more information about this project, please refer to the model documentation and code comments.

---

**Last Updated**: December 2025  
**Models Included**: XGBoost, Random Forest, Linear Regression, K-Nearest Neighbors  
**Dataset**: Transport Delay Analysis (500+ records)
