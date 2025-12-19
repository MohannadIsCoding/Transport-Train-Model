# Architecture Documentation

## System Architecture

### Overview
The Transport Delay Predictor follows a **Model-View-Controller (MVC)** pattern with separation between backend ML operations and frontend presentation layers.

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend Layer                         │
├─────────────────────────────────────────────────────────┤
│  Static Web UI (index.html, app.js, styles.css)         │
│  Streamlit Dashboard (app.py)                           │
│  Jupyter Notebook (transport_delay_analysis.ipynb)      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
├─────────────────────────────────────────────────────────┤
│  Data Loading & Caching (app.py, app.js)               │
│  Form Handling & Validation                            │
│  Prediction Orchestration                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Model Layer                            │
├─────────────────────────────────────────────────────────┤
│  XGBoost Model                                          │
│  Random Forest Model                                    │
│  Linear Regression Model                               │
│  K-Nearest Neighbors Model                             │
│  Feature Preprocessing & Scaling                       │
│  Label Encoding                                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Data Layer                             │
├─────────────────────────────────────────────────────────┤
│  CSV Datasets (cleaned & raw)                          │
│  Persisted Models (.pkl files)                         │
│  Metadata & Metrics (JSON, CSV)                        │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend Layer

#### 1. Static Web UI
- **Files**: `index.html`, `app.js`, `styles.css`
- **Technology**: HTML5, Vanilla JavaScript, CSS3, Plotly.js
- **Features**:
  - Single-page application with tabbed navigation
  - Client-side prediction simulation
  - Real-time chart rendering
  - Responsive design for all devices

#### 2. Streamlit Dashboard
- **File**: `app.py`
- **Technology**: Python Streamlit framework
- **Features**:
  - Advanced analytics and model comparison
  - Real model inference (loads trained .pkl files)
  - Custom prediction scenarios
  - Feature importance visualization
  - Cross-validation metrics

#### 3. Jupyter Notebook
- **File**: `transport_delay_analysis.ipynb`
- **Purpose**: Research, experimentation, model exploration
- **Features**: Cell-based interactive analysis

### Model Layer

#### Feature Engineering Pipeline
```
Raw Data (CSV)
    ↓
Temporal Features (hour, day_of_week, time_of_day, weekend)
    ↓
Location Features (latitude, longitude)
    ↓
Traffic Features (route_id, passenger_count)
    ↓
Weather Features (weather, weather_severity)
    ↓
Label Encoding (categorical → numeric)
    ↓
Scaling (StandardScaler for distance-based models)
    ↓
Train/Test Split (80/20)
    ↓
Model Training
```

#### Model Pipeline

```python
# Preprocessing Pipeline
Input Features → Label Encoding → StandardScaler → Model

# Output
Predicted Delay (minutes) ← Model Prediction
```

### Data Layer

#### Stored Artifacts
```
models/
├── xgboost.pkl              # XGBoost model
├── random_forest.pkl        # Random Forest model
├── linear_regression.pkl    # Linear Regression model
├── knn.pkl                  # KNN model
├── scaler.pkl               # StandardScaler object
├── label_encoder_route.pkl  # Route ID encoder
├── label_encoder_time_of_day.pkl
├── label_encoder_weather.pkl
├── label_encoder_weather_severity.pkl
└── metadata.json            # Model metadata
```

## Data Flow

### Training Flow
```
cleaned_transport_dataset.csv
    ↓
train_models.py (train_models.py)
├─ Load & Preprocess
├─ Feature Engineering
├─ Train/Test Split
├─ Train All Models (XGB, RF, LR, KNN)
├─ Evaluate Metrics
├─ Save Models
└─ Generate Metadata
    ↓
models/ (directory)
model_evaluation_summary.csv
```

### Prediction Flow

#### Streamlit (Real)
```
User Input (Streamlit Form)
    ↓
prepare_features() → Encode & Scale
    ↓
Load Model from models/*.pkl
    ↓
model.predict()
    ↓
Display Result (with metrics)
```

#### Static Web (Simulated)
```
User Input (HTML Form)
    ↓
makePrediction() (JavaScript)
    ↓
Generate Synthetic Prediction
    ↓
renderPredictionGauge()
    ↓
renderModelComparison()
```

## Feature Engineering

### Input Features

| Feature | Type | Range | Source |
|---------|------|-------|--------|
| route_id | Categorical | R001-R005 | CSV |
| scheduled_time | Datetime | - | CSV |
| weather | Categorical | sunny, cloudy, rainy, unknown | CSV |
| passenger_count | Numeric | 0-200 | CSV |
| latitude | Float | 23.0-26.0 | CSV |
| longitude | Float | 31.0-34.0 | CSV |

### Engineered Features

| Feature | Derivation | Type |
|---------|-----------|------|
| hour | Extract from scheduled_time | Numeric (0-23) |
| day_of_week | Extract from scheduled_time | Numeric (0-6) |
| time_of_day | Categorize by hour | Categorical |
| is_weekend | day_of_week >= 5 | Binary |
| day_type | Map is_weekend | Categorical |
| weather_severity | Map weather condition | Categorical |
| route_frequency | Count occurrences | Numeric |

### Target Variable

| Variable | Type | Range | Units |
|----------|------|-------|-------|
| delay_minutes | Numeric | -5 to 80+ | Minutes |

## Model Details

### XGBoost Configuration
- **Algorithm**: Gradient Boosting
- **Parameters**: Default sklearn parameters with optimization
- **Strengths**: Non-linear relationships, outlier handling
- **Use Case**: Primary production model

### Random Forest Configuration
- **Algorithm**: Ensemble of Decision Trees
- **Parameters**: 100 trees, max_depth optimized
- **Strengths**: Feature importance, interpretability
- **Use Case**: Verification, explainability

### Linear Regression Configuration
- **Algorithm**: Ordinary Least Squares
- **Parameters**: Standard linear fit
- **Strengths**: Fast, interpretable
- **Use Case**: Baseline comparison

### KNN Configuration
- **Algorithm**: K-Nearest Neighbors
- **Parameters**: k=5 (typical for regression)
- **Strengths**: Local pattern capture
- **Use Case**: Reference model, edge cases

## API Contracts

### Frontend Prediction Request (Streamlit)
```python
{
    "route_id": "R001",
    "scheduled_time": "2025-12-20T08:00:00",
    "hour": 8,
    "weather": "sunny",
    "passenger_count": 50,
    "latitude": 24.5,
    "longitude": 32.5
}
```

### Model Prediction Response
```python
{
    "predicted_delay": 15.3,  # minutes
    "model": "xgboost",
    "confidence": 0.85,  # R² based
    "all_predictions": {
        "xgboost": 15.3,
        "random_forest": 14.8,
        "linear_regression": 16.2,
        "knn": 13.5
    }
}
```

## Performance Considerations

### Optimization Strategies
1. **Model Caching**: Streamlit caches model loading
2. **Feature Scaling**: StandardScaler reduces computation
3. **Label Encoding**: Reduces memory footprint
4. **Client-side Simulation**: Static web reduces server load

### Scalability Notes
- Current setup handles ~1000 predictions/minute
- For production: Consider async queuing with Celery
- Database integration recommended for large datasets
- Model versioning system needed for A/B testing

## Deployment Architecture

### Development
```
localhost:8000 (Static Web)
localhost:8501 (Streamlit)
```

### Production Recommendations
```
├─ Docker Container (Models + Backend)
├─ CDN for Static Assets
├─ Database (PostgreSQL for historical data)
├─ API Gateway (FastAPI/Flask)
├─ Load Balancer
└─ Monitoring (Prometheus/Grafana)
```

## Security Considerations

- ✓ No sensitive user data collection
- ✓ Models are open-source algorithms
- ✓ Local-only operations (no cloud dependencies)
- ⚠ HTTPS recommended for production
- ⚠ Input validation on all forms
- ⚠ Rate limiting on predictions

---

**Last Updated**: December 2025
