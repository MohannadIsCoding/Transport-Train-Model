# Installation and Setup Guide

## System Requirements

- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 500MB for dependencies and models
- **Browser**: Modern browser (Chrome, Firefox, Safari, Edge)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/transport-delay-predictor.git
cd "Transport Train Model"
```

### 2. Create Virtual Environment

#### Windows (PowerShell/Command Prompt)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux (Bash/Zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verification**:
```bash
python --version  # Should show Python 3.8+
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Expected Installation Time**: 3-5 minutes

**Verify Installation**:
```bash
python -c "import pandas, sklearn, xgboost, streamlit; print('✓ All dependencies installed')"
```

### 4. Prepare Data

Ensure `cleaned_transport_dataset.csv` is in the project root:

```
Transport Train Model/
├── cleaned_transport_dataset.csv  ← Place here
├── app.py
├── train_models.py
└── ...
```

### 5. Train Models (First Time)

```bash
python train_models.py
```

**Expected Output**:
```
Loading cleaned dataset...
Dataset shape: (500, 12)
Training Linear Regression...
  Test MAE: 62.53, R²: 0.18
Training Random Forest...
  Test MAE: 56.29, R²: 0.43
Training XGBoost...
  Test MAE: 56.29, R²: 0.42
Training kNN Regression...
  Test MAE: 67.72, R²: -0.04
Models saved successfully!
Metadata updated: models/metadata.json
```

**Troubleshooting**:

| Error | Solution |
|-------|----------|
| `FileNotFoundError: cleaned_transport_dataset.csv` | Download dataset and place in project root |
| `ImportError: No module named sklearn` | Run `pip install scikit-learn` |
| `Memory Error` | Close other applications, reduce dataset size |

## Quick Start

### Option 1: Static Web Frontend (Fastest)

```bash
python -m http.server 8000
```

Then open browser:
```
http://localhost:8000
```

**Features**: Dashboard, predictions, analysis (simulated predictions)

### Option 2: Streamlit Dashboard (Full Features)

```bash
streamlit run app.py
```

Then browser opens automatically to:
```
http://localhost:8501
```

**Features**: Real predictions, advanced analytics, model comparison

### Option 3: Jupyter Notebook

```bash
jupyter notebook transport_delay_analysis.ipynb
```

**Features**: Interactive analysis, cell-by-cell execution

## Detailed Setup by Platform

### Windows Setup

```powershell
# 1. Clone repo
git clone https://github.com/yourusername/transport-delay-predictor.git
cd "Transport Train Model"

# 2. Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models
python train_models.py

# 5. Run application
python -m http.server 8000

# Open http://localhost:8000 in browser
```

### macOS Setup

```bash
# 1. Clone repo
git clone https://github.com/yourusername/transport-delay-predictor.git
cd "Transport Train Model"

# 2. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models
python3 train_models.py

# 5. Run application
python3 -m http.server 8000

# Open http://localhost:8000 in browser
```

### Linux Setup

```bash
# 1. Clone repo
git clone https://github.com/yourusername/transport-delay-predictor.git
cd "Transport Train Model"

# 2. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (may need build tools)
sudo apt-get install python3-dev build-essential
pip install -r requirements.txt

# 4. Train models
python3 train_models.py

# 5. Run application
python3 -m http.server 8000

# Open http://localhost:8000 in browser
```

## Configuration

### Environment Variables (Optional)

Create `.env` file:
```env
FLASK_ENV=development
DEBUG=True
DATA_PATH=./cleaned_transport_dataset.csv
MODEL_PATH=./models
```

### Port Configuration

**Change HTTP Server Port**:
```bash
python -m http.server 9000  # Instead of 8000
```

**Change Streamlit Port**:
```bash
streamlit run app.py --server.port 8502
```

### Model Configuration

Edit `train_models.py` to customize:

```python
# Random Forest parameters
rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Tree depth
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

# KNN parameters
knn_model = KNeighborsRegressor(
    n_neighbors=5,         # K value
    weights='distance'
)
```

## Verification Steps

### 1. Verify Installation
```bash
python -c "import pandas, sklearn, xgboost; print('✓ OK')"
```

### 2. Verify Data
```bash
python -c "import pandas as pd; df = pd.read_csv('cleaned_transport_dataset.csv'); print(f'Rows: {len(df)}, Columns: {len(df.columns)}')"
```

### 3. Verify Models
```bash
import joblib
models = joblib.load('models/xgboost.pkl')
print('✓ Models loaded successfully')
```

### 4. Test Frontend
Navigate to http://localhost:8000 and:
- [ ] Dashboard loads with data
- [ ] Can input values in prediction form
- [ ] Receives predictions for all models
- [ ] Charts render without errors

## Troubleshooting

### Issue: Models not found
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'models/xgboost.pkl'`

**Solution**:
```bash
# Retrain models
python train_models.py

# Verify models were created
dir models/  # Windows
ls models/   # macOS/Linux
```

### Issue: CSV encoding error
**Error**: `UnicodeDecodeError: 'utf-8' codec can't decode byte...`

**Solution**:
```bash
# Verify CSV encoding
file cleaned_transport_dataset.csv  # Shows encoding

# Convert if needed (Windows)
iconv -f ISO-8859-1 -t UTF-8 dirty_transport_dataset.csv -o cleaned_transport_dataset.csv
```

### Issue: Port already in use
**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 8000
# Windows
netstat -ano | findstr :8000

# macOS/Linux
lsof -i :8000

# Kill process and use different port
python -m http.server 9000
```

### Issue: Import errors
**Error**: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify virtual environment is activated
which python  # Should show path inside .venv/
```

### Issue: Low performance
**Solution**:
```bash
# Use more CPU cores
# Edit train_models.py:
# Set n_jobs=-1 in RandomForestRegressor
# Set nthread=-1 in XGBoostRegressor

# Or run with PyPy for speed:
pip install pypy3
pypy3 train_models.py
```

## Development Workflow

### Making Code Changes

1. **Activate virtual environment**:
   ```bash
   .\.venv\Scripts\Activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

2. **Edit code** (IDE/Editor):
   ```
   app.py
   train_models.py
   app.js
   etc.
   ```

3. **Test changes**:
   ```bash
   streamlit run app.py
   # or
   python -m http.server 8000
   ```

4. **Commit to git**:
   ```bash
   git add .
   git commit -m "Your message"
   git push
   ```

### Updating Dependencies

```bash
# Check for updates
pip list --outdated

# Update specific package
pip install --upgrade pandas

# Update all
pip install --upgrade -r requirements.txt

# Save new versions
pip freeze > requirements.txt
```

## Performance Optimization

### Speed Up Training

```bash
# Use GPU acceleration (if available)
pip install xgboost-gpu

# Then in train_models.py:
xgb_model = xgb.XGBRegressor(tree_method='gpu_hist')
```

### Reduce Memory Usage

```bash
# Use data types efficiently
import pandas as pd
df = df.astype({'passenger_count': 'int16', 'delay_minutes': 'float32'})
```

### Parallel Processing

```bash
# Already configured in train_models.py
# Random Forest uses n_jobs=-1 (all cores)
# Streamlit uses @st.cache for caching
```

## Next Steps

1. ✓ Installation complete
2. → Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. → Check [README.md](README.md) for feature overview
4. → Explore UI at http://localhost:8000 or http://localhost:8501
5. → Read model training code in `train_models.py`

---

**Last Updated**: December 2025  
**Tested with**: Python 3.8+, Windows 10/11, macOS 11+, Ubuntu 20.04+
