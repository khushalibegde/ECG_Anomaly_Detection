# ECG Anomaly Detection Model Architecture

## Overview
This model uses a hybrid approach combining classic machine learning stacking with deep learning morphology analysis for robust binary ECG beat classification (Normal vs Abnormal).

## Problem Framing

### Binary Classification Task
- **Normal (0)**: MIT-BIH symbols `{'N','L','R','e','j'}`
- **Abnormal (1)**: All other symbols
- **Rationale**: Simplifies to anomaly detection task common in research applications

### Data Structure
- **X_signal**: Raw beat waveform samples (200 samples per beat)
- **X_tabular**: Engineered statistical and spectral features
- **y_binary**: Mapped binary labels

## Architecture Components

### 1. Classic ML Stacking Pipeline

#### Base Models
```
RandomForest → Out-of-fold predictions
XGBoost      → Out-of-fold predictions  
LightGBM     → Out-of-fold predictions
```

#### Meta-Learner
- **LogisticRegression** trained on base model probabilities
- Uses StratifiedKFold cross-validation for out-of-fold training
- Prevents training-on-test leakage

#### Key Features
- Version-safe XGBoost implementation with fallback mechanisms
- Full model retraining after OOF generation for inference
- Robust to hyperparameter variations

### 2. Hybrid Deep Learning Model

#### Signal Processing Branch
```
Input (200 samples) 
    ↓
Conv1D(32, kernel=7) → BatchNorm → MaxPool
    ↓
Conv1D(64, kernel=5) → BatchNorm → MaxPool
    ↓
Bidirectional LSTM(64)
    ↓
Dense(128) → Dropout
```

#### Tabular Features Branch
```
Engineered Features
    ↓
Dense(64) → BatchNorm → Dropout
    ↓
Dense(32)
```

#### Fusion Layer
```
Signal Embedding + Tabular Embedding
    ↓
Concatenate → Dense(128) → Dropout
    ↓
Dense(1, sigmoid) → Binary Classification
```

### 3. Final Ensemble

#### Probability Combination
```python
final_probs = W_META * meta_probs + W_HYBRID * hybrid_probs
# Default weights: W_META = 0.5, W_HYBRID = 0.5
```

#### Decision Threshold
- Probability ≥ 0.5 → Abnormal
- Probability < 0.5 → Normal

## Training Configuration

### Preprocessing
- **StandardScaler** on tabular features (zero-mean, unit-variance)
- **Stratified splitting** to maintain class proportions
- **Class weights** computed to handle imbalance

### Neural Network Training
- **Loss**: Binary crossentropy
- **Callbacks**: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
- **Validation split**: 12% of training data
- **Class weighting**: Automatic balancing for imbalanced classes

### Cross-Validation
- **StratifiedKFold** for robust out-of-fold predictions
- **Bagging** of test probabilities across folds

## Model Artifacts

### Saved Components
```
scaler_tab.pkl             # Tabular feature scaler
rf_full.pkl                # Random Forest model
xgb_full.pkl               # XGBoost model  
lgb_full.txt               # LightGBM model
meta_lr.pkl                # Meta-learner (Logistic Regression)
hybrid_best_binary.h5      # Best neural network checkpoint
hybrid_final_binary.h5     # Final neural network model
label_mapping.json         # Class mapping configuration
artifacts.json             # Model manifest
```

## Key Strengths

### Complementary Learning
- **Tree ensembles**: Excel at tabular engineered features
- **Deep networks**: Learn morphological patterns from raw signals
- **Stacking**: Reduces variance and captures model strengths

### Robustness Features
- Out-of-fold training prevents overfitting
- Early stopping and callbacks prevent overtraining
- Class weighting handles imbalanced data
- Ensemble averaging increases stability