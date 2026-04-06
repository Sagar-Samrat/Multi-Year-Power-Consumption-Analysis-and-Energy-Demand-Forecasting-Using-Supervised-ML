# 🎯 ML PIPELINE - QUICK REFERENCE CHEATSHEET

## Pipeline Workflow (7 Steps)

```
┌─────────────────────────────────────────────────────────────────────┐
│                  DATA LOADING                                       │
│         Load CSV: 1.3M rows × 8 columns                             │
│  Power demand + Weather (temp, humidity, wind)                      │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  DATA CLEANING                                      │
│  ✓ Datetime processing   ✓ Remove duplicates                        │
│  ✓ Missing value handling ✓ Outlier removal (IQR method)           │
│         Result: Clean, ready-to-use data                           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                                    │
│  ✓ Temporal: hour, day, month, weekday (time patterns)             │
│  ✓ Lag: lag_24 (2 hrs ago), lag_288 (24 hrs ago)                   │
│  ✓ Rolling: 1-hour moving average                                   │
│         Total: 10 features for prediction                          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│          DATA SPLITTING & STANDARDIZATION                           │
│  Split: 70% Train | 15% Validation | 15% Test                      │
│  Scale: StandardScaler (mean=0, std=1)                             │
│         Result: Ready for model training                           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODEL TRAINING (6 MODELS)                              │
│  1. Linear Regression    4. Gradient Boosting                       │
│  2. Ridge Regression     5. K-Nearest Neighbors                     │
│  3. Random Forest        6. XGBoost                                 │
│                                                                      │
│  Evaluate on: Validation set (RMSE metric)                         │
│  Select: Model with lowest RMSE                                    │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│           FINAL EVALUATION ON TEST SET                              │
│  Metrics: MAE, RMSE, R²                                             │
│  Purpose: Unbiased performance on never-seen data                  │
│  Result: True model accuracy reported                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7 Key Steps Explained in 1 Sentence Each

| Step | What | Output |
|------|------|--------|
| 1️⃣ **Load** | Read CSV file | Raw DataFrame |
| 2️⃣ **Clean** | Remove bad data, outliers | Clean DataFrame |
| 3️⃣ **Engineer** | Create 10 features | Features + Target |
| 4️⃣ **Split** | 70-15-15 train-val-test | Scaled datasets |
| 5️⃣ **Train** | Train 6 models | 6 trained models |
| 6️⃣ **Select** | Choose best on validation | Best model |
| 7️⃣ **Evaluate** | Test on unseen data | Final accuracy |

---

## 6 Models at a Glance

```
┌──────────────────────┬────────────────┬──────────┬──────────┐
│ Model                │ Type           │ Speed    │ Accuracy │
├──────────────────────┼────────────────┼──────────┼──────────┤
│ Linear Regression    │ Linear         │ ⚡ Fast  │ ⭐ Low   │
│ Ridge Regression     │ Linear+Reg     │ ⚡ Fast  │ ⭐ Low   │
│ Random Forest        │ Tree Ensemble  │ 🔷 Mid   │ ⭐⭐⭐   │
│ Gradient Boosting    │ Sequential     │ 🔶 Slow  │ ⭐⭐⭐⭐  │
│ K-Nearest Neighbors  │ Instance-based │ 🔴 Slow  │ ⭐⭐    │
│ XGBoost              │ Opt. Gradient  │ 🔶 Slow  │ ⭐⭐⭐⭐⭐ │
└──────────────────────┴────────────────┴──────────┴──────────┘

Best for accuracy: XGBoost
Best for speed: Linear/Ridge
Best balanced: Random Forest
```

---

## 10 Features Explained

### Temporal Features (Time Patterns)
```
🕐 hour       (0-23)      What time of day?
📅 day        (1-31)      Which day of month?
🗓️ month      (1-12)      Which season?
📊 weekday    (0-6)       Weekday or weekend?
```

### Lag Features (Historical)
```
⏳ lag_24     (2 hrs ago)     Recent trend
📜 lag_288    (24 hrs ago)    Yesterday's pattern
```

### Rolling Features
```
📈 rolling_mean_12   1-hour moving average
```

### Weather Features
```
🌡️  temp       Temperature
💧 rhum       Humidity
💨 wspd       Wind speed
```

---

## 3 Evaluation Metrics

```
MAE (Mean Absolute Error)
├─ Formula: Average of |Actual - Predicted|
├─ Units: MW (same as power demand)
├─ Interpretation: "Average error is X MW"
└─ Good for: Understanding real-world error magnitude

RMSE (Root Mean Squared Error)
├─ Formula: √(Average of (Actual - Predicted)²)
├─ Units: MW
├─ Interpretation: "Typically off by X MW"
└─ Good for: When large errors are very costly

R² Score (Coefficient of Determination)
├─ Range: 0 to 1 (higher is better)
├─ Interpretation: "Model explains X% of variation"
├─ 0.8+ = Excellent, 0.6+ = Good, <0.4 = Poor
└─ Good for: Overall model quality
```

---

## Data Cleaning Summary

```
Step 1: Convert datetime
        "2021-01-01 00:00:00" ✓

Step 2: Remove duplicates
        Same timestamp twice? ❌

Step 3: Remove NaNs in target
        No power demand value? ❌

Step 4: Interpolate weather data
        Missing temperature? → Fill with adjacent values

Step 5: Remove outliers (IQR method)
        Extreme values (sensor errors)? → Cap them

Result: ~900,000 clean samples ✓
```

---

## Data Split Visualization

```
Timeline: 2021 ─────────────────────────────────────────► 2024

Training (70%)          Validation (15%)    Test (15%)
├──────────────────────┼──────────────────┤──────────────┤
2021        2022       2023       2023.6  2024
│                      │                  │
└─ Learn patterns ─────┴─ Select best ────┴─ Final test

Why chronological split?
• Real scenario: predict future from past
• Random split = uses future to predict past (cheating!)
• Time series requires temporal order
```

---

## Why Each Step Matters

| Step | Why Important | Consequence of Skipping |
|------|---------------|------------------------|
| Cleaning | Bad data = bad predictions | 50% accuracy drop |
| Features | Model learns from features | Can't capture patterns |
| Splitting | Test true generalization | Overfitting looks perfect |
| Scaling | Fair feature comparison | Some models fail |
| Training multiple | Find best approach | Suboptimal model chosen |
| Validation set | Select model wisely | Pick wrong winner |
| Test set | Unbiased evaluation | Claimed accuracy wrong |

---

## Important Concepts

### Data Leakage ❌
```
WRONG: Use lag_0 (current demand) to predict current
CORRECT: Use lag_24, lag_288 (past values only)
```

### Overfitting ❌
```
Training error: 10 MW (memorized)
Validation error: 500 MW (doesn't generalize)
Solution: Simpler models, regularization, validation set
```

### Underfitting ❌
```
Too simple model for complex pattern
Solution: More complex model (Random Forest, XGBoost)
```

### Goldilocks Zone ✅
```
Balance between:
- Simple enough to generalize (not overfit)
- Complex enough to capture patterns (not underfit)
```

---

## Output Files Generated

```
best_model.pkl
├─ Trained best model
├─ Used for predictions on new data
└─ Don't retrain every time!

scaler.pkl
├─ Feature scaling parameters
├─ Must use same scaler for new data
└─ StandardScaler (mean=0, std=1)

model_results.pkl
├─ All 6 models' metrics
├─ MAE, RMSE, R² for comparison
└─ Used by Streamlit app for charts

cleaned_data_sample.csv
├─ Last 2,000 rows of cleaned data
├─ Used for visualization in web app
└─ Shows trends and patterns
```

---

## How to Present Each Step

### Step 1: Loading
> "This dataset has 4 years of power demand measurements with weather data - over 1 million data points for training."

### Step 2: Cleaning
> "We handle missing values, remove sensor errors, and fix data quality issues - about 15% of raw data was problematic."

### Step 3: Features
> "We create meaningful features that capture time patterns (hour, day, season), historical influence (what happened 2/24 hours ago), and weather effects."

### Step 4: Splitting
> "We split chronologically: train on past, validate on more recent past, test on newest data - this simulates real prediction scenario."

### Step 5: Training
> "We train 6 different algorithms to find which works best for power demand prediction."

### Step 6: Selection
> "Based on validation set performance, [Best Model] wins with [RMSE] MW average error."

### Step 7: Testing
> "We verify this choice on completely new test data, confirming [accuracy]% effectiveness."

---

## Performance Expectations

```
Excellent (Reserve Prediction): RMSE < 50 MW
Good (for planning): RMSE 50-100 MW
Acceptable (rough estimates): RMSE 100-200 MW
Poor: RMSE > 200 MW
```

---

## For Different Audiences

### Technical Team:
- Discuss algorithms (ensemble methods vs linear)
- Explain hyperparameters
- Show cross-validation strategy
- Discuss feature importance

### Management:
- Focus on ROI ("save $X per week")
- Show accuracy percentages
- Explain timeline
- Discuss implementation

### Operations Team:
- Show how to use predictions
- Explain confidence intervals
- Discuss error handling
- Plan for integration

---

**Print or screenshot this for quick reference during presentations!** 📋
