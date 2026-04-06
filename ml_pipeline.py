"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   POWER DEMAND FORECASTING - COMPLETE ML PIPELINE                          ║
║   ═══════════════════════════════════════════════════════════════           ║
║                                                                              ║
║   This pipeline automates the entire machine learning workflow:             ║
║   1. Data Loading      → Read raw CSV dataset                               ║
║   2. Data Cleaning     → Handle missing values, duplicates, outliers        ║
║   3. Feature Engineering → Create temporal and lag features                 ║
║   4. Data Splitting    → Train/Validation/Test split (70/15/15)           ║
║   5. Model Training    → Train 6 different ML models                        ║
║   6. Model Selection   → Compare and select the best model                  ║
║   7. Evaluation        → Measure performance (RMSE, MAE, R²)               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd                                      # Data manipulation library
import numpy as np                                       # Numerical computing
from sklearn.model_selection import TimeSeriesSplit     # Time series splitting
from sklearn.preprocessing import StandardScaler        # Feature scaling/normalization
from sklearn.linear_model import LinearRegression       # Linear regression model
from sklearn.linear_model import Ridge                  # Ridge regression (L2 regularization)
from sklearn.ensemble import RandomForestRegressor      # Random Forest model
from sklearn.ensemble import GradientBoostingRegressor  # Gradient Boosting model
from sklearn.neighbors import KNeighborsRegressor       # K-Nearest Neighbors model
from sklearn.metrics import (mean_absolute_error,      # MAE metric
                            mean_squared_error,         # MSE metric
                            r2_score)                   # R² score metric
import xgboost as xgb                                   # XGBoost model
import joblib                                           # Model serialization
import os                                               # Operating system utilities

print("✅ All libraries imported successfully!")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA LOADING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(filepath):
    """
    Load raw dataset from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the dataset
        
    Returns:
    --------
    pandas.DataFrame
        Raw dataframe with all columns from the CSV file
        
    What it does:
    - Reads the CSV file using pandas
    - Displays basic info about the loaded data
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING RAW DATA")
    print("="*80)
    print(f"📁 Loading data from: {filepath}")
    
    df = pd.read_csv(filepath)
    
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   📋 Columns: {list(df.columns)}")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA CLEANING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def clean_data(df):
    """
    Clean and preprocess the raw dataset
    
    Cleaning Steps:
    1. Convert 'datetime' column to proper datetime format
    2. Sort by datetime to maintain chronological order
    3. Remove duplicate records (same datetime)
    4. Remove unnecessary columns (e.g., unnamed index)
    5. Handle missing values in target variable
    6. Interpolate missing weather data
    7. Remove outliers using Interquartile Range (IQR) method
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe ready for feature engineering
    """
    print("\n" + "="*80)
    print("STEP 2: DATA CLEANING & PREPROCESSING")
    print("="*80)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.1: Handle Datetime Column
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📅 2.1: Processing Datetime Column...")
    df['datetime'] = pd.to_datetime(df['datetime'])  # Convert to datetime format
    df = df.sort_values('datetime')                  # Sort chronologically
    df = df.drop_duplicates(subset=['datetime'])     # Remove duplicate timestamps
    df = df.set_index('datetime')                    # Set datetime as index (important for time series)
    print(f"   ✅ Datetime processed. Range: {df.index.min()} to {df.index.max()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.2: Remove Unnecessary Columns
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🗑️  2.2: Removing Unnecessary Columns...")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        print("   ✅ Removed unnamed index column")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.3: Handle Missing Target Variable (Power Demand)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🎯 2.3: Handling Missing Target Variable...")
    missing_target = df['Power demand'].isna().sum()
    print(f"   ⚠️  Missing values in 'Power demand': {missing_target}")
    df = df.dropna(subset=['Power demand'])  # Drop rows where power demand is missing
    print(f"   ✅ Removed {missing_target} rows with missing power demand")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.4: Interpolate Missing Weather Data
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🌡️  2.4: Handling Missing Weather Variables...")
    missing_before = df.isnull().sum().sum()
    print(f"   ⚠️  Total missing values before interpolation: {missing_before}")
    
    # Time-based interpolation for weather data (fills gaps using adjacent values)
    df = df.interpolate(method='time')  # Linear interpolation based on time
    df = df.bfill().ffill()              # Backward fill, then forward fill any remaining NaNs
    
    missing_after = df.isnull().sum().sum()
    print(f"   ✅ Missing values after interpolation: {missing_after}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2.5: Remove Outliers Using IQR Method
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊 2.5: Detecting and Handling Outliers (IQR Method)...")
    
    # Calculate quartiles and interquartile range (IQR)
    Q1 = df['Power demand'].quantile(0.25)      # 25th percentile (1st quartile)
    Q3 = df['Power demand'].quantile(0.75)      # 75th percentile (3rd quartile)
    IQR = Q3 - Q1                               # Interquartile range
    
    # Define outlier bounds (standard: Q1 - 1.5*IQR to Q3 + 1.5*IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"   📈 Q1 (25th percentile): {Q1:.2f} MW")
    print(f"   📈 Q3 (75th percentile): {Q3:.2f} MW")
    print(f"   📈 IQR: {IQR:.2f} MW")
    print(f"   📈 Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Count outliers before capping
    outliers_before = ((df['Power demand'] < lower_bound) | 
                       (df['Power demand'] > upper_bound)).sum()
    
    # Cap outliers instead of removing them (preserves time series continuity)
    df['Power demand'] = np.clip(df['Power demand'], lower_bound, upper_bound)
    
    print(f"   ✅ Capped {outliers_before} outlier values (preserved time series)")
    
    print("\n" + "─"*80)
    print("Data Cleaning Summary:")
    print(f"   • Final Dataset Size: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   • Missing Values: {df.isnull().sum().sum()}")
    print(f"   • Power Demand Range: {df['Power demand'].min():.2f} - {df['Power demand'].max():.2f} MW")
    print(f"   • Average Demand: {df['Power demand'].mean():.2f} MW")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FEATURE ENGINEERING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """
    Create new features from raw data
    
    Feature Engineering Strategy:
    ============================
    Good predictions depend on meaningful features. We create three types:
    
    1. TEMPORAL FEATURES: Capture time-based patterns
       - hour: Hour of day (0-23) - captures daily cycle
       - day: Day of month (1-31) - captures monthly patterns
       - month: Month of year (1-12) - captures seasonal patterns
       - weekday: Day of week (0=Mon, 6=Sun) - captures weekly patterns
    
    2. LAG FEATURES: Use past demand to predict current demand
       - lag_24: Power demand from 2 hours ago (24 × 5min intervals)
       - lag_288: Power demand from 24 hours ago (288 × 5min intervals)
    
    3. ROLLING FEATURES: Trend and smoothing
       - rolling_mean_12: 1-hour moving average from 2 hours ago
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with engineered features ready for model training
    """
    print("\n" + "="*80)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*80)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.1: Extract Temporal Features from DateTime Index
    # ─────────────────────────────────────────────────────────────────────────
    print("\n⏰ 3.1: Extracting Temporal Features...")
    
    df['hour'] = df.index.hour              # Hour of day (0-23)
    df['day'] = df.index.day                # Day of month (1-31)
    df['month'] = df.index.month            # Month of year (1-12)
    df['year'] = df.index.year              # Year (2021-2024)
    df['weekday'] = df.index.weekday        # Day of week (0=Monday, 6=Sunday)
    
    print("   Temporal Features Created:")
    print("   • hour (0-23): Captures daily demand cycle")
    print("   • day (1-31): Captures monthly variations")
    print("   • month (1-12): Captures seasonal patterns")
    print("   • weekday (0-6): Captures weekend vs weekday patterns")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.2: Create Lag Features (Historical Dependencies)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊 3.2: Creating Lag Features...")
    print("   Why lags? Power demand has temporal dependencies.")
    print("   Past demand helps predict current demand!")
    
    # Note about lag terminology:
    # Dataset has 5-minute intervals, so:
    # - 24 intervals = 2 hours (120 minutes)
    # - 288 intervals = 24 hours (1440 minutes)
    
    df['lag_24'] = df['Power demand'].shift(24)     # Demand from 2 hours ago
    print("   • lag_24: Power demand from 2 hours ago (shift by 24 × 5min intervals)")
    
    df['lag_288'] = df['Power demand'].shift(288)   # Demand from 24 hours ago
    print("   • lag_288: Power demand from 24 hours ago (shift by 288 × 5min intervals)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.3: Create Rolling Average Feature
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📈 3.3: Creating Rolling Average Feature...")
    
    # Rolling mean: 1-hour average (12 intervals of 5 minutes each)
    # Shifted by 24 intervals (2 hours) to avoid data leakage
    df['rolling_mean_12'] = df['Power demand'].shift(24).rolling(window=12).mean()
    print("   • rolling_mean_12: 1-hour moving average from 2 hours ago")
    print("     (12 windows × 5 minutes = 1 hour)")
    print("     (Shifted by 24 intervals = 2 hours back to prevent data leakage)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.4: Handle NaNs Introduced by Lag Features
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🔍 3.4: Handling Missing Values...")
    missing_before = df.isnull().sum().sum()
    
    df = df.dropna()  # Drop rows with NaN values (created by shifting)
    
    missing_after = df.isnull().sum().sum()
    print(f"   • Rows with NaN values removed: {missing_before - missing_after}")
    print(f"   • Final dataset size: {len(df):,} rows")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3.5: Display Feature Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("Feature Engineering Summary:")
    print(f"   📊 Total Features: {len(df.columns)}")
    print(f"   📋 Features: {list(df.columns)}")
    print(f"   🎯 Target: Power demand")
    print("\n   Feature Categories:")
    print("   • Temporal: hour, day, month, weekday")
    print("   • Lag: lag_24, lag_288")
    print("   • Rolling: rolling_mean_12")
    print("   • Weather: temp, rhum, wspd")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DATA SPLITTING FUNCTION (Train/Validation/Test)
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_val_test_split(df):
    """
    Split data into training, validation, and test sets
    
    Data Split Strategy (Time-Series Aware):
    ========================================
    For time-series data, we use CHRONOLOGICAL splitting, NOT random splitting!
    
    Why? Because:
    - Breaking temporal order would leak future information into the past
    - Models must learn from past data and predict future (just like real scenarios)
    
    Distribution:
    - Training Set (70%): Historical data for learning patterns
    - Validation Set (15%): For model selection and tuning
    - Test Set (15%): Final evaluation of chosen model
    
    Feature Standardization:
    - Scales features to have mean=0 and std=1
    - Ensures models treat all features equally
    - Important for distance-based and gradient-based models
    
    Returns:
    --------
    tuple: (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, features)
    """
    print("\n" + "="*80)
    print("STEP 4: DATA SPLITTING & STANDARDIZATION")
    print("="*80)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4.1: Chronological Data Splitting
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📑 4.1: Splitting Data (Time-Series Aware)...")
    
    n = len(df)
    train_end = int(n * 0.70)      # 70% for training
    val_end = int(n * 0.85)        # Next 15% for validation
    # Remaining 15% for testing
    
    train_df = df.iloc[:train_end]          # First 70%
    val_df = df.iloc[train_end:val_end]     # Middle 15%
    test_df = df.iloc[val_end:]             # Last 15%
    
    print(f"   ✅ Dataset Split (chronological order):")
    print(f"   • Training Set:   {len(train_df):6,} samples ({len(train_df)/n*100:.1f}%)")
    print(f"   • Validation Set: {len(val_df):6,} samples ({len(val_df)/n*100:.1f}%)")
    print(f"   • Test Set:       {len(test_df):6,} samples ({len(test_df)/n*100:.1f}%)")
    print(f"   • Total:          {n:6,} samples (100.0%)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4.2: Define Features and Target
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🎯 4.2: Defining Features and Target Variable...")
    
    # All features the model will use for prediction
    features = ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 
                'lag_24', 'lag_288', 'rolling_mean_12']
    target = 'Power demand'
    
    # Filter features that exist in the dataset
    features = [f for f in features if f in df.columns]
    
    print(f"   📊 Input Features ({len(features)}): {', '.join(features)}")
    print(f"   🎯 Target Variable: {target}")
    
    # Extract features and target for each split
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    print(f"   ✅ Feature matrices created:")
    print(f"   • X_train shape: {X_train.shape} (rows, features)")
    print(f"   • X_val shape:   {X_val.shape}")
    print(f"   • X_test shape:  {X_test.shape}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4.3: Feature Standardization (Scaling)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊 4.3: Feature Standardization (Scaling)...")
    print("   Why scale? Different features have different ranges:")
    print(f"   • Temperature: typically -10 to 50°C")
    print(f"   • Hour: 0 to 23")
    print(f"   • Power demand: 1000 to 5000 MW")
    print("   Scaling brings all to same range (mean=0, std=1)")
    
    scaler = StandardScaler()
    
    # Fit scaler on training data ONLY (prevent data leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply same scaler to validation and test sets
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("   ✅ Scaling complete:")
    print(f"   • Scaler fitted on training set")
    print(f"   • Training set scaled: mean={X_train_scaled.mean():.3f}, std={X_train_scaled.std():.3f}")
    print(f"   • Validation and test sets scaled with same scaler")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4.4: Save Scaler for Later Use
    # ─────────────────────────────────────────────────────────────────────────
    print("\n💾 4.4: Saving Scaler for Future Predictions...")
    
    joblib.dump(scaler, 'scaler.pkl')  # Save for inference/predictions
    print("   ✅ Scaler saved as 'scaler.pkl'")
    print("   (Used later to scale new prediction inputs)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4.5: Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("Data Splitting Summary:")
    print(f"   • Features used: {len(features)}")
    print(f"   • Scaling method: StandardScaler (Z-score normalization)")
    print(f"   • Train/Val/Test ratio: 70/15/15 (chronological)")
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, features


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MODEL EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate model performance using three key metrics
    
    Metrics Explained:
    ==================
    1. MAE (Mean Absolute Error)
       - Average of all prediction errors (absolute values)
       - Units: Same as target (MW)
       - Interpretation: "On average, predictions are off by X MW"
       - Better for understanding real-world error magnitude
    
    2. RMSE (Root Mean Squared Error)
       - Square root of average squared errors
       - Units: Same as target (MW)
       - Penalizes large errors more than small ones
       - Better when outlier errors are costly
    
    3. R² Score (Coefficient of Determination)
       - Measures what % of variance model explains
       - Range: 0 to 1 (can be negative if really bad)
       - Interpretation: "Model explains R² × 100% of the variation"
       - 0.8+: Excellent, 0.6+: Good, <0.3: Poor
    
    Parameters:
    -----------
    name : str
        Model name for display
    model : sklearn model
        Trained model object
    X_test : ndarray
        Test features
    y_test : series/array
        Test target values
        
    Returns:
    --------
    tuple: (rmse, r2, mae)
    """
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Display results in a formatted table
    print(f"[{name:25s}] → MAE: {mae:8.2f} MW  |  RMSE: {rmse:8.2f} MW  |  R²: {r2:.4f}")
    
    return rmse, r2, mae


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL TRAINING & SELECTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_models(X_train, y_train, X_val, y_val):
    """
    Train 6 different machine learning models and select the best one
    
    Models Comparison:
    ==================
    
    1. LINEAR REGRESSION
       - Assumes linear relationship between features and target
       - Pros: Fast, interpretable, stable
       - Cons: Can't capture complex patterns
       - Best for: Simple relationships
    
    2. RIDGE REGRESSION
       - Linear regression with L2 regularization penalty
       - Prevents overfitting by penalizing large coefficients
       - Pros: Handles multicollinearity better
       - Cons: Still assumes linear relationship
    
    3. RANDOM FOREST
       - Ensemble of decision trees (averages predictions)
       - Each tree learns different parts of the data
       - Pros: Handles non-linear patterns, robust
       - Cons: Slower, harder to interpret
       - Good for: Non-linear problems
    
    4. GRADIENT BOOSTING
       - Sequentially builds trees, each correcting previous errors
       - More powerful than Random Forest
       - Pros: High accuracy, learns complex patterns
       - Cons: Slower, more prone to overfitting
    
    5. K-NEAREST NEIGHBORS
       - Predicts based on K nearest neighbors in training data
       - Pros: Simple, no training needed
       - Cons: Slow predictions, doesn't work well with many features
    
    6. XGBOOST
       - Optimized gradient boosting with regularization
       - Industry standard for many competitions
       - Pros: Very high accuracy, fast
       - Cons: Complex hyperparameters, prone to overfitting if not tuned
    
    Selection Criterion:
    - Best model is selected based on LOWEST validation RMSE
    - RMSE penalizes large errors (good for power demand prediction)
    
    Returns:
    --------
    tuple: (best_model, best_model_name)
    """
    print("\n" + "="*80)
    print("STEP 5: MODEL TRAINING & SELECTION")
    print("="*80)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5.1: Initialize 6 Different Models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🤖 5.1: Initializing 6 Different Models...")
    print("   Models to train:")
    print("   1. Linear Regression      - Simple linear relationship")
    print("   2. Ridge Regression       - Linear + regularization")
    print("   3. Random Forest          - Ensemble of decision trees")
    print("   4. Gradient Boosting      - Sequential tree boosting")
    print("   5. K-Nearest Neighbors    - Instance-based learning")
    print("   6. XGBoost                - Optimized gradient boosting")
    
    models = {
        "Linear Regression":      LinearRegression(),
        "Ridge Regression":       Ridge(alpha=1.0),
        "Random Forest":          RandomForestRegressor(n_estimators=50, max_depth=10, 
                                                       random_state=42, n_jobs=-1),
        "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                                           learning_rate=0.1, random_state=42),
        "K-Neighbors":            KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "XGBoost":                xgb.XGBRegressor(n_estimators=100, max_depth=6, 
                                                  learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5.2: Train Each Model and Evaluate on Validation Set
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("5.2: Training Models on Training Set & Evaluating on Validation Set...")
    print("─"*80)
    
    best_model = None
    best_name = ""
    best_rmse = float('inf')  # Start with worst possible RMSE
    results = {}
    
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"\n   [{i}/6] Training {name}...", end=" ", flush=True)
        
        # Train model on training data
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        rmse, r2, mae = evaluate_model(name, model, X_val, y_val)
        
        # Store results
        results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
        
        # Track best model
        if rmse < best_rmse:
            print("  ✨ NEW BEST!", end="")
            best_rmse = rmse
            best_model = model
            best_name = name
        
        print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5.3: Display Model Comparison Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("5.3: MODEL COMPARISON SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    print("\nValidation Set Performance (sorted by RMSE - lower is better):")
    print(results_df.sort_values('RMSE').to_string())
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5.4: Save Best Model and Results
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("5.4: Saving Best Model and Results...")
    
    joblib.dump(best_model, 'best_model.pkl')      # Save the best trained model
    joblib.dump(results, 'model_results.pkl')      # Save all model metrics
    
    print(f"   ✅ Best model saved: 'best_model.pkl'")
    print(f"   ✅ Results saved: 'model_results.pkl'")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5.5: Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"🏆 BEST MODEL SELECTED: {best_name}")
    print("="*80)
    print(f"   Validation RMSE: {best_rmse:.2f} MW")
    print(f"   Validation R²:   {results[best_name]['R2']:.4f}")
    print(f"   Validation MAE:  {results[best_name]['MAE']:.2f} MW")
    print("\n   This model will be used for final testing and predictions!")
    
    return best_model, best_name


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN EXECUTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Execute the complete machine learning pipeline
    
    Pipeline Workflow:
    ==================
    1. Load Data          → Read from CSV file
    2. Clean Data         → Handle missing values, duplicates, outliers
    3. Engineer Features  → Create temporal, lag, and rolling features
    4. Split Data         → 70% train, 15% validation, 15% test
    5. Train Models       → Train 6 different algorithms
    6. Select Best        → Choose model with lowest validation RMSE
    7. Evaluate         → Test final model on test set
    """
    
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  POWER DEMAND FORECASTING - ML PIPELINE EXECUTION".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Check if Dataset Exists
    # ─────────────────────────────────────────────────────────────────────────
    filepath = "/Users/sagarsamrat/Downloads/powerdemand_5min_2021_to_2024_with weather.csv"
    
    if not os.path.exists(filepath):
        print(f"\n❌ ERROR: Dataset not found!")
        print(f"   Expected location: {filepath}")
        print(f"\n   Please ensure the CSV file exists at the specified path.")
        return
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2-4: Data Processing Pipeline
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("SECTIONS 1-4: DATA LOADING, CLEANING, ENGINEERING, & SPLITTING")
    print("─"*80)
    
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Save Sample for Visualization
    # ─────────────────────────────────────────────────────────────────────────
    print("\n💾 Saving cleaned data sample for web app visualizations...")
    sample_df = df.tail(2000)  # Last 2000 rows for streamlit app
    sample_df.to_csv("cleaned_data_sample.csv")
    print(f"   ✅ Saved {len(sample_df)} rows to 'cleaned_data_sample.csv'")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Split and Standardize Data
    # ─────────────────────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test, features = get_train_val_test_split(df)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Train & Select Best Model
    # ─────────────────────────────────────────────────────────────────────────
    best_model, best_name = build_models(X_train, y_train, X_val, y_val)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Final Evaluation on Test Set
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("STEP 6: FINAL EVALUATION ON TEST SET")
    print("="*80)
    print("\nNote: Test set has NOT been seen by the model during training.")
    print("This evaluates true generalization performance.\n")
    
    evaluate_model(best_name, best_model, X_test, y_test)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📊 Generated Artifacts:")
    print("   • best_model.pkl          - Trained best model (ready for predictions)")
    print("   • scaler.pkl              - Feature scaler (for normalizing new data)")
    print("   • model_results.pkl       - All models' performance metrics")
    print("   • cleaned_data_sample.csv - Sample data for visualizations")
    print("\n🚀 Next Steps:")
    print("   1. Use 'best_model.pkl' for making predictions on new data")
    print("   2. Visit Streamlit app to see visualizations and results")
    print("   3. Run Jupyter notebook for detailed analysis")
    print("\n" + "="*80)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PROGRAM ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Entry point: This code only runs if the script is executed directly
    (not imported as a module)
    """
    main()
