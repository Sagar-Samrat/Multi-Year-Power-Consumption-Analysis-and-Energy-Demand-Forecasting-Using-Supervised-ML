import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def load_data(filepath):
    """Step 1: Read raw CSV dataset into a Pandas DataFrame."""
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Step 2: Clean the data.
    - Format 'datetime' correctly and remove duplicates.
    - Interpolate (fill) missing weather data.
    - Cap extreme outliers in Power Demand using the IQR method.
    """
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').drop_duplicates(subset=['datetime']).set_index('datetime')
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    # Handle missing values
    df = df.dropna(subset=['Power demand'])
    df = df.interpolate(method='time').bfill().ffill()  
    
    # Cap Outliers (IQR method guarantees models won't skew on extreme erroneous spans)
    Q1 = df['Power demand'].quantile(0.25)
    Q3 = df['Power demand'].quantile(0.75)
    IQR = Q3 - Q1
    df['Power demand'] = np.clip(df['Power demand'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    
    return df

def engineer_features(df):
    """
    Step 3: Create AI Input Features.
    - Temporal Features: Break Date into hour, day, month (finds seasonal/daily cycles).
    - Lag Features: Look at the exact demand from 2-hours ago and 24-hours ago.
    """
    # Temporal Features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekday'] = df.index.weekday
    
    # Lag features (Past History) represented across 5-min intervals
    df['lag_24'] = df['Power demand'].shift(24) 
    df['lag_288'] = df['Power demand'].shift(288)
    df['rolling_mean_12'] = df['Power demand'].shift(24).rolling(window=12).mean()
    
    return df.dropna()

def get_train_val_test_split(df):
    """
    Step 4: Chronological Split & Standard Scaling. 
    - 70% Train (Past), 15% Val (Mid), 15% Test (Future).
    - We must NOT randomize (shuffle) in time series to prevent data leakage!
    """
    n = len(df)
    train_df = df.iloc[:int(n*0.70)]
    val_df   = df.iloc[int(n*0.70):int(n*0.85)]
    test_df  = df.iloc[int(n*0.85):]
    
    features = ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 'lag_24', 'lag_288', 'rolling_mean_12']
    features = [f for f in features if f in df.columns]
    target = 'Power demand'
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val     = val_df[features], val_df[target]
    X_test, y_test   = test_df[features], test_df[target]
    
    # Scale Data: Mean=0, Std=1 (Required for models like K-NN and Linear Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    
    # Save the scaler so the UI App can format new inputs accurately
    joblib.dump(scaler, 'scaler.pkl') 
    
    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, features

def evaluate_model(name, model, X_test, y_test):
    """Evaluates prediction error based on MAE, RMSE, and R2 metrics."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    return rmse, r2, mae

def build_models(X_train, y_train, X_val, y_val):
    """
    Step 5: Train 6 Different Models to find the best algorithmic fit.
    The model with the lowest Root Mean Squared Error (RMSE) wins.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        "K-Neighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    best_model, best_name, best_rmse = None, "", float('inf')
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        rmse, r2, mae = evaluate_model(name, model, X_val, y_val)
        results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}
        
        # Track the minimum RMSE score to save the best model
        if rmse < best_rmse:
            best_rmse, best_model, best_name = rmse, model, name
            
    # Serialize the best winning model
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(results, 'model_results.pkl')
    
    return best_model, best_name

def generate_viva_eda_graphs(df, model, model_name, X_test, y_test, features):
    """
    Step 6: Generate logical, highly clear EDA graphs for Viva/Interviews.
    Instead of noisy visualizations, these cleanly answer:
    1) Why does the model make its decisions?
    2) Does weather actually matter?
    3) Did the model predict correctly?
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    
    print("Generating clean, logical Viva explanation graphs...")
    
    # 1. Feature Importance (Answers: "What was the logic?")
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices], color='teal')
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
        plt.title("Logic Proof: What Features Drive the Predictions?", fontsize=15, fontweight='bold')
        plt.ylabel("Importance Level", fontsize=12)
        plt.tight_layout()
        plt.savefig('viva_1_logic_feature_importance.png', dpi=300)
        plt.close()
        
    # 2. Weather Impact (Answers: "How does nature affect our grid?")
    plt.figure(figsize=(8, 5))
    # Sample down to 3000 points to make the trendline visually meaningful instead of a massive blob
    sample = df.sample(min(len(df), 3000), random_state=42) 
    sns.regplot(x='temp', y='Power demand', data=sample, 
                scatter_kws={'alpha':0.3, 'color': '#1f77b4'}, 
                line_kws={'color':'red', 'linewidth': 3})
    plt.title("Logic Proof: Effect of Real-world Temperature on Demand", fontsize=15, fontweight='bold')
    plt.xlabel("Temperature (°C)", fontsize=12)
    plt.ylabel("Power Demand (MW)", fontsize=12)
    plt.tight_layout()
    plt.savefig('viva_2_logic_weather_impact.png', dpi=300)
    plt.close()
    
    # 3. Accuracy Overlap (Answers: "How accurate is it visually?")
    plt.figure(figsize=(12, 5))
    preds = model.predict(X_test)
    # Just show a tiny sliver of 500 records (~40 hours) so reviewers can see the overlap perfectly
    plt.plot(y_test.values[:500], label='Actual Recorded Data', color='#2ca02c', linewidth=2, alpha=0.8)
    plt.plot(preds[:500], label=f'AI Prediction ({model_name})', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.title("Logic Proof: Real vs Predicted Timelines exactly matching", fontsize=15, fontweight='bold')
    plt.xlabel("Time Instances (Every 5 minutes for 40 hours)", fontsize=12)
    plt.ylabel("Demand (MW)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('viva_3_logic_accuracy_overlap.png', dpi=300)
    plt.close()
    
    print("✅ Logical EDA Viva Graphs successfully saved as PNG files!")

def precompute_dashboard_data(df, model):
    """
    Step 7: Precompute heavy dashboard datasets to prevent memory-limit 503 errors on hosting platforms.
    """
    print("Precomputing dashboard summary files...")
    # 1. Seasonal patterns
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonal_data = df.groupby(['year', 'month'])['Power demand'].agg(['mean', 'min', 'max']).reset_index()
    seasonal_data.rename(columns={'year': 'Year', 'month': 'Month'}, inplace=True)
    seasonal_data['Year'] = seasonal_data['Year'].astype(str)
    seasonal_data['Month_Name'] = seasonal_data['Month'].map(lambda x: month_names[x-1])
    seasonal_data.to_csv("seasonal_data_precomputed.csv", index=False)
    print("✅ Precomputed seasonal patterns saved to 'seasonal_data_precomputed.csv'.")
    
    # Load scaler
    scaler = joblib.load('scaler.pkl')
    
    # 2. Yearly actual vs predicted comparison
    feature_cols = ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 'lag_24', 'lag_288', 'rolling_mean_12']
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    
    df_result = pd.DataFrame({
        'Power demand': df['Power demand'],
        'Predicted demand': preds
    }, index=df.index)
    
    df_monthly = df_result.resample('MS').mean()
    df_monthly['Year'] = df_monthly.index.year
    df_monthly['Month'] = df_monthly.index.month
    df_monthly['Month_Num'] = df_monthly.index.strftime('%b')
    df_monthly.to_csv("yearly_monthly_comparison.csv")
    print("✅ Precomputed predictions comparison saved to 'yearly_monthly_comparison.csv'.")

def main():
    """Main execution function triggered when the script runs."""
    filepath = "/Users/sagarsamrat/Downloads/powerdemand_5min_2021_to_2024_with weather.csv"
    if not os.path.exists(filepath):
        # Fallback to local path if Downloads path doesn't exist
        filepath = "powerdemand_5min_2021_to_2024_with weather.csv"
        if not os.path.exists(filepath):
            print("Data file not found. Please check filepath.")
            return
    
    # 1. Pipeline Execution
    print("Executing ML Pipeline...")
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    
    # 2. Export clean data for streamlit scatterplots 
    df.tail(2000).to_csv("cleaned_data_sample.csv") 
    
    # 3. Model Training
    X_train, y_train, X_val, y_val, X_test, y_test, features = get_train_val_test_split(df)
    best_model, best_name = build_models(X_train, y_train, X_val, y_val)
    
    # 4. Logical EDA Graph generation for Viva Presentation
    generate_viva_eda_graphs(df, best_model, best_name, X_test, y_test, features)
    
    # 5. Precompute Dashboard Datasets to avoid OOM in Streamlit
    precompute_dashboard_data(df, best_model)
    
    print(f"✅ Pipeline Completed! The winning model was: {best_name}")

if __name__ == "__main__":
    main()
