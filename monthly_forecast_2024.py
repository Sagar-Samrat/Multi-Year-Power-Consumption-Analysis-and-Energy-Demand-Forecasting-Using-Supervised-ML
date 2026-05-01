"""
Monthly Actual vs Predicted Power Demand for 2024
This script generates a comparison graph showing actual power demand vs model predictions
for each month of 2024.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Load the full dataset with datetime parsing."""
    try:
        filepath = "/Users/sagarsamrat/Downloads/powerdemand_5min_2021_to_2024_with weather.csv"
        df = pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
        return df
    except FileNotFoundError:
        print("Error: Data file not found at the specified path.")
        return None

def load_models():
    """Load the trained model and scaler."""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure ml_pipeline.py has been run to generate best_model.pkl and scaler.pkl")
        return None, None

def prepare_2024_data(df):
    """Filter and prepare 2024 data."""
    df_2024 = df.loc['2024-01-01':'2024-12-31'].copy()
    
    # Ensure required features exist
    required_features = ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 
                         'lag_24', 'lag_288', 'rolling_mean_12', 'Power demand']
    
    # Check if features exist, if not, regenerate them
    if not all(col in df_2024.columns for col in required_features):
        print("Regenerating missing features...")
        df_2024['hour'] = df_2024.index.hour
        df_2024['day'] = df_2024.index.day
        df_2024['month'] = df_2024.index.month
        df_2024['year'] = df_2024.index.year
        df_2024['weekday'] = df_2024.index.weekday
        df_2024['lag_24'] = df_2024['Power demand'].shift(24)
        df_2024['lag_288'] = df_2024['Power demand'].shift(288)
        df_2024['rolling_mean_12'] = df_2024['Power demand'].shift(24).rolling(window=12).mean()
    
    # Remove rows with missing values
    df_2024 = df_2024.dropna(subset=required_features)
    
    return df_2024

def get_monthly_actual_demand(df_2024):
    """Calculate actual monthly power demand (average per month)."""
    monthly_actual = df_2024['Power demand'].resample('MS').mean()
    return monthly_actual

def get_monthly_predicted_demand(df_2024, model, scaler):
    """Make predictions for 2024 data and calculate monthly averages."""
    feature_cols = ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 
                    'lag_24', 'lag_288', 'rolling_mean_12']
    
    X = df_2024[feature_cols]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    
    # Create dataframe with predictions
    df_pred = pd.DataFrame({
        'Predicted': predictions
    }, index=df_2024.index)
    
    # Calculate monthly averages of predictions
    monthly_predicted = df_pred['Predicted'].resample('MS').mean()
    
    return monthly_predicted

def create_monthly_comparison_graph(monthly_actual, monthly_predicted):
    """Create and save a professional comparison graph."""
    # Create month labels
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Align data for both actual and predicted
    dates = monthly_actual.index
    x_positions = np.arange(len(dates))
    
    # Create figure with professional styling
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid", palette="husl")
    
    # Plot bars for actual and predicted
    width = 0.35
    bars1 = plt.bar(x_positions - width/2, monthly_actual.values, width, 
                    label='Actual Power Demand', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = plt.bar(x_positions + width/2, monthly_predicted.values, width, 
                    label='Predicted Power Demand', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize the plot
    plt.xlabel('Month (2024)', fontsize=13, fontweight='bold')
    plt.ylabel('Average Power Demand (MW)', fontsize=13, fontweight='bold')
    plt.title('Monthly Actual vs Predicted Power Demand - 2024', 
              fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_positions, months, fontsize=11, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left', framealpha=0.95)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('monthly_actual_vs_predicted_2024.png', dpi=300, bbox_inches='tight')
    print("✅ Graph saved as 'monthly_actual_vs_predicted_2024.png'")
    plt.close()
    
    return monthly_actual, monthly_predicted

def calculate_metrics(monthly_actual, monthly_predicted):
    """Calculate and display performance metrics."""
    mae = np.mean(np.abs(monthly_actual.values - monthly_predicted.values))
    rmse = np.sqrt(np.mean((monthly_actual.values - monthly_predicted.values) ** 2))
    mape = np.mean(np.abs((monthly_actual.values - monthly_predicted.values) / monthly_actual.values)) * 100
    
    print("\n" + "="*60)
    print("MONTHLY FORECAST PERFORMANCE METRICS (2024)")
    print("="*60)
    print(f"Mean Absolute Error (MAE):     {mae:,.2f} MW")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} MW")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("="*60 + "\n")
    
    # Display month-by-month comparison
    print("Month-by-Month Comparison:")
    print("-" * 80)
    print(f"{'Month':<12} {'Actual (MW)':<18} {'Predicted (MW)':<18} {'Difference (MW)':<15}")
    print("-" * 80)
    
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    for i, month in enumerate(months):
        if i < len(monthly_actual):
            actual = monthly_actual.values[i]
            predicted = monthly_predicted.values[i]
            diff = actual - predicted
            print(f"{month:<12} {actual:>15,.2f}    {predicted:>15,.2f}    {diff:>13,.2f}")
    
    print("-" * 80 + "\n")

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("MONTHLY ACTUAL VS PREDICTED POWER DEMAND - 2024")
    print("="*60 + "\n")
    
    # Step 1: Load data
    print("Loading dataset...")
    df = load_data()
    if df is None:
        return
    
    # Step 2: Load models
    print("Loading trained model and scaler...")
    model, scaler = load_models()
    if model is None or scaler is None:
        return
    
    # Step 3: Prepare 2024 data
    print("Preparing 2024 data...")
    df_2024 = prepare_2024_data(df)
    
    if df_2024.empty:
        print("Error: No valid 2024 data found.")
        return
    
    print(f"Found {len(df_2024)} records for 2024")
    
    # Step 4: Calculate actual monthly demand
    print("Calculating actual monthly power demand...")
    monthly_actual = get_monthly_actual_demand(df_2024)
    
    # Step 5: Generate predictions
    print("Generating model predictions...")
    monthly_predicted = get_monthly_predicted_demand(df_2024, model, scaler)
    
    # Step 6: Create visualization
    print("Creating visualization...")
    create_monthly_comparison_graph(monthly_actual, monthly_predicted)
    
    # Step 7: Calculate and display metrics
    calculate_metrics(monthly_actual, monthly_predicted)
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
