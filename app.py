import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Page Configuration
st.set_page_config(page_title="Power Demand Forecaster", page_icon="⚡", layout="wide")

# Helper to load trained model artifacts
@st.cache_resource
def load_models():
    model = None
    scaler = None
    results = None
    try:
        model = joblib.load('best_model.pkl')
    except Exception:
        model = None

    try:
        scaler = joblib.load('scaler.pkl')
    except Exception:
        scaler = None

    try:
        results = joblib.load('model_results.pkl')
    except Exception:
        results = None

    return model, scaler, results

# Helper to load full dataset for analysis (with cache)
@st.cache_data
def load_full_data():
    try:
        filepath = "powerdemand_5min_2021_to_2024_with weather.csv"
        return pd.read_csv(filepath, parse_dates=['datetime'], index_col='datetime')
    except Exception:
        return None

# Helper to load sample data for charts
@st.cache_data
def load_data():
    try:
        return pd.read_csv('cleaned_data_sample.csv', parse_dates=['datetime'], index_col='datetime')
    except Exception as e:
        return None

@st.cache_data
def make_prediction_comparison_yearly(df, _model, _scaler):
    """Generate yearly actual vs predicted comparison"""
    feature_cols = ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 'lag_24', 'lag_288', 'rolling_mean_12']
    df_work = df.copy()
    
    # Ensure datetime index exists
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)
    
    # Regenerate features if missing
    if 'hour' not in df_work.columns:
        df_work['hour'] = df_work.index.hour
    if 'day' not in df_work.columns:
        df_work['day'] = df_work.index.day
    if 'month' not in df_work.columns:
        df_work['month'] = df_work.index.month
    if 'year' not in df_work.columns:
        df_work['year'] = df_work.index.year
    if 'weekday' not in df_work.columns:
        df_work['weekday'] = df_work.index.weekday
    if 'lag_24' not in df_work.columns:
        df_work['lag_24'] = df_work['Power demand'].shift(24)
    if 'lag_288' not in df_work.columns:
        df_work['lag_288'] = df_work['Power demand'].shift(288)
    if 'rolling_mean_12' not in df_work.columns:
        df_work['rolling_mean_12'] = df_work['Power demand'].shift(24).rolling(window=12).mean()
    
    # Drop rows with missing values
    df_clean = df_work.dropna(subset=feature_cols + ['Power demand']).copy()
    
    if df_clean.empty or len(df_clean) == 0:
        raise ValueError(f'No valid data available after feature engineering. Required features: {feature_cols}')
    
    # Make predictions for all available data
    X = df_clean[feature_cols]
    X_scaled = _scaler.transform(X)
    preds = _model.predict(X_scaled)
    
    df_result = df_clean[['Power demand']].copy()
    df_result['Predicted demand'] = preds
    
    # Monthly aggregation for yearly view
    df_monthly = df_result.resample('MS').mean()  # Monthly Start
    df_monthly['Year'] = df_monthly.index.year
    df_monthly['Month'] = df_monthly.index.month
    df_monthly['Month_Num'] = df_monthly.index.strftime('%b')
    
    return df_monthly

@st.cache_data
def make_prediction_comparison(df, _model, _scaler):
    feature_cols = ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 'lag_24', 'lag_288', 'rolling_mean_12']
    df_clean = df.dropna(subset=feature_cols + ['Power demand']).copy()
    df_filtered = df_clean.loc['2024-07-01':'2024-12-31']
    if df_filtered.empty:
        raise ValueError('No valid sample rows available for prediction comparison in July-Dec 2024.')
    X = df_filtered[feature_cols]
    X_scaled = _scaler.transform(X)
    preds = _model.predict(X_scaled)
    df_result = df_filtered[['Power demand']].copy()
    df_result['Predicted demand'] = preds
    df_monthly = df_result.resample('M').mean()
    df_monthly.index = df_monthly.index.to_period('M').to_timestamp()
    return df_monthly

# Custom CSS to improve UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3256/3256150.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Module:", 
                        ["🏠 Home Page", 
                         "📊 Data Visualization", 
                         "🔮 Prediction Page", 
                         "📈 Model Performance",
                         "💻 Pipeline Source Code"])

# Contexts
if page == "🏠 Home Page":
    st.title("⚡ Power Demand Forecasting System")
    st.markdown("""
    This modern application uses **Machine Learning** to predict electric power demand using historical interval data and weather variables.
    
    ### 🎯 Project Overview
    Power demand planning is crucial for utility companies. By observing historical trends, specific temporal patterns (day, night, weekends), and weather correlations (like temperature and humidity), we can train Supervised Learning models to predict future demand accurately.
    
    ### 🛠️ Features Engineered
    - **Time Patterns:** Hour of day, Day of month, Month, Weekday
    - **Weather Inputs:** Temperature, Relative Humidity, Wind Speed
    - **Lag Indicators:** 2-Hour lag $(t-24)$ and 24-Hour lag $(t-288)$
    - **Rolling Averages:** 1-hour Moving Average (Delayed)
    
    Navigate to the other pages via the sidebar to view historical visualizations and predict future demand!
    """)

elif page == "📊 Data Visualization":
    st.title("📊 Historical Data Insights & Analysis")
    
    # Try to load full dataset first for seasonal analysis
    df_full = load_full_data()
    df_sample = load_data()
    
    if df_sample is not None:
        st.write("Sample of Cleaned Data:")
        st.dataframe(df_sample.head(10))
    
    # Use full data for seasonal pattern if available, otherwise use sample
    df_for_analysis = df_full if df_full is not None else df_sample
    
    if df_for_analysis is not None:
        st.markdown("---")
        st.header("🌦️ Seasonal Pattern - Yearly Analysis")
        st.markdown("**Monthly average power demand showing seasonal patterns across years.**")
        
        # Prepare data for seasonal analysis
        df_analysis = df_for_analysis.copy()
        df_analysis['Year'] = df_analysis.index.year.astype(str)
        df_analysis['Month'] = df_analysis.index.month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Group by Year and Month to get average demand
        seasonal_data = df_analysis.groupby(['Year', 'Month'])['Power demand'].agg(['mean', 'min', 'max']).reset_index()
        seasonal_data['Month_Name'] = seasonal_data['Month'].map(lambda x: month_names[x-1])
        
        # Create line chart for seasonal patterns
        fig_seasonal = px.line(seasonal_data, 
                               x='Month', 
                               y='mean', 
                               color='Year',
                               markers=True,
                               title='Seasonal Pattern: Monthly Average Power Demand by Year',
                               labels={'mean': 'Avg Demand (MW)', 'Month': 'Month'},
                               template='plotly_white',
                               line_shape='spline')
        
        fig_seasonal.update_xaxes(
            ticktext=month_names,
            tickvals=list(range(1, 13))
        )
        fig_seasonal.update_layout(
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(240,240,240,0.5)',
            xaxis_title='Month',
            yaxis_title='Average Power Demand (MW)',
            legend=dict(title='Year', orientation='v')
        )
        fig_seasonal.update_traces(line=dict(width=2.5), marker=dict(size=8))
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Show data summary
        st.markdown("---")
        st.header("📊 Yearly Seasonal Summary")
        summary_table = seasonal_data.groupby('Year').agg({
            'mean': ['min', 'max', 'mean']
        }).round(2)
        summary_table.columns = ['Min Demand (MW)', 'Max Demand (MW)', 'Avg Demand (MW)']
        st.dataframe(summary_table, use_container_width=True)
        
        # === ACTUAL VS PREDICTED YEARLY ===
        st.markdown("---")
        st.header("🤖 Actual vs Predicted Demand - Yearly View")
        st.markdown("**Comparing actual power demand with model predictions on a monthly basis across years.**")
        
        model, scaler, _ = load_models()
        if model is not None and scaler is not None:
            try:
                pred_monthly = make_prediction_comparison_yearly(df_for_analysis, model, scaler)
                
                # Create line chart with both actual and predicted
                fig_actual_pred = go.Figure()
                
                # Add actual demand line
                fig_actual_pred.add_trace(go.Scatter(
                    x=pred_monthly.index,
                    y=pred_monthly['Power demand'],
                    mode='lines+markers',
                    name='Actual Demand',
                    line=dict(color='#2ecc71', width=2.5),
                    marker=dict(size=6),
                    hovertemplate='<b>Actual</b><br>Date: %{x|%b %Y}<br>Demand: %{y:.1f} MW<extra></extra>'
                ))
                
                # Add predicted demand line
                fig_actual_pred.add_trace(go.Scatter(
                    x=pred_monthly.index,
                    y=pred_monthly['Predicted demand'],
                    mode='lines+markers',
                    name='Predicted Demand',
                    line=dict(color='#e74c3c', width=2.5, dash='dash'),
                    marker=dict(size=6),
                    hovertemplate='<b>Predicted</b><br>Date: %{x|%b %Y}<br>Demand: %{y:.1f} MW<extra></extra>'
                ))
                
                fig_actual_pred.update_layout(
                    title='Actual vs Predicted Monthly Demand (Yearly View)',
                    xaxis_title='Month',
                    yaxis_title='Demand (MW)',
                    height=500,
                    hovermode='x unified',
                    plot_bgcolor='rgba(240,240,240,0.5)',
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                
                st.plotly_chart(fig_actual_pred, use_container_width=True)
                
                # Calculate and show metrics
                st.markdown("---")
                st.header("📈 Model Performance Metrics (Yearly Data)")
                
                rmse = np.sqrt(np.mean((pred_monthly['Power demand'] - pred_monthly['Predicted demand']) ** 2))
                mae = np.mean(np.abs(pred_monthly['Power demand'] - pred_monthly['Predicted demand']))
                residuals = pred_monthly['Power demand'] - pred_monthly['Predicted demand']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 RMSE", f"{rmse:.2f} MW")
                with col2:
                    st.metric("📏 MAE", f"{mae:.2f} MW")
                with col3:
                    mape = np.mean(np.abs(residuals / pred_monthly['Power demand'])) * 100
                    st.metric("🎯 MAPE", f"{mape:.2f}%")
                with col4:
                    r2 = 1 - (np.sum(residuals**2) / np.sum((pred_monthly['Power demand'] - pred_monthly['Power demand'].mean())**2))
                    st.metric("📈 R² Score", f"{r2:.4f}")
                
                # Explanation about why R² is higher
                st.warning("""
                📌 **Why is R² higher here than validation metrics?**
                
                - **Validation R² (XGBoost): 0.9468** - Calculated on individual 5-minute intervals
                - **Yearly R² (shown above): Often >0.99** - Calculated on monthly aggregated averages
                
                **Reason:** Monthly averaging smooths out noise and reduces variance in the data. Predicting smooth monthly averages is easier than predicting individual volatile 5-minute intervals.
                
                **True Model Performance:** The validation set metrics (RMSE, MAE, R²) are more representative of real-world performance on raw data.
                """)
                
                # === 2024 MONTHLY FORECAST ===
                st.markdown("---")
                st.header("📅 2024 Monthly Forecast - Actual vs Predicted")
                st.markdown("**Monthly average power demand for 2024 with model predictions vs actual recorded values.**")
                
                if os.path.exists('monthly_actual_vs_predicted_2024.png'):
                    from PIL import Image
                    img = Image.open('monthly_actual_vs_predicted_2024.png')
                    st.image(img, use_column_width=True, caption="Monthly Actual vs Predicted Power Demand - 2024")
                    
                    # Display 2024 metrics
                    st.markdown("---")
                    st.subheader("📊 2024 Forecast Performance Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 MAE (2024)", "67.21 MW")
                    with col2:
                        st.metric("📊 RMSE (2024)", "73.83 MW")
                    with col3:
                        st.metric("✅ MAPE (2024)", "1.49%")
                    
                    st.info("""
                    ℹ️ **Note on 2024 Performance Metrics:**
                    - These metrics are on **monthly aggregated data** (12 data points total)
                    - This is different from **validation set metrics** which use raw 5-minute intervals (~90K points)
                    - Aggregated monthly performance is inherently easier to predict than volatile minute-level data
                    - Compare with **Validation R² (XGBoost): 0.9468** on raw data for true model performance
                    """)
                    
                    # Monthly breakdown table
                    st.markdown("---")
                    st.subheader("📋 Month-by-Month Breakdown (2024)")
                    
                    months_data = {
                        'Month': ['January', 'February', 'March', 'April', 'May', 'June', 
                                 'July', 'August', 'September', 'October', 'November', 'December'],
                        'Actual (MW)': [4002.89, 3541.57, 3557.99, 4223.68, 5543.26, 5590.68,
                                       5377.00, 4878.87, 4620.07, 4138.03, 3525.38, 4403.54],
                        'Predicted (MW)': [3915.13, 3484.90, 3520.99, 4191.44, 5439.54, 5456.09,
                                          5295.31, 4837.89, 4567.71, 4105.46, 3463.51, 4318.52],
                        'Difference (MW)': [87.77, 56.68, 37.01, 32.24, 103.72, 134.59,
                                           81.70, 40.97, 52.36, 32.57, 61.88, 85.02]
                    }
                    df_2024_breakdown = pd.DataFrame(months_data)
                    
                    # Highlight rows
                    def highlight_diff(row):
                        if row['Difference (MW)'] > 100:
                            return ['background-color: #fff3cd'] * len(row)
                        elif row['Difference (MW)'] < 40:
                            return ['background-color: #d4edda'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(df_2024_breakdown.style.apply(highlight_diff, axis=1),
                                use_container_width=True)
                    
                    st.caption("🟡 Yellow = Higher deviation (>100 MW) | 🟢 Green = Lower error (<40 MW)")
                else:
                    st.info("📌 2024 monthly forecast graph not yet generated. Run the monthly_forecast_2024.py script to create it.")
                
            except Exception as e:
                st.error(f"⚠️ Could not generate yearly comparison chart: {str(e)}")
                st.info("💡 This may happen if the dataset doesn't have enough valid records after feature engineering. Try refreshing the page or check that all data files are present.")
        
    elif df_sample is not None:
        st.markdown("---")
        st.header("📊 Data Summary Statistics")
        st.warning("⚠️ Full dataset not found. Showing sample data statistics only.")
        st.write("**Basic Statistics:**")
        st.dataframe(df_sample.describe())
        
    else:
        st.warning("⚠️ Data not found. Please run `ml_pipeline.py` first.")

elif page == "🔮 Prediction Page":
    st.title("🔮 Demand Prediction Engine")
    st.markdown("Enter future environmental conditions and recent historical lags to predict the expected demand.")
    
    with st.container():
        st.subheader("Input Parameters")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**Weather Details**")
            temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.5)
            rhum = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            wspd = st.number_input("Wind Speed", min_value=0.0, max_value=150.0, value=10.0, step=0.5)
            
        with c2:
            st.markdown("**Temporal Details**")
            hour = st.slider("Hour of Day", 0, 23, 14)
            weekday = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
            day = st.slider("Day of Month", 1, 31, 15)
            month = st.slider("Month of Year", 1, 12, 6)
            
        with c3:
            st.markdown("**Machine Learning Lags**")
            lag_24 = st.number_input("Lag 2 hours (t-24)", value=2300.0)
            lag_288 = st.number_input("Lag 24 hours (t-288)", value=2400.0)
            rolling_mean = st.number_input("Rolling 1hr Mean (Delayed)", value=2350.0)
            
    st.markdown("---")
    if st.button("Genreate Power Demand Forecast", use_container_width=True):
        model, scaler, _ = load_models()
        if model is not None and scaler is not None:
            # ['temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday', 'lag_24', 'lag_288', 'rolling_mean_12']
            input_array = np.array([[temp, rhum, wspd, hour, day, month, weekday, lag_24, lag_288, rolling_mean]])
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)[0]
            
            st.success(f"### 📊 Expected Power Demand: {prediction:,.2f} MW")
            gauge_max = max(5000, prediction * 1.2)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "Predicted Demand (MW)"},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, gauge_max * 0.5], "color": "#dbeafe"},
                        {"range": [gauge_max * 0.5, gauge_max * 0.8], "color": "#93c5fd"},
                        {"range": [gauge_max * 0.8, gauge_max], "color": "#3b82f6"}
                    ]
                }
            ))
            fig.update_layout(margin={"t": 0, "b": 0, "l": 0, "r": 0}, height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Use this estimate to compare demand against your dispatch plan and identify peak load windows.")
        else:
            st.error("⚠️ Trained models are missing. Please ensure `best_model.pkl` and `scaler.pkl` exist in the directory.")

elif page == "📈 Model Performance":
    st.title("📈 Model Selection & Performance Analysis")
    st.markdown("**Comprehensive model performance comparison across all evaluation metrics.**")
    
    _, _, results = load_models()
    
    if results:
        st.markdown("### 🏆 Validation Performance Comparison")
        df_metrics = pd.DataFrame(results).T
        
        # Highlight best scores
        st.dataframe(df_metrics.style.highlight_min(subset=['RMSE', 'MAE'], color='#059669')\
                                     .highlight_max(subset=['R2'], color='#059669'),
                     use_container_width=True)
        
        st.markdown("---")
        st.header("📊 Performance Visualizations")
        
        # Create 3 columns for metric charts
        col1, col2, col3 = st.columns(3)
        
        # 1. RMSE Comparison
        with col1:
            fig_rmse = px.bar(
                x=df_metrics.index,
                y=df_metrics['RMSE'],
                title='RMSE Comparison',
                labels={'x': 'Model', 'y': 'RMSE (MW)'},
                color=df_metrics['RMSE'],
                color_continuous_scale='RdYlGn_r',
                template='plotly_white'
            )
            fig_rmse.update_layout(
                height=400,
                showlegend=False,
                yaxis_title='RMSE (MW)',
                xaxis_title='Model'
            )
            fig_rmse.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_rmse, use_container_width=True)
            st.caption("🔴 Lower is better - penalizes large errors")
        
        # 2. MAE Comparison
        with col2:
            fig_mae = px.bar(
                x=df_metrics.index,
                y=df_metrics['MAE'],
                title='MAE Comparison',
                labels={'x': 'Model', 'y': 'MAE (MW)'},
                color=df_metrics['MAE'],
                color_continuous_scale='RdYlGn_r',
                template='plotly_white'
            )
            fig_mae.update_layout(
                height=400,
                showlegend=False,
                yaxis_title='MAE (MW)',
                xaxis_title='Model'
            )
            fig_mae.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_mae, use_container_width=True)
            st.caption("🔴 Lower is better - average absolute error")
        
        # 3. R² Score Comparison
        with col3:
            fig_r2 = px.bar(
                x=df_metrics.index,
                y=df_metrics['R2'],
                title='R² Score Comparison',
                labels={'x': 'Model', 'y': 'R² Score'},
                color=df_metrics['R2'],
                color_continuous_scale='YlGn',
                template='plotly_white'
            )
            fig_r2.update_layout(
                height=400,
                showlegend=False,
                yaxis_title='R² Score',
                xaxis_title='Model'
            )
            fig_r2.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_r2, use_container_width=True)
            st.caption("🟢 Higher is better - variance explained")
        

        
        st.markdown("---")
        st.header("📈 Model Rankings & Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🥇 Best RMSE")
            best_rmse_model = df_metrics['RMSE'].idxmin()
            best_rmse_value = df_metrics['RMSE'].min()
            st.metric(best_rmse_model, f"{best_rmse_value:.2f} MW", delta=None)
            st.caption("Lower RMSE is better - penalizes large errors")
        
        with col2:
            st.subheader("🥇 Best R² Score")
            best_r2_model = df_metrics['R2'].idxmax()
            best_r2_value = df_metrics['R2'].max()
            st.metric(best_r2_model, f"{best_r2_value:.4f}", delta=None)
            st.caption("Higher R² is better - explains variance")
        
        with col3:
            st.subheader("🥇 Best MAE")
            best_mae_model = df_metrics['MAE'].idxmin()
            best_mae_value = df_metrics['MAE'].min()
            st.metric(best_mae_model, f"{best_mae_value:.2f} MW", delta=None)
            st.caption("Lower MAE is better - average error")
        
        best_model = df_metrics['RMSE'].idxmin()
        st.markdown("---")
        st.success(f"🏆 **Selected Model:** {best_model} achieved the best performance with the lowest Root Mean Squared Error (RMSE: {df_metrics.loc[best_model, 'RMSE']:.2f} MW)")
    else:
        st.warning("⚠️ Evaluation metrics not found. Run the training script first.")

elif page == "💻 Pipeline Source Code":
    st.title("💻 Pipeline Source Code & Notebook Comparison")
    st.markdown("This page displays the Python code used for the entire Data Cleaning, Feature Engineering, and Modeling process. You can natively compare the structured **Production Script** against the simplified **Jupyter Notebook**!")
    
    tab1, tab2 = st.tabs(["🐍 Production Script (ml_pipeline.py)", "📓 Interactive Notebook (ML_Pipeline_Interactive.ipynb)"])
    
    with tab1:
        st.markdown("### 🐍 Production Pipeline Script")
        st.info("This is the highly robust, modularized Python script used for automated execution in server environments.")
        try:
            with open('ml_pipeline.py', 'r') as f:
                code = f.read()
            st.code(code, language='python')
        except FileNotFoundError:
            st.error("⚠️ `ml_pipeline.py` file not found.")
            
    with tab2:
        st.markdown("### 📓 Jupyter Notebook Code Cells")
        st.info("This extracts all the pure Python Code from the Jupyter Notebook cells. Notice how it is broken down differently with visualizations integrated into the workflow!")
        try:
            import json
            with open('ML_Pipeline_Interactive.ipynb', 'r', encoding='utf-8') as f:
                nb_data = json.load(f)
            
            nb_code = ""
            cell_count = 1
            for cell in nb_data.get('cells', []):
                if cell.get('cell_type') == 'code':
                    nb_code += f"# {'='*60}\n# 🟢 JUPYTER CODE CELL {cell_count}\n# {'='*60}\n"
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        nb_code += "".join(source) + "\n\n\n"
                    else:
                        nb_code += source + "\n\n\n"
                    cell_count += 1
            st.code(nb_code, language='python')
        except Exception as e:
            st.error(f"⚠️ Could not load `ML_Pipeline_Interactive.ipynb`: {str(e)}")


