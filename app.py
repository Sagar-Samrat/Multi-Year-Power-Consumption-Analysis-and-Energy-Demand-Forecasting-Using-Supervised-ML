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

# Helper to load sample data for charts
@st.cache_data
def load_data():
    try:
        return pd.read_csv('cleaned_data_sample.csv', parse_dates=['datetime'], index_col='datetime')
    except Exception as e:
        return None

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
                         "💻 Pipeline Source Code",
                         "🚀 Run Pipeline Live"])

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
    df = load_data()
    
    if df is not None:
        st.write("Sample of Cleaned Data:")
        st.dataframe(df.head(10))
        
        # === SECTION 1: TREND ANALYSIS ===
        st.markdown("---")
        st.header("📈 1. DEMAND TREND & TIME SERIES ANALYSIS")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📉 Power Demand Over Time (Full Dataset)")
            fig1 = px.line(df, x=df.index, y='Power demand', 
                           title="Complete Power Demand Trend", 
                           template="plotly_white",
                           labels={'Power demand': 'Demand (MW)', 'datetime': 'DateTime'})
            fig1.update_layout(hovermode='x unified', height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("📊 Demand Distribution")
            fig_dist = px.histogram(df, x='Power demand', nbins=50,
                                   title="Distribution of Power Demand",
                                   template="plotly_white",
                                   labels={'Power demand': 'Demand (MW)', 'count': 'Frequency'})
            fig_dist.update_layout(height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # === SECTION 2: HOURLY PATTERNS ===
        st.markdown("---")
        st.header("⏰ 2. HOURLY DEMAND PATTERNS (24-HOUR CYCLES)")
        
        col1, col2 = st.columns(2)
        with col1:
            hourly_avg = df.groupby(df.index.hour)['Power demand'].agg(['mean', 'std', 'min', 'max']).reset_index()
            hourly_avg.columns = ['Hour', 'Mean', 'Std Dev', 'Min', 'Max']
            fig_hourly = px.line(hourly_avg, x='Hour', y=['Mean', 'Min', 'Max'],
                                 title="Average Demand by Hour of Day",
                                 template="plotly_white",
                                 labels={'value': 'Demand (MW)', 'variable': 'Metric'})
            fig_hourly.update_layout(height=400)
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            fig_hourly_box = px.box(df.reset_index(), x=df.reset_index()[df.index.name].dt.hour, 
                                    y='Power demand',
                                    title="Demand Distribution by Hour",
                                    template="plotly_white",
                                    labels={'Hour': 'Hour of Day', 'Power demand': 'Demand (MW)'})
            fig_hourly_box.update_xaxes(title_text="Hour of Day")
            fig_hourly_box.update_layout(height=400)
            st.plotly_chart(fig_hourly_box, use_container_width=True)
        
        # === SECTION 3: WEEKLY & DAILY PATTERNS ===
        st.markdown("---")
        st.header("📅 3. WEEKLY & DAILY PATTERNS")
        
        col1, col2 = st.columns(2)
        with col1:
            daily_avg = df.groupby(df.index.weekday)['Power demand'].mean().reset_index()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_avg['Day'] = daily_avg['datetime'].map(lambda x: day_names[x])
            fig_daily = px.bar(daily_avg, x='Day', y='Power demand',
                              title="Average Demand by Day of Week",
                              template="plotly_white",
                              labels={'Power demand': 'Demand (MW)'})
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            df_copy = df.reset_index()
            df_copy['DayOfMonth'] = df_copy['datetime'].dt.day
            daily_of_month = df_copy.groupby('DayOfMonth')['Power demand'].mean().reset_index()
            fig_dom = px.line(daily_of_month, x='DayOfMonth', y='Power demand',
                             title="Average Demand by Day of Month",
                             template="plotly_white",
                             labels={'Power demand': 'Demand (MW)', 'DayOfMonth': 'Day of Month'},
                             markers=True)
            fig_dom.update_layout(height=400)
            st.plotly_chart(fig_dom, use_container_width=True)
        
        # === SECTION 4: SEASONAL & MONTHLY PATTERNS ===
        st.markdown("---")
        st.header("🌞 4. SEASONAL & MONTHLY TRENDS")
        
        col1, col2 = st.columns(2)
        with col1:
            monthly_avg = df.groupby(df.index.month)['Power demand'].agg(['mean', 'min', 'max']).reset_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_avg['Month'] = monthly_avg['datetime'].map(lambda x: month_names[x-1] if x <= 12 else 'Unknown')
            fig_monthly = px.bar(monthly_avg, x='Month', y='mean',
                                title="Average Demand by Month",
                                template="plotly_white",
                                labels={'mean': 'Demand (MW)'})
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            fig_seasonal_box = px.box(df.reset_index(), x=df.reset_index()[df.index.name].dt.month,
                                      y='Power demand',
                                      title="Demand Variability by Month",
                                      template="plotly_white",
                                      labels={'Power demand': 'Demand (MW)', 'datetime': 'Month'})
            fig_seasonal_box.update_xaxes(title_text="Month")
            fig_seasonal_box.update_layout(height=400)
            st.plotly_chart(fig_seasonal_box, use_container_width=True)
        
        # === SECTION 5: WEATHER IMPACTS ===
        st.markdown("---")
        st.header("🌡️ 5. WEATHER CORRELATIONS WITH DEMAND")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_temp = px.scatter(df, x='temp', y='Power demand', color='hour',
                                 title="Temperature Impact on Demand",
                                 template="plotly_white",
                                 labels={'temp': 'Temperature (°C)', 'Power demand': 'Demand (MW)'},
                                 color_continuous_scale="RdYlBu_r",
                                 opacity=0.6)
            fig_temp.update_layout(height=400)
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            fig_humidity = px.scatter(df, x='rhum', y='Power demand', color='temp',
                                     title="Humidity Impact on Demand",
                                     template="plotly_white",
                                     labels={'rhum': 'Humidity (%)', 'Power demand': 'Demand (MW)', 'temp': 'Temp (°C)'},
                                     color_continuous_scale="Viridis",
                                     opacity=0.6)
            fig_humidity.update_layout(height=400)
            st.plotly_chart(fig_humidity, use_container_width=True)
        
        with col3:
            fig_wind = px.scatter(df, x='wspd', y='Power demand', color='rhum',
                                 title="Wind Speed Impact on Demand",
                                 template="plotly_white",
                                 labels={'wspd': 'Wind Speed', 'Power demand': 'Demand (MW)', 'rhum': 'Humidity (%)'},
                                 color_continuous_scale="Plasma",
                                 opacity=0.6)
            fig_wind.update_layout(height=400)
            st.plotly_chart(fig_wind, use_container_width=True)
        
        # === SECTION 6: CORRELATION HEATMAP ===
        st.markdown("---")
        st.header("🔗 6. FEATURE CORRELATION ANALYSIS")
        
        corr_matrix = df[['Power demand', 'temp', 'rhum', 'wspd', 'hour', 'day', 'month', 'weekday']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto='.2f',
                            title="Feature Correlation Heatmap",
                            template="plotly_white",
                            color_continuous_scale="RdBu",
                            zmin=-1, zmax=1)
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # === SECTION 7: LAG ANALYSIS ===
        st.markdown("---")
        st.header("⏱️ 7. LAG FEATURES ANALYSIS")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_lag24 = px.scatter(df, x='lag_24', y='Power demand',
                                  title="2-Hour Lag vs Current Demand",
                                  template="plotly_white",
                                  labels={'lag_24': 'Power 2hrs ago (MW)', 'Power demand': 'Current Demand (MW)'},
                                  opacity=0.5)
            fig_lag24.update_layout(height=400)
            st.plotly_chart(fig_lag24, use_container_width=True)
        
        with col2:
            fig_lag288 = px.scatter(df, x='lag_288', y='Power demand',
                                   title="24-Hour Lag vs Current Demand",
                                   template="plotly_white",
                                   labels={'lag_288': 'Power 24hrs ago (MW)', 'Power demand': 'Current Demand (MW)'},
                                   color_continuous_scale="Greens",
                                   opacity=0.5)
            fig_lag288.update_layout(height=400)
            st.plotly_chart(fig_lag288, use_container_width=True)
        
        with col3:
            fig_rolling = px.scatter(df, x='rolling_mean_12', y='Power demand',
                                    title="1-Hour Rolling Mean vs Current Demand",
                                    template="plotly_white",
                                    labels={'rolling_mean_12': 'Rolling Mean (MW)', 'Power demand': 'Current Demand (MW)'},
                                    color_continuous_scale="Oranges",
                                    opacity=0.5)
            fig_rolling.update_layout(height=400)
            st.plotly_chart(fig_rolling, use_container_width=True)
        
        # === SECTION 8: MODEL PREDICTIONS ===
        model, scaler, results = load_models()
        if model is not None and scaler is not None:
            st.markdown("---")
            st.header("🤖 8. MODEL PERFORMANCE & PREDICTIONS")
            
            try:
                comparison_df = make_prediction_comparison(df, model, scaler)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_pred = px.line(comparison_df,
                                      y=['Power demand', 'Predicted demand'],
                                      title='Actual vs Predicted Demand (Monthly Average)',
                                      labels={'value': 'Demand (MW)', 'variable': 'Series'},
                                      template='plotly_white',
                                      markers=True)
                    fig_pred.update_layout(hovermode='x unified', height=400)
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    rmse = np.sqrt(np.mean((comparison_df['Power demand'] - comparison_df['Predicted demand']) ** 2))
                    mae = np.mean(np.abs(comparison_df['Power demand'] - comparison_df['Predicted demand']))
                    residuals = comparison_df['Power demand'] - comparison_df['Predicted demand']
                    
                    comparison_df['Residuals'] = residuals
                    fig_residual = px.line(comparison_df, y='Residuals',
                                          title='Prediction Residuals Over Time',
                                          template='plotly_white',
                                          labels={'Residuals': 'Error (MW)'})
                    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_residual.update_layout(height=400, hovermode='x unified')
                    st.plotly_chart(fig_residual, use_container_width=True)
                
                # Model Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 RMSE", f"{rmse:.2f} MW")
                with col2:
                    st.metric("📏 MAE", f"{mae:.2f} MW")
                with col3:
                    mape = np.mean(np.abs(residuals / comparison_df['Power demand'])) * 100
                    st.metric("🎯 MAPE", f"{mape:.2f}%")
                with col4:
                    r2 = 1 - (np.sum(residuals**2) / np.sum((comparison_df['Power demand'] - comparison_df['Power demand'].mean())**2))
                    st.metric("📈 R² Score", f"{r2:.4f}")
                
            except Exception as e:
                st.warning(f"Could not generate prediction charts: {str(e)}")
            
            # Model Comparison
            if results is not None:
                st.subheader("Model Comparison")
                results_df = pd.DataFrame(results).T
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_rmse_comp = px.bar(results_df.reset_index(), x='index', y='RMSE',
                                          title='Model Comparison - RMSE (Lower is Better)',
                                          labels={'index': 'Model', 'RMSE': 'RMSE (MW)'},
                                          template='plotly_white',
                                          color='RMSE',
                                          color_continuous_scale='Reds')
                    fig_rmse_comp.update_layout(height=400)
                    st.plotly_chart(fig_rmse_comp, use_container_width=True)
                
                with col2:
                    fig_r2_comp = px.bar(results_df.reset_index(), x='index', y='R2',
                                        title='Model Comparison - R² Score (Higher is Better)',
                                        labels={'index': 'Model', 'R2': 'R2 Score'},
                                        template='plotly_white',
                                        color='R2',
                                        color_continuous_scale='Greens')
                    fig_r2_comp.update_layout(height=400)
                    st.plotly_chart(fig_r2_comp, use_container_width=True)
        else:
            st.info("ℹ️ Model artifacts not available for prediction comparison. Run `ml_pipeline.py` first.")
    else:
        st.warning("⚠️ Cleaned data sample not found. Please run `ml_pipeline.py` first.")

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
    
    _, _, results = load_models()
    
    if results:
        st.markdown("### 🏆 Validation Performance Comparison")
        df_metrics = pd.DataFrame(results).T
        
        # Highlight best scores
        st.dataframe(df_metrics.style.highlight_min(subset=['RMSE', 'MAE'], color='#059669')\
                                     .highlight_max(subset=['R2'], color='#059669'),
                     use_container_width=True)
        
        st.markdown("---")
        st.header("📊 Detailed Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # RMSE Comparison
        with col1:
            fig_rmse = px.bar(df_metrics.reset_index(), x='index', y='RMSE',
                             title='Root Mean Squared Error (RMSE)',
                             labels={'index': 'Model', 'RMSE': 'RMSE (MW)'},
                             template='plotly_white',
                             color='RMSE',
                             color_continuous_scale='Reds')
            fig_rmse.update_layout(height=400, showlegend=False)
            fig_rmse.update_xaxes(tickangle=45)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # MAE Comparison
        with col2:
            fig_mae = px.bar(df_metrics.reset_index(), x='index', y='MAE',
                            title='Mean Absolute Error (MAE)',
                            labels={'index': 'Model', 'MAE': 'MAE (MW)'},
                            template='plotly_white',
                            color='MAE',
                            color_continuous_scale='Blues')
            fig_mae.update_layout(height=400, showlegend=False)
            fig_mae.update_xaxes(tickangle=45)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # R2 Score Comparison
        with col3:
            fig_r2 = px.bar(df_metrics.reset_index(), x='index', y='R2',
                           title='R² Score (Coefficient of Determination)',
                           labels={'index': 'Model', 'R2': 'R² Score'},
                           template='plotly_white',
                           color='R2',
                           color_continuous_scale='Greens')
            fig_r2.update_layout(height=400, showlegend=False)
            fig_r2.update_xaxes(tickangle=45)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        st.markdown("---")
        
        # Combined comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_combined = go.Figure()
            fig_combined.add_trace(go.Bar(name='RMSE', x=df_metrics.index, y=df_metrics['RMSE'], marker_color='#ef4444'))
            fig_combined.add_trace(go.Bar(name='MAE', x=df_metrics.index, y=df_metrics['MAE'], marker_color='#3b82f6'))
            fig_combined.update_layout(
                title='Error Metrics Comparison (RMSE vs MAE)',
                barmode='group',
                template='plotly_white',
                height=400,
                xaxis_title='Model',
                yaxis_title='Error (MW)',
                hovermode='x unified'
            )
            fig_combined.update_xaxes(tickangle=45)
            st.plotly_chart(fig_combined, use_container_width=True)
        
        with col2:
            # R2 Score visualization with annotations
            fig_r2_line = px.line(df_metrics.reset_index(), x='index', y='R2',
                                 title='R² Score Across Models',
                                 labels={'index': 'Model', 'R2': 'R² Score'},
                                 template='plotly_white',
                                 markers=True,
                                 line_shape='spline')
            fig_r2_line.update_traces(line=dict(width=3), marker=dict(size=10))
            fig_r2_line.update_layout(height=400)
            fig_r2_line.update_xaxes(tickangle=45)
            st.plotly_chart(fig_r2_line, use_container_width=True)
        
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
        
        st.markdown("---")
        st.header("📊 Performance Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**RMSE Statistics**")
            st.dataframe({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"{df_metrics['RMSE'].mean():.2f}",
                    f"{df_metrics['RMSE'].std():.2f}",
                    f"{df_metrics['RMSE'].min():.2f}",
                    f"{df_metrics['RMSE'].max():.2f}",
                    f"{df_metrics['RMSE'].max() - df_metrics['RMSE'].min():.2f}"
                ]
            }, use_container_width=True)
        
        with col2:
            st.write("**R² Statistics**")
            st.dataframe({
                'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"{df_metrics['R2'].mean():.4f}",
                    f"{df_metrics['R2'].std():.4f}",
                    f"{df_metrics['R2'].min():.4f}",
                    f"{df_metrics['R2'].max():.4f}",
                    f"{df_metrics['R2'].max() - df_metrics['R2'].min():.4f}"
                ]
            }, use_container_width=True)
        
        best_model = df_metrics['RMSE'].idxmin()
        st.markdown("---")
        st.success(f"🏆 **Selected Model:** {best_model} achieved the best performance with the lowest Root Mean Squared Error (RMSE: {df_metrics.loc[best_model, 'RMSE']:.2f} MW)")
    else:
        st.warning("⚠️ Evaluation metrics not found. Run the training script first.")

elif page == "💻 Pipeline Source Code":
    st.title("💻 Pipeline Source Code")
    st.markdown("This page displays the Python code used for the entire Data Cleaning, Feature Engineering, and Modeling process.")
    
    try:
        with open('ml_pipeline.py', 'r') as f:
            code = f.read()
        st.code(code, language='python')
    except FileNotFoundError:
        st.error("⚠️ `ml_pipeline.py` file not found.")

elif page == "🚀 Run Pipeline Live":
    st.title("🚀 Run Pipeline Live")
    st.markdown("Execute the complete Machine Learning pipeline—Data Cleaning, Feature Engineering, and Modeling—directly from the UI.")
    
    if st.button("▶️ Start Pipeline Execution", type="primary"):
        import ml_pipeline
        filepath = "/Users/sagarsamrat/Downloads/powerdemand_5min_2021_to_2024_with weather.csv"
        
        if not os.path.exists(filepath):
            st.error(f"Dataset not found at {filepath}")
        else:
            with st.status("Executing Machine Learning Pipeline...", expanded=True) as status:
                st.write("📥 **1. Loading data...**")
                df = ml_pipeline.load_data(filepath)
                st.write(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
                
                st.write("🧹 **2. Cleaning data...**")
                df = ml_pipeline.clean_data(df)
                st.write("Data cleaned (duplicates removed, outliers handled, missing values interpolated).")
                
                st.write("⚙️ **3. Engineering features...**")
                df = ml_pipeline.engineer_features(df)
                st.write(f"Time-based and lag features created. New shape: {df.shape[0]} rows, {df.shape[1]} columns.")
                
                st.write("✂️ **4. Splitting data...**")
                X_train, y_train, X_val, y_val, X_test, y_test, features = ml_pipeline.get_train_val_test_split(df)
                st.write(f"Data splitted: Train ({len(X_train)}), Validation ({len(X_val)}), Test ({len(X_test)}).")
                
                st.write("🧠 **5. Training Models...** (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, K-Neighbors) -> *This might take a few minutes.*")
                best_model, best_name = ml_pipeline.build_models(X_train, y_train, X_val, y_val)
                st.write(f"✅ **Models trained successfully!** Best model selected: {best_name}.")
                
                status.update(label="Pipeline Execution Complete!", state="complete", expanded=False)
            
            st.success("Pipeline executed and models saved successfully!")
            
            st.subheader("📊 Accuracy & Model Selection")
            st.markdown("Here is the evaluation of all trained models on the Validation Set:")
            
            # Load fresh results
            load_models.clear()
            _, _, results = load_models()
            if results:
                df_metrics = pd.DataFrame(results).T
                st.dataframe(df_metrics.style.highlight_min(subset=['RMSE', 'MAE'], color='#059669').highlight_max(subset=['R2'], color='#059669'), use_container_width=True)
                
                best_m = df_metrics['RMSE'].idxmin()
                st.info(f"**🎯 Automatic Selection:** **{best_m}** was officially selected due to obtaining the lowest RMSE score ({df_metrics.loc[best_m, 'RMSE']:.2f}).")
