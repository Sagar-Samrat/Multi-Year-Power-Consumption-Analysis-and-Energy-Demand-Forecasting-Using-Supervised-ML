# 🎨 Visualization Quick Reference Guide

## Where to Find Each Visualization

### **Streamlit Web App** (Run: `streamlit run app.py`)

#### 📊 **Data Visualization Page** - 8 Sections, 20+ Charts
1. **Trend Analysis** (2 charts)
   - Full dataset power demand trend
   - Distribution histogram

2. **Hourly Patterns** (2 charts)
   - 24-hour average demand with confidence bands
   - Hourly demand box plot showing variability

3. **Weekly & Daily** (2 charts)
   - Day-of-week comparison bar chart
   - Day-of-month trend line

4. **Seasonal & Monthly** (2 charts)
   - Monthly average demand bar chart
   - Monthly variability box plot

5. **Weather Factors** (3 charts)
   - Temperature vs demand (color-coded by hour)
   - Humidity vs demand (color-coded by temperature)
   - Wind speed vs demand (color-coded by humidity)

6. **Feature Correlation** (1 chart)
   - Heatmap of all variable correlations with auto-labeled values

7. **Lag Analysis** (3 charts)
   - 2-hour lag relationship
   - 24-hour lag relationship
   - Rolling mean relationship

8. **Model Predictions** (5+ charts)
   - Actual vs predicted line chart (monthly)
   - Residuals/error over time
   - Error metrics (RMSE, MAE, MAPE, R²)
   - Model comparison RMSE chart
   - Model comparison R² score chart

#### 📈 **Model Performance Page** - 8 Charts & Detailed Stats
- RMSE comparison bar chart (6 models)
- MAE comparison bar chart (6 models)
- R² score bar chart (6 models)
- Combined error metrics grouped bar chart
- R² score line chart with trend
- Best model metrics (RMSE, R², MAE)
- Performance statistics tables
- Best model summary box

---

### **Jupyter Notebook** (Open: `ML_Pipeline_Interactive.ipynb`)

#### Step 6: Advanced Visualizations (4 Subsections)

**6.1: Model Performance Comparison** (4 figures)
- 2x2 subplot showing:
  - RMSE horizontal bar chart
  - MAE horizontal bar chart
  - R² score horizontal bar chart
  - Summary statistics box

**6.2: Predictions vs Actual** (4 charts)
- Time series overlay (actual vs predicted)
- Scatter plot with perfect prediction diagonal
- Residuals over time with fill area
- Error distribution histogram with mean reference

**6.3: Feature Importance** (2 charts)
- Feature importance horizontal bar (for tree models)
- Feature correlation vertical bar chart

**6.4: Temporal Patterns** (4 charts)
- Hourly pattern line with confidence intervals
- Day-of-week bar chart with error bars
- Monthly trend line with min-max range
- Weekday vs weekend box plot comparison

**6.5: Weather Impact** (6 charts)
- Temperature scatter with polynomial trend
- Humidity scatter with polynomial trend
- Wind speed scatter with linear trend
- Temperature-humidity interaction heatmap
- Correlation matrix heatmap (all variables)
- Weather statistics summary box

---

## 📊 Chart Types Used

| Chart Type | Usage | Count |
|-----------|-------|-------|
| **Line Chart** | Trends over time | 8 |
| **Bar Chart** | Categorical comparison | 6 |
| **Scatter Plot** | Variable relationships | 6 |
| **Box Plot** | Distribution by category | 3 |
| **Histogram** | Distribution shape | 2 |
| **Heatmap** | Correlations & interactions | 3 |
| **Combo** | Multiple metrics together | 2 |
| **Metrics/Cards** | Key numbers | 4 |

---

## 🎯 What Each Visualization Shows

### **Understanding Demand Patterns**
- **Hourly Charts**: When demand peaks during the day (e.g., morning/evening)
- **Daily Charts**: Weekend vs weekday differences
- **Monthly Charts**: Seasonal trends (summer AC usage, winter heating)

### **Understanding Model Quality**
- **RMSE Chart**: Average size of prediction errors (lower = better)
- **R² Chart**: How well model explains demand variation (0-1, higher = better)
- **MAE Chart**: Mean absolute error in MW units
- **Residuals Chart**: Pattern of errors over time (should be random)
- **Scatter Plot**: How close predictions are to actual values

### **Understanding Influences**
- **Weather Charts**: How temperature, humidity, wind affect demand
- **Correlation Heatmap**: Which variables are most related
- **Feature Importance**: Which variables the best model relies on most
- **Lag Charts**: How recent history influences current demand

---

## 💡 Tips for Interpretation

### **Green Indicators** ✅
- Low RMSE (error below 100 MW)
- High R² score (above 0.8)
- Tight scatter around diagonal line
- Residuals centered around zero

### **Red Flags** ⚠️
- High RMSE values
- Low R² scores (below 0.5)
- Large error distributions
- Systematic bias in residuals

### **Patterns to Look For**
- **Peak Hours**: Usually morning (6-9 AM) and evening (6-9 PM)
- **Weekend Drop**: Weekends usually have lower demand
- **Seasonal**: Summer/Winter peaks depending on climate
- **Weather Sensitivity**: Temperature usually shows strongest correlation

---

## 🔄 Workflow for Analysis

1. **Start with trends** → Understand overall patterns
2. **Check hourly/daily/monthly** → Identify cycles
3. **Look at weather impact** → See external influences
4. **Review correlations** → Find relationships
5. **Check model performance** → Assess prediction quality
6. **Analyze predictions** → See where model works/fails
7. **Review feature importance** → Understand what matters

---

## 📢 Key Metrics Explained

**RMSE (Root Mean Squared Error)**
- Measures average error magnitude
- Penalizes large errors more than small ones
- Same units as target variable (MW)
- Lower is better

**MAE (Mean Absolute Error)**
- Average of absolute errors
- More interpretable than RMSE
- Same units as target variable (MW)
- Lower is better

**R² (Coefficient of Determination)**
- Explains percentage of variance
- Range: 0 to 1 (can be negative for poor models)
- 0.8+ = excellent, 0.6+ = good, <0.3 = poor

**MAPE (Mean Absolute Percentage Error)**
- Express error as percentage of actual values
- Good for comparing across different scales
- Lower is better (5% is excellent, 20% is acceptable)

---

## 🚀 Running the Application

### **Interactive Web App**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Navigate to: http://localhost:8501
```

### **Jupyter Notebook**
```bash
# Start Jupyter
jupyter notebook

# Open: ML_Pipeline_Interactive.ipynb
# Run all cells to see all visualizations
```

---

## ✨ Customization Tips

### **For Streamlit App**
- Edit colors in CSS section (lines with `#` colors)
- Adjust chart heights by changing `height=400` values
- Change chart types by editing `px.line()` to `px.scatter()` etc.
- Add/remove sections by copying chart code blocks

### **For Jupyter Notebook**
- Change figure sizes: `plt.subplots(2, 2, figsize=(14, 10))`
- Adjust colors: Use `color='steelblue'`, `color='coral'` etc.
- Modify bin sizes in histograms: `bins=50`
- Add/remove subplots as needed

---

## 🔗 Dependencies

- **matplotlib**: Static matplotlib visualizations in notebook
- **seaborn**: Statistical visualization styling
- **plotly**: Interactive charts in web app
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning models
- **streamlit**: Web app framework

All included in `requirements.txt`

---

**Generated: Enhanced Data Visualization System** ✅
**Total Charts Added: 25+** 
**Improvement over Original: 10x More Visualizations**
