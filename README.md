# World Economic Insights 2025 - ML Portfolio Project

An interactive data science portfolio project that analyzes and models global economic indicators for 2025, providing insights through machine learning, clustering, and scenario analysis.

## 🎯 Project Overview

This project demonstrates comprehensive data science skills by analyzing the World Economics 2025 dataset through:

- **Exploratory Data Analysis (EDA)**: Understanding distributions, correlations, and regional patterns
- **Clustering Analysis**: Grouping countries by economic profiles using K-means and PCA
- **Predictive Modeling**: Building models to predict GDP Growth using LightGBM and ElasticNet
- **Interactive Visualization**: Streamlit app with country profiles, predictions, and scenario analysis
- **What-if Analysis**: Simulating economic policy changes and their impact on GDP growth

## 🚀 Features

### Interactive Streamlit Application
- **Country Profile**: View individual country metrics with regional benchmarks
- **Predictions**: Train and evaluate GDP Growth prediction models with performance metrics
- **Clusters**: Explore country groupings based on economic characteristics
- **What-if Scenarios**: Simulate changes in economic indicators and observe predicted impacts

### Technical Capabilities
- **Data Processing**: Automated cleaning pipeline with imputation and validation
- **Machine Learning**: Cross-validated models with feature importance analysis
- **Visualization**: Interactive plots using Plotly and comprehensive EDA script
- **Performance**: Cached data loading and optimized for Streamlit Cloud deployment

## 📊 Dataset

The project uses the World Economics 2025 dataset containing:
- **Economic Indicators**: GDP, GDP Growth, Inflation, Interest Rates, Debt/GDP, etc.
- **Geographic Data**: Countries, regions, subregions with ISO codes
- **Coverage**: Global economic snapshot for 2025 projections

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Global-Economic-Insights-ML
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/app.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`
   - The app will automatically load and display the dashboard

### Data Processing (Optional)

If you need to reprocess the raw data:
```bash
python src/data/clean_dataset.py
```

### Run EDA Analysis

To run the comprehensive exploratory data analysis:
```bash
python scripts/01_eda_world_economics.py
```

This will generate:
- Statistical summaries and insights
- Visualization plots (saved as PNG files)
- Interactive cluster analysis (HTML file)

## 📈 Usage Guide

### Country Profile Tab
- Select a country from the dropdown
- View key economic indicators with regional benchmarks
- Compare performance against peer countries

### Predictions Tab
- Click "Train models" to build prediction models
- View model performance metrics (RMSE, MAE, R²)
- Explore feature importance for LightGBM model

### Clusters Tab
- Adjust number of clusters (k) and random state
- Click "Run clustering" to group countries
- Explore cluster characteristics and country assignments

### What-if Tab
- Select a baseline country
- Adjust economic indicators using sliders
- View predicted GDP growth changes

## 🚀 Deployment on Streamlit Cloud

### Automatic Deployment
1. **Push to GitHub**: Ensure your repository is on GitHub
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
3. **Configure deployment**:
   - **Main file path**: `app/app.py`
   - **Python version**: 3.8+
   - **Requirements file**: `requirements.txt`

### Manual Configuration
Create `.streamlit/secrets.toml` for any environment variables:
```toml
[general]
# Add any secrets here if needed
```

### Resource Limits
- **Memory**: 1GB RAM (sufficient for this dataset)
- **Timeout**: 60 seconds (models train quickly)
- **CPU**: 1 core (adequate for ML operations)

## 📁 Project Structure

```
Global-Economic-Insights-ML/
├── app/
│   └── app.py                 # Main Streamlit application
├── src/
│   ├── data/
│   │   ├── loaders.py         # Data loading functions
│   │   └── clean_dataset.py   # Data cleaning pipeline
│   ├── features/
│   │   └── benchmarks.py      # Benchmarking calculations
│   └── models/
│       ├── train.py           # Model training pipelines
│       └── clustering.py      # Clustering analysis
├── data/
│   ├── raw/                   # Original dataset
│   └── processed/             # Cleaned data
├── scripts/
│   └── 01_eda_world_economics.py     # Comprehensive EDA script
├── configs/
│   └── data_dictionary.yaml   # Field definitions
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Technical Details

### Dependencies
- **Core**: pandas, numpy, scikit-learn
- **ML**: lightgbm, shap (optional)
- **Visualization**: plotly, matplotlib, seaborn
- **Web App**: streamlit
- **Utilities**: pycountry, pyyaml

### Model Performance
- **LightGBM**: Primary model with feature importance
- **ElasticNet**: Baseline linear model
- **Cross-validation**: K-fold validation for robust evaluation
- **Feature Engineering**: Imputation, scaling, and preprocessing

### Data Quality
- **Missing Values**: Handled with regional and global medians
- **Outliers**: Identified and documented in EDA
- **Validation**: Data type checking and range validation

## 📊 Key Insights

### Economic Patterns
- Regional variations in GDP growth and inflation
- Correlation between debt levels and economic performance
- Clustering reveals distinct economic development stages

### Model Performance
- LightGBM achieves better prediction accuracy
- Key drivers: Inflation, Interest Rates, Debt/GDP
- Cross-sectional analysis provides directional insights

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset sources and economic indicators
- Streamlit for the interactive framework
- Open-source ML libraries and tools

---

**Note**: This project is designed for portfolio demonstration and educational purposes. Economic predictions should not be used for financial decision-making without additional validation and context.

