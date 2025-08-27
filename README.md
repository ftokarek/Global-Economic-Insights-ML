# World Economic Insights 2025 

This project demonstrates comprehensive data science skills by analyzing the World Economics 2025 dataset through:

- **Exploratory Data Analysis (EDA)**: Understanding distributions, correlations, and regional patterns
- **Clustering Analysis**: Grouping countries by economic profiles using K-means and PCA
- **Predictive Modeling**: Building models to predict GDP Growth using LightGBM and ElasticNet
- **Interactive Visualization**: Streamlit app with country profiles, predictions, and scenario analysis
- **What-if Analysis**: Simulating economic policy changes and their impact on GDP growth

## Dataset

The project uses the World Economics 2025 dataset containing:
- **Economic Indicators**: GDP, GDP Growth, Inflation, Interest Rates, Debt/GDP, etc.
- **Geographic Data**: Countries, regions, subregions with ISO codes
- **Coverage**: Global economic snapshot for 2025 projections

## Technical Details

### Model Performance
- **LightGBM**: Primary model with feature importance
- **ElasticNet**: Baseline linear model
- **Cross-validation**: K-fold validation for robust evaluation
- **Feature Engineering**: Imputation, scaling, and preprocessing

### Data Quality
- **Missing Values**: Handled with regional and global medians
- **Outliers**: Identified and documented in EDA
- **Validation**: Data type checking and range validation

## Key Insights

### Economic Patterns
- Regional variations in GDP growth and inflation
- Correlation between debt levels and economic performance
- Clustering reveals distinct economic development stages

### Model Performance
- LightGBM achieves better prediction accuracy
- Key drivers: Inflation, Interest Rates, Debt/GDP
- Cross-sectional analysis provides directional insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

