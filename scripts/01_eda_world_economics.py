#!/usr/bin/env python3
"""
World Economics 2025 - Exploratory Data Analysis

This script provides comprehensive exploratory analysis of the World Economics 2025 dataset,
examining distributions, relationships, regional patterns, and outliers across key economic indicators.

Key Questions:
1. What are the distributions and characteristics of key economic indicators?
2. How do countries cluster by economic profiles?
3. What are the relationships between different economic indicators?
4. How do regions compare in terms of economic performance?
5. Which countries are outliers or interesting cases?
6. What patterns emerge in GDP growth drivers?

Usage:
    python scripts/01_eda_world_economics.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loaders import load_dataset
from src.features.benchmarks import compute_global_percentiles, compute_region_aggregates

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    """Main analysis function"""
    print("=" * 80)
    print("WORLD ECONOMICS 2025 - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # 1. Data Loading and Overview
    print("\n1. DATA LOADING AND OVERVIEW")
    print("-" * 40)
    
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Countries: {df['name'].nunique()}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Subregions: {df['subregion'].nunique()}")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Data info and missing values
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nMissing Values (%):")
    print((df.isnull().sum() / len(df) * 100).round(2))
    
    # Basic statistics for numeric columns
    numeric_cols = ['GDP', 'GDP Growth', 'Interest Rate', 'Inflation Rate', 
                    'Jobless Rate', 'Gov. Budget', 'Debt/GDP', 'Current Account', 'Population']
    
    print("\nBasic Statistics:")
    print(df[numeric_cols].describe().round(2))
    
    # 2. Regional Analysis
    print("\n\n2. REGIONAL ANALYSIS")
    print("-" * 40)
    
    # Regional distribution
    print("\nCountries per region:")
    region_counts = df['region'].value_counts()
    print(region_counts)
    
    # Regional statistics
    print("\nRegional Statistics:")
    regional_stats = df.groupby('region')[numeric_cols].agg(['mean', 'median', 'std']).round(2)
    print(regional_stats)
    
    # Create regional plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Countries per region
    axes[0,0].bar(region_counts.index, region_counts.values, color='skyblue')
    axes[0,0].set_title('Countries per Region')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # GDP by region
    region_gdp = df.groupby('region')['GDP'].agg(['mean', 'median']).round(2)
    axes[0,1].bar(region_gdp.index, region_gdp['mean'], color='lightgreen')
    axes[0,1].set_title('Average GDP by Region')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # GDP Growth by region
    region_growth = df.groupby('region')['GDP Growth'].agg(['mean', 'median']).round(2)
    axes[1,0].bar(region_growth.index, region_growth['mean'], color='orange')
    axes[1,0].set_title('Average GDP Growth by Region')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Inflation by region
    region_inflation = df.groupby('region')['Inflation Rate'].agg(['mean', 'median']).round(2)
    axes[1,1].bar(region_inflation.index, region_inflation['mean'], color='red')
    axes[1,1].set_title('Average Inflation Rate by Region')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('scripts/regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Distribution Analysis
    print("\n\n3. DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    # Distribution plots for key indicators
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        # Remove outliers for better visualization (top/bottom 5%)
        data = df[col].dropna()
        q1, q3 = data.quantile([0.05, 0.95])
        filtered_data = data[(data >= q1) & (data <= q3)]
        
        axes[i].hist(filtered_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        
        # Add mean line
        mean_val = data.mean()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('scripts/distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Correlation Analysis
    print("\n\n4. CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Correlation matrix
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Economic Indicators')
    plt.tight_layout()
    plt.savefig('scripts/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Top correlations with GDP Growth
    gdp_growth_corr = correlation_matrix['GDP Growth'].sort_values(ascending=False)
    print("\nTop correlations with GDP Growth:")
    print(gdp_growth_corr)
    
    # Scatter plots for key relationships
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # GDP Growth vs Inflation
    axes[0,0].scatter(df['Inflation Rate'], df['GDP Growth'], alpha=0.6)
    axes[0,0].set_xlabel('Inflation Rate (%)')
    axes[0,0].set_ylabel('GDP Growth (%)')
    axes[0,0].set_title('GDP Growth vs Inflation Rate')
    
    # GDP Growth vs Interest Rate
    axes[0,1].scatter(df['Interest Rate'], df['GDP Growth'], alpha=0.6)
    axes[0,1].set_xlabel('Interest Rate (%)')
    axes[0,1].set_ylabel('GDP Growth (%)')
    axes[0,1].set_title('GDP Growth vs Interest Rate')
    
    # GDP Growth vs Debt/GDP
    axes[1,0].scatter(df['Debt/GDP'], df['GDP Growth'], alpha=0.6)
    axes[1,0].set_xlabel('Debt/GDP (%)')
    axes[1,0].set_ylabel('GDP Growth (%)')
    axes[1,0].set_title('GDP Growth vs Debt/GDP')
    
    # GDP vs Population
    axes[1,1].scatter(df['Population'], df['GDP'], alpha=0.6)
    axes[1,1].set_xlabel('Population')
    axes[1,1].set_ylabel('GDP')
    axes[1,1].set_title('GDP vs Population')
    
    plt.tight_layout()
    plt.savefig('scripts/scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Outlier Analysis
    print("\n\n5. OUTLIER ANALYSIS")
    print("-" * 40)
    
    # Identify outliers using IQR method
    def find_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    
    print("Outliers by indicator:")
    for col in numeric_cols:
        outliers = find_outliers(df, col)
        if len(outliers) > 0:
            print(f"\n{col} outliers ({len(outliers)} countries):")
            for _, row in outliers.iterrows():
                print(f"  {row['name']}: {row[col]:.2f}")
        else:
            print(f"\n{col}: No outliers detected")
    
    # 6. Top Performers Analysis
    print("\n\n6. TOP PERFORMERS ANALYSIS")
    print("-" * 40)
    
    # Top and bottom performers
    print("Top 10 GDP Growth countries:")
    top_growth = df.nlargest(10, 'GDP Growth')[['name', 'region', 'GDP Growth', 'GDP', 'Inflation Rate']]
    print(top_growth.round(2))
    
    print("\nBottom 10 GDP Growth countries:")
    bottom_growth = df.nsmallest(10, 'GDP Growth')[['name', 'region', 'GDP Growth', 'GDP', 'Inflation Rate']]
    print(bottom_growth.round(2))
    
    print("\nTop 10 GDP countries:")
    top_gdp = df.nlargest(10, 'GDP')[['name', 'region', 'GDP', 'GDP Growth', 'Population']]
    print(top_gdp.round(2))
    
    # 7. Economic Clusters Analysis
    print("\n\n7. ECONOMIC CLUSTERS ANALYSIS")
    print("-" * 40)
    
    try:
        from src.models.clustering import fit_kmeans_pca
        
        # Run clustering
        print("Running clustering analysis...")
        cluster_result = fit_kmeans_pca(df, k=4, random_state=42)
        
        # Display cluster profiles
        print("\nCluster Profiles:")
        print(cluster_result.cluster_profiles.round(2))
        
        # Summary statistics by cluster
        cluster_summary = cluster_result.pca_coords.groupby('cluster').agg({
            'name': 'count',
            'GDP': ['mean', 'median'],
            'GDP Growth': ['mean', 'median'],
            'Inflation Rate': ['mean', 'median'],
            'Debt/GDP': ['mean', 'median']
        }).round(2)
        
        print("\nCluster Summary:")
        print(cluster_summary)
        
        # Regional distribution within clusters
        cluster_region = pd.crosstab(cluster_result.pca_coords['cluster'], 
                                    cluster_result.pca_coords['region'], 
                                    margins=True)
        print("\nRegional distribution within clusters:")
        print(cluster_region)
        
        # Create cluster visualization
        fig = px.scatter(
            cluster_result.pca_coords, 
            x='PC1', 
            y='PC2', 
            color='cluster',
            hover_data=['name', 'region'],
            title='Country Clusters based on Economic Indicators'
        )
        fig.write_html('scripts/clusters_visualization.html')
        print("\nCluster visualization saved as 'scripts/clusters_visualization.html'")
        
    except Exception as e:
        print(f"Clustering analysis failed: {e}")
    
    # 8. Key Insights Summary
    print("\n\n8. KEY INSIGHTS SUMMARY")
    print("-" * 40)
    
    print("Key Findings:")
    print("1. Regional Patterns: Different regions show distinct economic characteristics")
    print("2. GDP Growth Drivers: Key correlations with inflation, interest rates, and debt levels")
    print("3. Top Performers: Countries with highest GDP growth and economic output")
    print("4. Economic Relationships: How different indicators relate to each other")
    print("5. Outliers: Several countries show extreme values in various indicators")
    print("6. Clusters: Countries naturally group into distinct economic profiles")
    
    print("\nRecommendations for Further Analysis:")
    print("1. Investigate specific outlier countries for case studies")
    print("2. Analyze temporal trends if historical data becomes available")
    print("3. Build predictive models for GDP growth forecasting")
    print("4. Conduct scenario analysis for policy implications")
    
    print("\nNext Steps:")
    print("1. Implement the findings in the Streamlit application")
    print("2. Add interactive visualizations for deeper exploration")
    print("3. Develop predictive models with feature importance analysis")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("- scripts/regional_analysis.png")
    print("- scripts/distributions.png")
    print("- scripts/correlation_matrix.png")
    print("- scripts/scatter_plots.png")
    print("- scripts/clusters_visualization.html")

if __name__ == "__main__":
    main()
