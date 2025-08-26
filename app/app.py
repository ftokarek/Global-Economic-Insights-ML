import streamlit as st
import pandas as pd
import sys
import os

# Ensure project root is on sys.path so `src.*` imports work when running from app/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loaders import load_dataset, list_countries
from src.features.benchmarks import attach_benchmarks

st.set_page_config(page_title='Global Economic Insights 2025', page_icon='🌍', layout='wide')

st.title('Global Economic Insights 2025')
st.caption('Interactive cross-sectional analysis, clustering, and predictive insights for 2025.')

# Load data
with st.spinner('Loading dataset...'):
    df = load_dataset()

if df.empty:
    st.error('No dataset found. Please ensure processed or raw CSV is present in data/.')
    st.stop()

if 'source' in df.columns and df['source'].iloc[0] == 'raw':
    st.warning('Using RAW dataset. Cleaning pipeline outputs not found yet. Some features may be limited.')

# Navigation
tabs = st.tabs(['Country', 'Predictions', 'Clusters', 'What-if', 'Methodology'])

with tabs[0]:
    st.subheader('Country Profile')
    countries = list_countries(df)
    sel = st.selectbox('Select a country', countries, index=countries.index('Poland') if 'Poland' in countries else 0)
    if sel:
        bdf = attach_benchmarks(df)
        bc = bdf[bdf['name'] == sel].head(1)
        # KPIs
        c1,c2,c3,c4,c5 = st.columns(5)
        def metric(col, fmt='{:.2f}'):
            val = bc[col].iloc[0] if col in bc.columns and not bc.empty else None
            return '—' if pd.isna(val) else (fmt.format(val) if isinstance(val,(int,float)) else val)
        c1.metric('GDP (USD bn)', metric('GDP','{:.0f}'))
        c2.metric('GDP Growth %', metric('GDP Growth'))
        c3.metric('Inflation %', metric('Inflation Rate'))
        c4.metric('Debt/GDP %', metric('Debt/GDP'))
        c5.metric('Population (m)', metric('Population','{:.2f}'))
        st.divider()
        # Benchmarks
        st.subheader('Benchmarks')
        bench_cols = ['GDP Growth','Inflation Rate','Debt/GDP','Gov. Budget','Current Account']
        rows = []
        for col in bench_cols:
            rc = f"{col}__region_median"
            if col in bc.columns and rc in bc.columns:
                rows.append({
                    'Indicator': col,
                    'Country': bc[col].iloc[0],
                    'Region median': bc[rc].iloc[0]
                })
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info('Benchmarks will appear once processed data is available.')

with tabs[1]:
    st.subheader('Predictions')
    st.info('Model training and SHAP explanations will appear here.')

with tabs[2]:
    st.subheader('Clusters')
    st.info('Clustering visuals will appear here.')

with tabs[3]:
    st.subheader('What-if')
    st.info('Scenario sliders and on-the-fly predictions will appear here.')

with tabs[4]:
    st.subheader('Methodology & Limitations')
    st.markdown('- Cross-sectional 2025 snapshot.\n- Cleaning: type coercion, regional/global median imputation.\n- Benchmarks: region, income group (planned), and clusters (planned).')
