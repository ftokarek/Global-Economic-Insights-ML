import streamlit as st
import pandas as pd

from src.data.loaders import load_dataset, list_countries

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
        cdf = df[df['name'] == sel]
        st.write('Basics', cdf[['region','subregion','Population','GDP','GDP Growth']].head(1))

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
