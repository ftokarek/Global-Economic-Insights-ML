import streamlit as st
import pandas as pd
import sys
import os
import time

# Ensure project root is on sys.path so imports work when running from app/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loaders import load_dataset, list_countries
from src.features.benchmarks import attach_benchmarks

# Performance optimizations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_data():
    """Cache the main dataset with benchmarks"""
    df = load_dataset()
    df_with_benchmarks = attach_benchmarks(df)
    return df_with_benchmarks

@st.cache_data(ttl=3600)
def get_cached_countries(df):
    """Cache country list"""
    return list_countries(df)

@st.cache_resource(ttl=1800)  # Cache for 30 minutes
def get_cached_models(df):
    """Cache trained models"""
    from src.models.train import train_elasticnet, train_lightgbm
    elasticnet_result = train_elasticnet(df)
    lightgbm_result = train_lightgbm(df)
    return elasticnet_result, lightgbm_result

st.set_page_config(page_title='Global Economic Insights 2025', page_icon='🌍', layout='wide')

st.title('Global Economic Insights 2025')
st.caption('Interactive cross-sectional analysis, clustering, and predictive insights for 2025.')

# Load data with caching
with st.spinner('Loading dataset...'):
    df = get_cached_data()
    countries = get_cached_countries(df)

if df.empty:
    st.error('No dataset found. Please ensure processed or raw CSV is present in data/.')
    st.stop()

if 'source' in df.columns and df['source'].iloc[0] == 'raw':
    st.warning('Using RAW dataset. Cleaning pipeline outputs not found yet. Some features may be limited.')

# Navigation
tabs = st.tabs(['Country', 'Predictions', 'Clusters', 'What-if', 'Methodology'])

with tabs[0]:
    st.subheader('Country Profile')
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
    st.caption('Cross-validated metrics and feature importances')
    from src.models.train import train_elasticnet, train_lightgbm

    n_obs = int(df.dropna(subset=['GDP Growth']).shape[0])
    st.write(f"Samples used (non-missing target): {n_obs}")

    # Glossary expander
    with st.expander('Glossary of indicators'):
        glossary = pd.DataFrame([
            {'Indicator':'GDP','Description':'Gross Domestic Product, USD billions (size of economy).'},
            {'Indicator':'GDP Growth','Description':'Real GDP growth, % y/y (target variable).'},
            {'Indicator':'Interest Rate','Description':'Policy interest rate, %.'},
            {'Indicator':'Inflation Rate','Description':'Consumer price inflation, % y/y.'},
            {'Indicator':'Jobless Rate','Description':'Unemployment rate, %.'},
            {'Indicator':'Gov. Budget','Description':'General government balance, % of GDP (surplus + / deficit -).'},
            {'Indicator':'Debt/GDP','Description':'Public debt, % of GDP.'},
            {'Indicator':'Current Account','Description':'External current account balance, % of GDP.'},
            {'Indicator':'Population','Description':'Population in millions.'},
        ])
        st.dataframe(glossary, hide_index=True)

    if st.button('Train models'):
        enet = None
        lgbm = None
        with st.spinner('Training models...'):
            try:
                enet, lgbm = get_cached_models(df)
            except Exception as e:
                st.warning(f'Model training failed: {e}')
                # Fallback to individual training
                try:
                    enet = train_elasticnet(df)
                except Exception as e2:
                    st.error(f'ElasticNet training failed: {e2}')
                try:
                    lgbm = train_lightgbm(df)
                except Exception as e3:
                    st.warning(f'LightGBM unavailable: {e3}')
        # Metrics cards with tooltips
        c1, c2, c3 = st.columns(3)
        c1.metric('ElasticNet RMSE', f"{enet.cv_metrics['RMSE']:.2f}", help='Root Mean Squared Error (lower is better).')
        c2.metric('ElasticNet MAE', f"{enet.cv_metrics['MAE']:.2f}", help='Mean Absolute Error (lower is better).')
        c3.metric('ElasticNet R²', f"{enet.cv_metrics['R2']:.2f}", help='Explained variance (1=perfect, 0=mean baseline).')
        if enet.cv_metrics['R2'] < 0:
            st.info('R² < 0 indicates poor explanatory power in this snapshot; consider additional features or different modeling.')
        if lgbm is not None:
            d1, d2, d3 = st.columns(3)
            d1.metric('LightGBM RMSE', f"{lgbm.cv_metrics['RMSE']:.2f}", help='Root Mean Squared Error (lower is better).')
            d2.metric('LightGBM MAE', f"{lgbm.cv_metrics['MAE']:.2f}", help='Mean Absolute Error (lower is better).')
            d3.metric('LightGBM R²', f"{lgbm.cv_metrics['R2']:.2f}", help='Explained variance (1=perfect, 0=mean baseline).')
            if lgbm.importances is not None:
                imp = pd.DataFrame({'feature': lgbm.feature_names, 'importance': lgbm.importances}).sort_values('importance', ascending=False)
                st.markdown('**LightGBM feature importances**')
                st.bar_chart(imp.set_index('feature'))
        # Interpretation box
        st.subheader('Interpretation')
        st.markdown('- Metrics compare models via cross-validation on the 2025 cross-section.\n- Low/negative R² is expected on a single-year snapshot; importances show which inputs the non-linear model uses most.\n- Use the What-if tab to see how predicted growth reacts to small changes in inputs (not causal).')
    else:
        st.info('Click Train models to compute metrics.')

with tabs[2]:
    st.subheader('Clusters')
    from src.models.clustering import fit_kmeans_pca
    import plotly.express as px

    k = st.slider('Number of clusters (k)', 3, 8, 5)
    rs = st.number_input('Random state', value=42, step=1)
    if st.button('Run clustering'):
        with st.spinner('Clustering countries...'):
            res = fit_kmeans_pca(df, k=k, random_state=int(rs))
        st.markdown('**PCA scatter by cluster**')
        fig = px.scatter(
            res.pca_coords,
            x='PC1', y='PC2', color=res.pca_coords['cluster'].astype(str),
            hover_name='name', hover_data=['region','subregion']
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('**Cluster profiles (median values)**')
        st.dataframe(res.profile)
    else:
        st.info('Choose k and click Run clustering to see clusters.')

with tabs[3]:
    st.subheader('What-if')
    st.caption('Adjust key indicators to simulate changes in predicted GDP Growth (not causal).')

    # Robust import of training helpers
    try:
        from src.models import train as mtrain
        fit_lightgbm_full = getattr(mtrain, 'fit_lightgbm_full', None)
        fit_elasticnet_full = getattr(mtrain, 'fit_elasticnet_full', None)
        NUMERIC_COLS = mtrain.NUMERIC_COLS
    except Exception:
        fit_lightgbm_full = None
        fit_elasticnet_full = None
        from src.models.train import NUMERIC_COLS  # type: ignore

    # Choose a baseline country
    countries = list_countries(df)
    base_country = st.selectbox('Baseline country', countries, index=countries.index('Poland') if 'Poland' in countries else 0)
    base_row = df[df['name'] == base_country].iloc[0]

    # Slider ranges based on global percentiles for safety
    def pct_range(col, lo=5, hi=95):
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        return float(s.quantile(lo/100.0)), float(s.quantile(hi/100.0))

    cols = ['Inflation Rate','Interest Rate','Debt/GDP','Gov. Budget','Current Account']
    values = {}
    g1,g2 = st.columns(2)
    with g1:
        for col in cols[:3]:
            lo, hi = pct_range(col)
            values[col] = st.slider(f'{col}', lo, hi, float(base_row[col]))
    with g2:
        for col in cols[3:]:
            lo, hi = pct_range(col)
            values[col] = st.slider(f'{col}', lo, hi, float(base_row[col]))

    # Construct scenario input using baseline for other features
    scenario = {c: float(base_row[c]) if c in df.columns else 0.0 for c in NUMERIC_COLS}
    scenario.update(values)
    scenario_df = pd.DataFrame([scenario])

    # Fit model (prefer LightGBM)
    model = None
    if fit_lightgbm_full is not None:
        try:
            model = fit_lightgbm_full(df)
        except Exception:
            model = None
    if model is None and fit_elasticnet_full is not None:
        model = fit_elasticnet_full(df)

    if model is None:
        st.error('No model available to run the scenario. Please train dependencies or try again.')
    else:
        # Predict baseline and scenario
        baseline_input = pd.DataFrame([{c: float(base_row[c]) for c in NUMERIC_COLS}])
        baseline_pred = float(model.predict(baseline_input)[0])
        scenario_pred = float(model.predict(scenario_df)[0])
        delta = scenario_pred - baseline_pred

        c1,c2,c3 = st.columns(3)
        c1.metric('Baseline GDP Growth %', f'{baseline_pred:.2f}')
        c2.metric('Scenario GDP Growth %', f'{scenario_pred:.2f}')
        c3.metric('Delta (pp)', f'{delta:+.2f}')
        st.info('These are model-based effects in a cross-sectional snapshot; interpret as directional, not causal.')

with tabs[4]:
    st.subheader('Methodology & Limitations')
    st.markdown('- Cross-sectional 2025 snapshot.\n- Cleaning: type coercion, regional/global median imputation.\n- Benchmarks: region, income group (planned), and clusters (planned).')
    
    # Performance monitoring
    st.subheader('Performance')
    st.markdown(f'- **Dataset size**: {len(df)} countries, {len(df.columns)} features')
    st.markdown(f'- **Memory usage**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
    st.markdown(f'- **Caching**: Data cached for 1 hour, models for 30 minutes')
    st.markdown('- **Optimizations**: Lazy loading, cached computations, efficient data structures')
