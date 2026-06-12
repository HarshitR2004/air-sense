import streamlit as st
import pandas as pd
import numpy as np
import textwrap
from dashboard.components.charts import create_bar_chart, create_line_chart
from src.inference import get_global_feature_importance
from src.config import MODELS_DIR, FEATURES_DIR, PK
from src.utils import load_model

# Page titles
st.markdown('<div class="dashboard-header">Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Machine Learning Model Insights & Key Drivers of Customer Attrition</div>', unsafe_allow_html=True)

# Session state security guard
if 'df' not in st.session_state:
    st.warning('Please load the main app first.')
    st.stop()

df = st.session_state['df']
meta = st.session_state['meta']
filtered_df = df

# Metric performance indicators section
st.markdown('### Churn Prediction Model Performance')
col_acc, col_prec, col_rec, col_f1, col_roc, col_pr = st.columns(6)

with col_acc:
    st.metric(label='Accuracy', value=f"{meta.get('accuracy', 0.0) * 100:.1f}%")
with col_prec:
    st.metric(label='Precision', value=f"{meta.get('precision', 0.0) * 100:.1f}%")
with col_rec:
    st.metric(label='Recall', value=f"{meta.get('recall', 0.0) * 100:.1f}%")
with col_f1:
    st.metric(label='F1 Score', value=f"{meta.get('f1', 0.0) * 100:.1f}%")
with col_roc:
    st.metric(label='ROC AUC', value=f"{meta.get('roc_auc', 0.0):.3f}")
with col_pr:
    st.metric(label='Precision-Recall AUC', value=f"{meta.get('pr_auc', 0.0):.3f}")

st.markdown(f"\n    > **Model Architecture**: The system selected **{meta.get('model_name', 'LightGBM (Tuned)')}** as the optimal model based on **Precision-Recall AUC**, which is the industry standard metric for class-imbalanced datasets (e.g. churn rate of ~17-20%).\n    ")
st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)

# Multi-column chart layout (Drivers / Distribution)
col1, col2 = st.columns(2)

with col1:
    st.markdown('### Top Churn Drivers')
    st.markdown('\n        Longer bars indicate features that have a stronger influence on loyalty disengagement.\n        ')

    # Load machine learning model and compute feature SHAP importance with caching
    @st.cache_data(show_spinner='Computing global SHAP feature importance...')
    def cached_feature_importance():
        try:
            model = load_model(MODELS_DIR / 'churn_model.joblib')
            feat_df = pd.read_csv(FEATURES_DIR / 'customer_features.csv')
            feature_cols = pd.read_csv(MODELS_DIR / 'feature_columns.csv').iloc[:, 0].tolist()
            X = feat_df[feature_cols]
            importance_df = get_global_feature_importance(model, X, max_samples=250)
            return importance_df.head(10)
        except Exception as e:
            st.error(f'Error computing SHAP values: {e}')
            return None

    importance_df = cached_feature_importance()
    if importance_df is not None:
        importance_plot = importance_df.sort_values('Mean_SHAP', ascending=True)
        import plotly.express as px
        fig_importance = px.bar(importance_plot, x='Mean_SHAP', y='Business_Name', orientation='h', color='Mean_SHAP', color_continuous_scale='Reds', labels={'Mean_SHAP': 'Average Impact on Churn Risk', 'Business_Name': 'Driver'})
        fig_importance.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1', family='Inter, sans-serif'), margin=dict(l=20, r=20, t=10, b=40), coloraxis_showscale=False, xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'), yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'))
        st.plotly_chart(fig_importance, width='stretch')
    else:
        st.info('Feature importance data is currently unavailable.')

with col2:
    st.markdown('### Churn Risk Probability Distribution')
    st.markdown('\n        Below is the distribution of predicted churn probabilities across the active user base.\n        Customers above the 50% threshold are classified as at-risk.\n        ')
    
    probs = filtered_df['Churn_Probability']
    import plotly.figure_factory as ff
    hist_data = [probs]
    group_labels = ['Churn Probability']
    import plotly.graph_objects as go
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=probs, nbinsx=40, name='Customers', marker_color='#f43f5e', opacity=0.8, xbins=dict(start=0.0, end=1.0, size=0.025)))
    fig_dist.add_vline(x=0.5, line_width=3, line_dash='dash', line_color='#e2e8f0', annotation_text='Decision Boundary (50% Risk)', annotation_position='top right', annotation_font_color='#e2e8f0')
    fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1', family='Inter, sans-serif'), margin=dict(l=40, r=40, t=20, b=40), xaxis=dict(title='Predicted Churn Probability', gridcolor='rgba(255, 255, 255, 0.05)', range=[0, 1]), yaxis=dict(title='Number of Customers', gridcolor='rgba(255, 255, 255, 0.05)'))
    st.plotly_chart(fig_dist, width='stretch')

# Context info cards for qualitative drivers
st.markdown('### What Drives Loyalty Churn in Our Program?')
col_drv1, col_drv2, col_drv3 = st.columns(3)

with col_drv1:
    st.markdown(textwrap.dedent('\n        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; height: 180px;">\n            <b style="color: #f43f5e; font-size: 1.05rem;">1. Recency & Inactivity Cadence</b>\n            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">\n                The most dominant risk factor is <strong>Months Inactive</strong> and <strong>Days since last flight</strong>. \n                Loyalty members who cease points earning or flight activities for over 3-6 months exhibit an exponential increase in churn risk.\n            </p>\n        </div>\n        '), unsafe_allow_html=True)

with col_drv2:
    st.markdown(textwrap.dedent('\n        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; height: 180px;">\n            <b style="color: #3b82f6; font-size: 1.05rem;">2. Earning & Engagement Trend</b>\n            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">\n                A negative slope in <strong>Engagement Trend</strong> captures declining activity. \n                Even if a customer has high lifetime flights (CLV), a decline in their rolling 3-month or 6-month activity triggers strong churn predictions.\n            </p>\n        </div>\n        '), unsafe_allow_html=True)

with col_drv3:
    st.markdown(textwrap.dedent('\n        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 15px; height: 180px;">\n            <b style="color: #8b5cf6; font-size: 1.05rem;">3. Redemption & Sticky Behaviors</b>\n            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">\n                Customers with low <strong>Redemption Ratios</strong> (accumulating points but never redeeming them) are highly susceptible to silent churn. \n                Points redemptions are highly sticky behaviors that keep members locked into the brand.\n            </p>\n        </div>\n        '), unsafe_allow_html=True)