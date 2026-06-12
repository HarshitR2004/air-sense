import streamlit as st
import pandas as pd
import numpy as np
import textwrap
from dashboard.components.charts import create_gauge_chart
from src.config import PK, MODELS_DIR, FEATURES_DIR
from src.inference import get_customer_explanation, generate_customer_explanation_text
from src.utils import load_model

# Page headers
st.markdown('<div class="dashboard-header">Customer 360 View</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Individual Customer Dossier, Behavioral Diagnostics, and Local SHAP Explanations</div>', unsafe_allow_html=True)

# Guard logic for missing session state
if 'df' not in st.session_state:
    st.warning('Please load the main app first.')
    st.stop()

df = st.session_state['df']

# Select search customer input widget
st.markdown('### Search Customer Database')
loyalty_ids = sorted(df[PK].unique().tolist())
customer_id_input = st.selectbox('Select Loyalty Number', options=loyalty_ids, index=0, help="Search or select a customer's Loyalty Number from the database.")
cust_data_df = df[df[PK] == customer_id_input]

if cust_data_df.empty:
    st.error(f'Loyalty Number `{customer_id_input}` not found in the database. Please try another ID.')
    st.stop()

customer = cust_data_df.iloc[0]
st.divider()

# Left-Right columns layout (Profile / Actions)
col_left, col_right = st.columns([1, 2])
with col_left:
    gender_icon = 'Male' if customer.get('Gender') == 'Male' else 'Female'
    
    with st.container(border=True):
        st.subheader('Profile Dossier')

        # Format and render a key-value row inside the dossier profile card
        def render_row(label, val):
            c1, c2 = st.columns([1, 1])
            c1.markdown(f'**{label}**')
            c2.text(val)
            
        render_row('Loyalty ID', str(customer[PK]))
        render_row('Loyalty Card', f"{customer.get('Loyalty Card', 'Star')} Card")
        render_row('Gender', gender_icon)
        render_row('Location', f"{customer.get('City')}, {customer.get('Province')}")
        render_row('Education', str(customer.get('Education')))
        render_row('Salary', f"${customer.get('Salary', 0):,.0f}")
        render_row('Marital Status', str(customer.get('Marital Status')))
        render_row('Tenure', f"{customer.get('Tenure_Months')} Months")
        render_row('Historical CLV', f"${customer.get('CLV', 0):,.2f}")
        render_row('Segment', str(customer.get('Segment')))
        
    # Radial gauge indicator layout
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        risk_pct = customer['Churn_Probability']
        fig_g1 = create_gauge_chart(risk_pct, 'Churn Risk', range_max=1.0, color='#f43f5e')
        st.plotly_chart(fig_g1, width='stretch')
        
    with col_g2:
        fvs_val = customer['FVS']
        fig_g2 = create_gauge_chart(fvs_val, 'Future Value Score', range_max=100.0, color='#8b5cf6')
        st.plotly_chart(fig_g2, width='stretch')
        
with col_right:
    priority = customer.get('Priority', 'LOW')
    actions_list = str(customer.get('All_Actions', '')).split(' | ')
    
    with st.container(border=True):
        st.subheader('Retention Prescription')
        if priority == 'CRITICAL':
            st.error('CRITICAL PRIORITY — Immediate save actions required')
        elif priority == 'HIGH':
            st.warning('HIGH PRIORITY — Active save recommended')
        elif priority == 'MEDIUM':
            st.info('MEDIUM PRIORITY — Standard campaign target')
        else:
            st.success('LOW PRIORITY — Monitor engagement status')
            
        st.markdown('**Recommended Interventions**')
        for act in actions_list:
            if act.strip():
                st.markdown(f'- {act.strip()}')
        st.divider()
        
        c1, c2 = st.columns(2)
        c1.metric(label='Campaign Urgency', value=customer.get('Urgency'))
        c2.metric(label='Action Cost Estimate', value=f"${customer.get('Estimated_Cost', 0):,.0f}")
        
    st.markdown('### Individual Churn Diagnostics (Local SHAP)')
    st.markdown("\n        The chart below shows how specific behaviors either **increase (red)** or **decrease (blue/green)** \n        this customer's churn risk relative to the average.\n        ")

    # Load modeling data and compute localized customer SHAP metrics with caching
    @st.cache_data(show_spinner='Computing diagnostic explainers...')
    def cached_local_explanation(cust_id):
        try:
            model = load_model(MODELS_DIR / 'churn_model.joblib')
            feat_df = pd.read_csv(FEATURES_DIR / 'customer_features.csv')
            feature_cols = pd.read_csv(MODELS_DIR / 'feature_columns.csv').iloc[:, 0].tolist()
            cust_idx = feat_df[feat_df[PK] == cust_id].index[0]
            X = feat_df[feature_cols]
            explanation = get_customer_explanation(X.iloc[cust_idx], model)
            return explanation['explanation'].head(8)
        except Exception as e:
            st.error(f'Error computing local diagnostics: {e}')
            return None
            
    local_explanation_df = cached_local_explanation(customer_id_input)
    if local_explanation_df is not None:
        local_plot_df = local_explanation_df.sort_values('SHAP_Value', ascending=True)
        local_plot_df['Influence'] = local_plot_df['SHAP_Value'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
        
        import plotly.express as px
        fig_local = px.bar(local_plot_df, x='SHAP_Value', y='Business_Name', orientation='h', color='Influence', color_discrete_map={'Increases Risk': '#f43f5e', 'Decreases Risk': '#10b981'}, hover_data=['Value'], labels={'SHAP_Value': 'Attribution Impact', 'Business_Name': 'Behavioral Factor'})
        fig_local.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1', family='Inter, sans-serif'), margin=dict(l=20, r=20, t=10, b=40), legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'), xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'), yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)'))
        st.plotly_chart(fig_local, width='stretch')
    else:
        st.info('Diagnostic explainer is currently unavailable.')