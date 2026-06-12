import streamlit as st
import pandas as pd
import numpy as np
from dashboard.components.charts import create_scatter_chart, create_radar_chart
from dashboard.components.filters import render_sidebar_filters
from src.config import SEGMENT_NAMES

# Page titles
st.markdown('<div class="dashboard-header">Customer Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Strategic Micro-Segmentation & Behavioral Profile Analysis</div>', unsafe_allow_html=True)

# Guard logic for missing session state
if 'df' not in st.session_state:
    st.warning('Please load the main app first.')
    st.stop()

df = st.session_state['df']

# Sidebar localized filters
filtered_df = render_sidebar_filters(df)

# Render main scatter cluster visualization space
st.markdown('### Loyalty Segment Scatter Space')
st.markdown('\n    Each dot represents an individual customer. The horizontal axis represents their **Future Value Score (FVS)** (forward-looking value potential), \n    and the vertical axis represents their **Predicted Churn Risk** (model probability).\n    Customers are grouped into 5 distinct behavioral segments.\n    ')

if 'Segment' in filtered_df.columns and 'FVS' in filtered_df.columns:
    import plotly.express as px
    segment_colors = {'Champions': '#10b981', 'VIP At Risk': '#f43f5e', 'Loyal Travelers': '#3b82f6', 'Growth Potential': '#f59e0b', 'Dormant Members': '#64748b'}
    sample_size = min(len(filtered_df), 3000)
    plot_sample = filtered_df.sample(sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df
    
    fig_scatter = px.scatter(plot_sample, x='FVS', y='Churn_Probability', color='Segment', color_discrete_map=segment_colors, hover_data=['Loyalty Number', 'Loyalty Card', 'CLV', 'Flights_12M', 'Tenure_Months'], opacity=0.6, size_max=12, labels={'Churn_Probability': 'Churn Risk Probability', 'FVS': 'Future Value Score (FVS)'})
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1', family='Inter, sans-serif'), legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'), xaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 100]), yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 1.05]))
    st.plotly_chart(fig_scatter, width='stretch')
else:
    st.info('Segment data not available for visualization.')

st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)

# Render comparison table profiles and radar alignment overlay
st.markdown('### Segment Comparison Profiles')
col_left, col_right = st.columns([3, 2])

with col_left:
    if 'Segment' in filtered_df.columns:
        profile_cols = ['Churn_Probability', 'FVS', 'Flights_12M', 'CLV', 'Tenure_Months']
        segment_profile = filtered_df.groupby('Segment')[profile_cols].mean()
        segment_profile['Count'] = filtered_df.groupby('Segment').size()
        segment_profile['Percentage'] = (segment_profile['Count'] / len(filtered_df) * 100).round(1)
        
        segment_profile['Churn_Probability'] = (segment_profile['Churn_Probability'] * 100).round(1).apply(lambda x: f'{x}%')
        segment_profile['FVS'] = segment_profile['FVS'].round(1)
        segment_profile['Flights_12M'] = segment_profile['Flights_12M'].round(1)
        segment_profile['CLV'] = segment_profile['CLV'].apply(lambda x: f'${x:,.0f}')
        segment_profile['Tenure_Months'] = segment_profile['Tenure_Months'].round(0).astype(int)
        
        display_profile = segment_profile[['Count', 'Percentage', 'FVS', 'Churn_Probability', 'Flights_12M', 'CLV', 'Tenure_Months']]
        st.dataframe(display_profile, width='stretch')
    else:
        st.info('Profile summary not available.')
        
with col_right:
    st.markdown("<b style='color:#cbd5e1; font-size:1.1rem;'>Segment Radar Overlays</b>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-top:-5px;'>Comparison of normalized values across primary metrics.</p>", unsafe_allow_html=True)
    
    if 'Segment' in filtered_df.columns:
        radar_features = ['Churn_Probability', 'FVS', 'Flights_12M', 'Redemption_Ratio']
        profiles_avg = filtered_df.groupby('Segment')[radar_features].mean()
        profiles_norm = (profiles_avg - profiles_avg.min()) / (profiles_avg.max() - profiles_avg.min() + 1e-08)
        
        categories = ['Churn Risk', 'FVS', 'Flights (12M)', 'Redemption Ratio']
        values_dict = {}
        for segment in profiles_norm.index:
            values_dict[segment] = profiles_norm.loc[segment].tolist()
            
        fig_radar = create_radar_chart(categories, values_dict, 'Multi-Segment Profile Alignment')
        st.plotly_chart(fig_radar, width='stretch')
    else:
        st.info('Radar chart data not available.')