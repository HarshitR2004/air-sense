import streamlit as st
import pandas as pd
import numpy as np
import textwrap
from dashboard.components.kpi_cards import metric_row
from dashboard.components.charts import create_donut_chart, create_bar_chart, create_line_chart
from src.config import PK

# Render page headers
st.markdown('<div class="dashboard-header">Executive Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">FlightSense Loyalty Program Health & Revenue at Risk Control Board</div>', unsafe_allow_html=True)

# Guard logic for missing session state
if 'df' not in st.session_state:
    st.warning('Please load the main app first.')
    st.stop()

# Retrieve loaded datasets
df = st.session_state['df']
meta = st.session_state['meta']
filtered_df = df

# Compute high-level KPI metrics
total_customers = len(filtered_df)
avg_churn_risk = filtered_df['Churn_Probability'].mean() * 100 if total_customers > 0 else 0
at_risk_mask = filtered_df['Churn_Probability'] > 0.5
revenue_at_risk = filtered_df[at_risk_mask]['CLV'].sum() if total_customers > 0 else 0
avg_fvs = filtered_df['FVS'].mean() if total_customers > 0 else 0

# Baseline comparisons
base_total = len(df)
base_churn = df['Churn_Probability'].mean() * 100
base_rev_at_risk = df[df['Churn_Probability'] > 0.5]['CLV'].sum()
base_avg_fvs = df['FVS'].mean()

# Format metrics dictionary
metrics = [
    {'title': 'Total Customers', 'value': f'{total_customers:,}', 'delta': f'{total_customers - base_total:+,} from base' if total_customers != base_total else None, 'delta_type': 'neutral', 'icon': None, 'glow_color': '#3b82f6'},
    {'title': 'Avg Churn Risk', 'value': f'{avg_churn_risk:.1f}%', 'delta': f'{avg_churn_risk - base_churn:+.2f}% vs base' if total_customers != base_total else None, 'delta_type': 'negative' if avg_churn_risk > base_churn else 'positive', 'icon': None, 'glow_color': '#f43f5e'},
    {'title': 'Revenue At Risk', 'value': f'${revenue_at_risk:,.0f}', 'delta': f'${revenue_at_risk - base_rev_at_risk:+,.0f} vs base' if total_customers != base_total else None, 'delta_type': 'negative' if revenue_at_risk > base_rev_at_risk else 'positive', 'icon': None, 'glow_color': '#f59e0b'},
    {'title': 'Average FVS', 'value': f'{avg_fvs:.1f} / 100', 'delta': f'{avg_fvs - base_avg_fvs:+.1f} vs base' if total_customers != base_total else None, 'delta_type': 'positive' if avg_fvs > base_avg_fvs else 'negative', 'icon': None, 'glow_color': '#8b5cf6'}
]

# Display high level KPI row
metric_row(metrics)
st.markdown('<br>', unsafe_allow_html=True)

# Render first row of charts (Donut Segment Dist / FVS Bins Dist)
col1, col2 = st.columns(2)
with col1:
    if 'Segment' in filtered_df.columns:
        segment_counts = filtered_df['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Customers']
        fig_segment = create_donut_chart(segment_counts, 'Customers', 'Segment', 'Loyalty Segment Distribution')
        st.plotly_chart(fig_segment, width='stretch')
    else:
        st.info('Segment data not found.')
        
with col2:
    if 'FVS' in filtered_df.columns:
        fvs_bins = pd.cut(filtered_df['FVS'], bins=[0, 20, 40, 60, 80, 100], labels=['Very Low (0-20)', 'Low (20-40)', 'Medium (40-60)', 'High (60-80)', 'Very High (80-100)'], include_lowest=True)
        fvs_dist = fvs_bins.value_counts().reset_index()
        fvs_dist.columns = ['FVS Range', 'Customers']
        fvs_dist['sort_idx'] = fvs_dist['FVS Range'].cat.codes
        fvs_dist = fvs_dist.sort_values('sort_idx')
        fig_fvs = create_bar_chart(fvs_dist, 'FVS Range', 'Customers', 'Future Value Score (FVS) Distribution', color_col='FVS Range')
        st.plotly_chart(fig_fvs, width='stretch')
    else:
        st.info('FVS data not found.')

# Render second row (Average Churn by Tier / Executive Insights summary text)
col3, col4 = st.columns(2)
with col3:
    if 'Loyalty Card' in filtered_df.columns and 'Churn_Probability' in filtered_df.columns:
        tier_churn = filtered_df.groupby('Loyalty Card')['Churn_Probability'].mean().reset_index()
        tier_churn['Churn Rate %'] = (tier_churn['Churn_Probability'] * 100).round(1)
        tier_order = {'Star': 0, 'Nova': 1, 'Aurora': 2}
        tier_churn['sort_idx'] = tier_churn['Loyalty Card'].map(tier_order)
        tier_churn = tier_churn.sort_values('sort_idx')
        fig_tier = create_bar_chart(tier_churn, 'Loyalty Card', 'Churn Rate %', 'Average Churn Risk by Loyalty Tier', color_col='Loyalty Card')
        st.plotly_chart(fig_tier, width='stretch')
    else:
        st.info('Tier and Churn data not found.')
        
with col4:
    st.markdown(textwrap.dedent(f"""\n        <div style="\n            background: rgba(30, 41, 59, 0.7);\n            backdrop-filter: blur(10px);\n            border: 1px solid rgba(255, 255, 255, 0.08);\n            border-radius: 12px;\n            padding: 24px;\n            height: 380px;\n            overflow-y: auto;\n        ">\n            <h3 style="color: #cbd5e1; margin-top: 0; margin-bottom: 15px; font-size: 1.2rem; font-weight: 700;">Executive Insights</h3>\n            <ul style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6; padding-left: 20px;">\n                <li style="margin-bottom: 10px;">\n                    <strong style="color: #f8fafc;">High Risk Concentration:</strong> Approximately \n                    <span style="color: #f43f5e; font-weight: 600;">{len(filtered_df[at_risk_mask]):,}</span> customers are currently identified as high churn risk (probability > 50%), representing a total value of \n                    <span style="color: #f43f5e; font-weight: 600;">${revenue_at_risk:,.0f}</span> in CLV.\n                </li>\n                <li style="margin-bottom: 10px;">\n                    <strong style="color: #f8fafc;">Best Performing Model:</strong> Model predictions generated by \n                    <span style="color: #3b82f6; font-weight: 600;">{meta.get('model_name', 'LightGBM')}</span> (Tuned) with a test Precision-Recall AUC of \n                    <span style="color: #10b981; font-weight: 600;">{meta.get('pr_auc', 0.85):.3f}</span>.\n                </li>\n                <li style="margin-bottom: 10px;">\n                    <strong style="color: #f8fafc;">Value Score Divergence:</strong> Traditional CLV is backward-looking. Our forward-looking <strong>Future Value Score (FVS)</strong> integrates tenure, recent flight cadence, and redemption behavior to catch declining VIPs early.\n                </li>\n                <li style="margin-bottom: 10px;">\n                    <strong style="color: #f8fafc;">Program Activity:</strong> Segment Champions show high FVS and low churn risk, while VIP At Risk represents critical customer saves that the Retention Command Center should target immediately.\n                </li>\n            </ul>\n        </div>\n        """), unsafe_allow_html=True)