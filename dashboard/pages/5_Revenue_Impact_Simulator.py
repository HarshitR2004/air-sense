import streamlit as st
import pandas as pd
import numpy as np
from dashboard.components.kpi_cards import kpi_card
from src.inference import simulate_retention_impact, calculate_roi

# Page titles
st.markdown('<div class="dashboard-header">Revenue Impact Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Financial Simulation of Churn Mitigation and ROI Forecasting</div>', unsafe_allow_html=True)

# Guard logic for missing session state
if 'df' not in st.session_state:
    st.warning('Please load the main app first.')
    st.stop()

df = st.session_state['df']
filtered_df = df

# Interactive simulation scenario controls
st.markdown('### Simulation Parameters')
col_ctrl1, col_ctrl2 = st.columns(2)

with col_ctrl1:
    retention_improvement = st.slider('Target Churn Reduction (Success Rate %)', min_value=5, max_value=30, value=15, step=5, help='The percentage of at-risk customers (churn risk > 50%) that will be successfully saved by marketing campaigns.')
    
with col_ctrl2:
    campaign_cost_input = st.number_input('Average Campaign Cost per Saved Customer ($)', min_value=10, max_value=1000, value=150, step=10, help='Average dollar budget needed to prevent one customer from churning (includes bonuses, flight perks, tier adjustments).')

# Run impact simulation
sim_df = simulate_retention_impact(filtered_df, improvement_pcts=[retention_improvement / 100.0])
roi_df = calculate_roi(sim_df, cost_per_customer=campaign_cost_input)
scenario_metrics = roi_df.iloc[0]

saved_customers = int(scenario_metrics['Customers_Saved'])
clv_retained = scenario_metrics['CLV_Retained']
campaign_cost = scenario_metrics['Campaign_Cost']
net_revenue = scenario_metrics['Net_Revenue']
roi_pct = scenario_metrics['ROI']

# Render simulation ROI results layout
st.markdown('#### Scenario Financial Impact')
col_k1, col_k3, col_k5 = st.columns(3)

with col_k1:
    kpi_card('Saved Customers', f'{saved_customers:,}', glow_color='#3b82f6', icon=None)
with col_k3:
    kpi_card('Campaign Cost', f'${campaign_cost:,.0f}', glow_color='#f43f5e', icon=None)
with col_k5:
    roi_color = '#10b981' if roi_pct > 0 else '#f43f5e'
    kpi_card('Forecasted ROI', f'{roi_pct:.0f}%', glow_color=roi_color, icon=None)

st.write('')

col_k2, col_k4 = st.columns(2)
with col_k2:
    kpi_card('CLV Retained', f'${clv_retained:,.0f}', glow_color='#10b981', icon=None)
with col_k4:
    kpi_card('Net Return', f'${net_revenue:,.0f}', glow_color='#8b5cf6', icon=None)

st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'><br>", unsafe_allow_html=True)

# Multi-Scenario comparison section
st.markdown('### Multi-Scenario Return Comparisons')
st.markdown('Compare the financial returns across different retention success scenarios (5% to 25%).')

all_scenarios_sim = simulate_retention_impact(filtered_df, improvement_pcts=[0.05, 0.1, 0.15, 0.2, 0.25])
all_scenarios_roi = calculate_roi(all_scenarios_sim, cost_per_customer=campaign_cost_input)

import plotly.graph_objects as go
fig_compare = go.Figure()

# Add metrics traces
fig_compare.add_trace(go.Bar(x=all_scenarios_roi['Retention_Improvement'], y=all_scenarios_roi['Revenue_Protected'], name='Revenue Protected (CLV Retained)', marker_color='#3b82f6'))
fig_compare.add_trace(go.Bar(x=all_scenarios_roi['Retention_Improvement'], y=all_scenarios_roi['Campaign_Cost'], name='Campaign Cost', marker_color='#f43f5e'))
fig_compare.add_trace(go.Scatter(x=all_scenarios_roi['Retention_Improvement'], y=all_scenarios_roi['Net_Revenue'], name='Net Revenue Saved', mode='lines+markers', line=dict(color='#10b981', width=3), marker=dict(size=8)))

# Apply layout updates with correct legend positioning to avoid overlaps
fig_compare.update_layout(
    title='Retention Lift Scenario Matrix',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#cbd5e1', family='Inter, sans-serif'),
    margin=dict(l=40, r=40, t=50, b=80),
    legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
    xaxis=dict(title='Retention Reduction Scenario', gridcolor='rgba(255,255,255,0.05)'),
    yaxis=dict(title='Financial Value ($)', gridcolor='rgba(255,255,255,0.05)'),
    barmode='group'
)
st.plotly_chart(fig_compare, width='stretch')

st.markdown('\n    > [!TIP]\n    > **High-Value Save Priority**: The ROI calculation assumes saving efforts are prioritized by **Future Value Score (FVS)**. \n    > Retaining a customer with high engagement momentum protects a larger slice of prospective CLV, keeping efficiency levels high.\n    ')