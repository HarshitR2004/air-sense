import streamlit as st
import pandas as pd
import numpy as np
import textwrap
from dashboard.components.kpi_cards import kpi_card
from src.config import PK

# Render header details
st.markdown('<div class="dashboard-header">Retention Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Prioritized Customer Interventions, Action Triggers, and Marketing CRM Exports</div>', unsafe_allow_html=True)

# Guard logic for missing session state
if 'df' not in st.session_state:
    st.warning('Please load the main app first.')
    st.stop()

df = st.session_state['df']
filtered_df = df

# Search/Filter target control inputs
st.markdown('### Actionable Customer List')
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    priorities = ['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    selected_priority = st.selectbox('Filter by Action Priority', options=priorities, index=0)
    
with col_f2:
    segments = ['ALL'] + sorted(filtered_df['Segment'].dropna().unique().tolist())
    selected_segment = st.selectbox('Filter by Segment Group', options=segments, index=0)
    
with col_f3:
    search_query = st.text_input('Search Customer by Loyalty ID', placeholder='Enter Loyalty Number...')

# Apply filter selections
list_df = filtered_df.copy()
if selected_priority != 'ALL':
    list_df = list_df[list_df['Priority'] == selected_priority]
if selected_segment != 'ALL':
    list_df = list_df[list_df['Segment'] == selected_segment]
if search_query:
    try:
        loyalty_num = int(search_query.strip())
        list_df = list_df[list_df[PK] == loyalty_num]
    except ValueError:
        st.error('Please enter a valid numeric Loyalty Number.')

# Calculate aggregated savings impact metrics
total_targets = len(list_df)
critical_targets = len(list_df[list_df['Priority'] == 'CRITICAL'])
total_budget = list_df['Estimated_Cost'].sum()
avg_lift = list_df['Expected_Retention_Lift'].mean() * 100 if total_targets > 0 and 'Expected_Retention_Lift' in list_df.columns else 0

# Render KPI row overview
col_k1, col_k2, col_k3, col_k4 = st.columns(4)
with col_k1:
    kpi_card('Targets Identified', f'{total_targets:,}', glow_color='#3b82f6', icon=None)
with col_k2:
    kpi_card('Critical Saves', f'{critical_targets:,}', glow_color='#f43f5e', icon=None)
with col_k3:
    kpi_card('Intervention Cost', f'${total_budget:,.0f}', glow_color='#f59e0b', icon=None)
with col_k4:
    kpi_card('Expected Success Lift', f'{avg_lift:.1f}%', glow_color='#10b981', icon=None)

st.markdown('<br>', unsafe_allow_html=True)

# Prepare DataFrame table presentation
display_cols = [PK, 'Segment', 'FVS', 'Churn_Probability', 'Priority', 'Primary_Action', 'Urgency', 'Estimated_Cost']
rename_map = {PK: 'Loyalty ID', 'Segment': 'Segment', 'FVS': 'FVS', 'Churn_Probability': 'Churn Risk', 'Priority': 'Priority', 'Primary_Action': 'Primary Action', 'Urgency': 'Urgency Label', 'Estimated_Cost': 'Cost Estimate ($)'}
table_df = list_df[display_cols].copy()
table_df['Churn_Probability'] = (table_df['Churn_Probability'] * 100).round(1).apply(lambda x: f'{x}%')
table_df = table_df.rename(columns=rename_map)

# Sort by priority order
priority_weights = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
table_df['sort_idx'] = table_df['Priority'].map(priority_weights)
table_df = table_df.sort_values('sort_idx').drop(columns=['sort_idx'])

# Render interactive table
st.dataframe(table_df, width='stretch', hide_index=True)
st.markdown('<br>', unsafe_allow_html=True)

# CRM Action buttons (Export / Tip)
col_exp1, col_exp2 = st.columns([1, 4])
with col_exp1:
    csv_data = list_df.to_csv(index=False).encode('utf-8')
    st.download_button(label='Export to CRM (CSV)', data=csv_data, file_name='retention_targets.csv', mime='text/csv', type='primary', help='Export the filtered list to a CSV file for import into your CRM platform.')
    
with col_exp2:
    st.markdown(textwrap.dedent('\n        <div style="color: #64748b; font-size: 0.85rem; padding-top: 10px;">\n            <strong>CRM Integration Tip</strong>: Upload this list of target members directly into CRM marketing campaigns. \n            VIP At Risk profiles have pre-mapped high-value perks designed to prevent churn, prioritizing the preservation of airline revenue.\n        </div>\n        '), unsafe_allow_html=True)