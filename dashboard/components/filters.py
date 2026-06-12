import streamlit as st
import pandas as pd
import textwrap

# Render sidebar filter widgets and return the filtered dataframe
def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown(textwrap.dedent('\n        <div style="text-align: center; margin-bottom: 20px;">\n            <h2 style="color: #3b82f6; font-size: 1.5rem; font-weight: 700; margin-bottom: 5px;">FlightSense</h2>\n            <p style="color: #64748b; font-size: 0.85rem;">Global Dashboard Filters</p>\n            <hr style="border-color: rgba(255,255,255,0.08); margin: 15px 0;">\n        </div>\n        '), unsafe_allow_html=True)
    
    filtered_df = df.copy()
    
    if 'Loyalty Card' in df.columns:
        tiers = sorted(df['Loyalty Card'].unique())
        selected_tiers = st.sidebar.multiselect('Loyalty Card Tier', options=tiers, default=tiers, help='Select loyalty membership cards to filter by.')
        if selected_tiers:
            filtered_df = filtered_df[filtered_df['Loyalty Card'].isin(selected_tiers)]
            
    if 'Gender' in df.columns:
        genders = sorted(df['Gender'].unique())
        selected_genders = st.sidebar.multiselect('Gender', options=genders, default=genders)
        if selected_genders:
            filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
            
    if 'Education' in df.columns:
        education_opts = sorted(df['Education'].unique())
        selected_edu = st.sidebar.multiselect('Education Level', options=education_opts, default=education_edu if (education_edu := [e for e in education_opts if e in ['Bachelor', 'Master', 'Doctor']]) else education_opts)
        if selected_edu:
            filtered_df = filtered_df[filtered_df['Education'].isin(selected_edu)]
            
    if 'Province' in df.columns:
        provinces = sorted(df['Province'].unique())
        selected_prov = st.sidebar.multiselect('Province', options=provinces, default=[])
        if selected_prov:
            filtered_df = filtered_df[filtered_df['Province'].isin(selected_prov)]
            
    if 'Churn_Probability' in df.columns:
        st.sidebar.markdown("<br><b style='color:#cbd5e1; font-size:0.9rem;'>Risk Severity Filter</b>", unsafe_allow_html=True)
        min_prob, max_prob = st.sidebar.slider('Churn Probability Range', min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05)
        filtered_df = filtered_df[(filtered_df['Churn_Probability'] >= min_prob) & (filtered_df['Churn_Probability'] <= max_prob)]
        
    total_rows = len(df)
    filtered_rows = len(filtered_df)
    pct = filtered_rows / total_rows * 100 if total_rows > 0 else 0
    
    st.sidebar.markdown(textwrap.dedent(f'\n        <div style="\n            margin-top: 30px;\n            background: rgba(30, 41, 59, 0.4);\n            border: 1px solid rgba(255,255,255,0.05);\n            border-radius: 8px;\n            padding: 15px;\n            text-align: center;\n        ">\n            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px;">Active Segment</div>\n            <div style="font-size: 1.3rem; font-weight: 700; color: #3b82f6;">{filtered_rows:,} <span style="font-size: 0.9rem; color: #64748b;">/ {total_rows:,}</span></div>\n            <div style="font-size: 0.75rem; color: #64748b; margin-top: 3px;">({pct:.1f}% of customer base)</div>\n        </div>\n        '), unsafe_allow_html=True)
    
    return filtered_df