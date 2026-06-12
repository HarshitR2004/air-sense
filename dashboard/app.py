import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import textwrap
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import *
from src.utils import load_model, logger

# Set logging level globally for the dashboard session
logger.setLevel(logging.WARNING)

# Page configuration setup
st.set_page_config(page_title='FlightSense - Loyalty Behavioral Intelligence', page_icon=None, layout='wide', initial_sidebar_state='expanded')

# Inject premium dark theme styling
st.markdown(textwrap.dedent('\n    <style>\n    @import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap\');\n    \n    html, body, [class*="css"] {\n        font-family: \'Inter\', sans-serif;\n    }\n    \n    .stApp {\n        background-color: #0f172a;\n        color: #f8fafc;\n    }\n    \n    /* Premium glowing header styling */\n    .dashboard-header {\n        font-size: 2.2rem;\n        font-weight: 800;\n        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);\n        -webkit-background-clip: text;\n        -webkit-text-fill-color: transparent;\n        margin-bottom: 20px;\n        letter-spacing: -0.03em;\n    }\n    \n    .dashboard-subtitle {\n        font-size: 1rem;\n        color: #64748b;\n        margin-top: -15px;\n        margin-bottom: 30px;\n    }\n    \n    /* Sidebar styling */\n    section[data-testid="stSidebar"] {\n        background-color: #1e293b;\n        border-right: 1px solid rgba(255, 255, 255, 0.05);\n    }\n    \n    /* DataFrame/Table custom styles */\n    div[data-testid="stDataFrame"] {\n        background-color: #1e293b80;\n        border: 1px solid rgba(255, 255, 255, 0.05);\n        border-radius: 8px;\n    }\n    \n    /* Alert cards styling */\n    .stAlert {\n        background-color: #1e293b;\n        border: 1px solid rgba(255, 255, 255, 0.08);\n        border-radius: 8px;\n        color: #e2e8f0;\n    }\n    </style>\n    '), unsafe_allow_html=True)

# Load and merge segmented datasets, recommendation mappings, and metadata
@st.cache_data(show_spinner='Loading customer database...')
def load_dashboard_data():
    segmented_path = FEATURES_DIR / 'customer_segmented.csv'
    recs_path = FEATURES_DIR / 'retention_recommendations.csv'
    metadata_path = MODELS_DIR / 'model_metadata.json'
    
    if not segmented_path.exists() or not recs_path.exists():
        st.error('Pipeline outputs not found! Please ensure the customer datasets are generated and models are trained.')
        return (None, None, None)
        
    df = pd.read_csv(segmented_path)
    recs = pd.read_csv(recs_path)
    
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
        
    desired_cols = [PK, 'Priority', 'Primary_Action', 'All_Actions', 'Urgency', 'Estimated_Cost', 'Expected_Retention_Lift']
    available_cols = [c for c in desired_cols if c in recs.columns]
    df_merged = df.merge(recs[available_cols], on=PK, how='left')
    
    raw_history_path = RAW_DATA_DIR / LOYALTY_HISTORY_FILE
    if raw_history_path.exists():
        cols_to_merge = [PK, 'City', 'Province', 'Gender', 'Education', 'Marital Status', 'Loyalty Card']
        raw_cols = pd.read_csv(raw_history_path, nrows=0).columns
        available_merge_cols = [c for c in cols_to_merge if c in raw_cols]
        raw_df = pd.read_csv(raw_history_path, usecols=available_merge_cols)
        
        cols_to_drop = [c for c in available_merge_cols if c != PK and c in df_merged.columns]
        if cols_to_drop:
            df_merged = df_merged.drop(columns=cols_to_drop)
            
        df_merged = df_merged.merge(raw_df, on=PK, how='left')
    else:
        if 'Loyalty Card' not in df_merged.columns and 'Loyalty_Card_Ordinal' in df_merged.columns:
            card_map_inv = {0: 'Star', 1: 'Nova', 2: 'Aurora'}
            df_merged['Loyalty Card'] = df_merged['Loyalty_Card_Ordinal'].map(card_map_inv)
        if 'Gender' not in df_merged.columns and 'Is_Male' in df_merged.columns:
            df_merged['Gender'] = df_merged['Is_Male'].map({1: 'Male', 0: 'Female'})
        if 'Education' not in df_merged.columns and 'Education_Ordinal' in df_merged.columns:
            education_map_inv = {0: 'High School or Below', 1: 'College', 2: 'Bachelor', 3: 'Master', 4: 'Doctor'}
            df_merged['Education'] = df_merged['Education_Ordinal'].map(education_map_inv)
        if 'Marital Status' not in df_merged.columns and 'Is_Married' in df_merged.columns and ('Is_Single' in df_merged.columns):
            df_merged['Marital Status'] = 'Divorced'
            df_merged.loc[df_merged['Is_Married'] == 1, 'Marital Status'] = 'Married'
            df_merged.loc[df_merged['Is_Single'] == 1, 'Marital Status'] = 'Single'
            
    if 'Churn_Probability' not in df_merged.columns:
        df_merged['Churn_Probability'] = df_merged['Churn'].astype(float)
        
    return (df_merged, recs, meta)

# Main Application Entrypoint
if 'df' not in st.session_state or 'recs' not in st.session_state:
    df, recs, meta = load_dashboard_data()
    
    if df is not None:
        st.session_state['df'] = df
        st.session_state['recs'] = recs
        st.session_state['meta'] = meta
        st.session_state['pipeline_ready'] = True
    else:
        st.session_state['pipeline_ready'] = False

if st.session_state.get('pipeline_ready', False):
    pages = {
        'Executive Control': [
            st.Page('pages/1_Executive_Overview.py', title='Executive Overview')
        ],
        'Loyalty Intelligence': [
            st.Page('pages/2_Churn_Intelligence.py', title='Churn Intelligence'),
            st.Page('pages/3_Customer_Segmentation.py', title='Customer Segmentation')
        ],
        'Retention & Action': [
            st.Page('pages/4_Retention_Command_Center.py', title='Retention Command'),
            st.Page('pages/5_Revenue_Impact_Simulator.py', title='Revenue Simulator'),
            st.Page('pages/6_Customer_360_View.py', title='Customer 360')
        ]
    }
    pg = st.navigation(pages)
    pg.run()
else:
    st.markdown('<div class="dashboard-header">FlightSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Behavioral Intelligence & Retention Optimization System</div>', unsafe_allow_html=True)
    st.warning('Loyalty analytics pipeline has not been executed yet. The database is empty.')
    
    st.markdown('\n        ### System Onboarding\n        To populate the dashboard and inspect customer retention insights, you need to execute the end-to-end data science pipeline. \n        This will:\n        1. **Understand & Audit** the raw airline data.\n        2. **Clean & Impute** missing values (e.g. Salary using province/education medians).\n        3. **Define Churn** using the hybrid activity/cancellation definition.\n        4. **Engineer 35+ Behavioral Features** including RFM rolling averages, trends, and risk signals.\n        5. **Train 4 machine learning models** and optimize hyper-parameters.\n        6. **Generate Future Value Scores (FVS)** and customer micro-segments.\n        7. **Map segments to prioritized retention actions** and financial impact simulations.\n        ')