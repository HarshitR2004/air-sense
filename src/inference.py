import pandas as pd
import numpy as np
import shap
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.config import *
from src.utils import logger, timer, load_model

FEATURE_BUSINESS_NAMES = {
    'Days_Since_Last_Flight': 'Days since last flight',
    'Days_Since_Last_Earning': 'Days since last points earned',
    'Days_Since_Last_Redemption': 'Days since last points redeemed',
    'Flights_3M': 'Flights in last 3 months',
    'Flights_6M': 'Flights in last 6 months',
    'Flights_12M': 'Flights in last 12 months',
    'Flights_Lifetime': 'Total lifetime flights',
    'Active_Months': 'Number of active months',
    'Points_Earned_3M': 'Points earned (3 months)',
    'Points_Earned_6M': 'Points earned (6 months)',
    'Points_Earned_12M': 'Points earned (12 months)',
    'Points_Redeemed_3M': 'Points redeemed (3 months)',
    'Points_Redeemed_6M': 'Points redeemed (6 months)',
    'Points_Redeemed_12M': 'Points redeemed (12 months)',
    'Distance_3M': 'Distance flown (3 months)',
    'Distance_6M': 'Distance flown (6 months)',
    'Distance_12M': 'Distance flown (12 months)',
    'Points_Earned_Lifetime': 'Total lifetime points earned',
    'Points_Redeemed_Lifetime': 'Total lifetime points redeemed',
    'Distance_Lifetime': 'Total lifetime distance',
    'Dollar_Cost_Lifetime': 'Lifetime dollar cost of redeemed points',
    'Flight_Trend_6M': 'Flight activity trend (6 months)',
    'Points_Trend_6M': 'Points earning trend (6 months)',
    'Distance_Trend_6M': 'Distance trend (6 months)',
    'Engagement_Trend': 'Overall engagement trend',
    'Loyalty_Card_Ordinal': 'Loyalty tier level',
    'Tenure_Months': 'Membership tenure (months)',
    'Education_Ordinal': 'Education level',
    'Salary': 'Annual salary',
    'CLV': 'Customer lifetime value (CLV)',
    'Salary_Missing': 'Salary was missing (imputed)',
    'Enrollment_Age_Months': 'Months since enrollment',
    'Is_Promo_Enrollment': 'Enrolled via promotion',
    'Is_Male': 'Gender (Male)',
    'Is_Married': 'Married',
    'Is_Single': 'Single',
    'Redemption_Ratio': 'Points redemption ratio',
    'Avg_Distance_Per_Flight': 'Average distance per flight',
    'Activity_Consistency': 'Activity consistency score',
    'Preferred_Quarter': 'Preferred travel quarter',
    'Declining_Activity': 'Declining flight activity',
    'Declining_Redemption': 'Declining redemption activity',
    'Months_Inactive': 'Consecutive months inactive'
}

# Translate code-friendly feature name to a business-friendly description
def get_business_name(feature: str) -> str:
    return FEATURE_BUSINESS_NAMES.get(feature, feature)

# Calculate SHAP values for local or sampled customer records
@timer
def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 1000) -> shap.Explanation:
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=RANDOM_SEED)
    else:
        X_sample = X.copy()
        
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        logger.info(f'SHAP values computed with TreeExplainer for {len(X_sample)} samples')
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:100])
        shap_values = explainer(X_sample)
        logger.info(f'SHAP values computed with KernelExplainer for {len(X_sample)} samples')
        
    return shap_values

# Perform global SHAP analysis to determine top program-wide churn drivers
@timer
def global_shap_analysis(model, X: pd.DataFrame, max_samples: int = 1000) -> Dict[str, Any]:
    shap_values = compute_shap_values(model, X, max_samples)
    
    if len(shap_values.shape) == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values
        
    mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Mean_SHAP': mean_abs_shap,
        'Business_Name': [get_business_name(f) for f in X.columns]
    }).sort_values('Mean_SHAP', ascending=False)
    
    top_drivers = feature_importance.head(10)
    
    return {
        'shap_values': shap_vals,
        'feature_importance': feature_importance,
        'top_drivers': top_drivers,
        'X_sample': X.iloc[:max_samples] if len(X) > max_samples else X
    }

# Compute localized SHAP details for a single target customer
def local_shap_explanation(model, X: pd.DataFrame, customer_idx: int, shap_values = None) -> Dict[str, Any]:
    if shap_values is None:
        shap_values = compute_shap_values(model, X.iloc[[customer_idx]])
        
    if len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values
        
    if hasattr(vals, 'values'):
        customer_shap = vals.values[0] if len(vals.values.shape) > 1 else vals.values
    else:
        customer_shap = vals[0]
        
    explanation = pd.DataFrame({
        'Feature': X.columns,
        'Value': X.iloc[customer_idx].values,
        'SHAP_Value': customer_shap,
        'Business_Name': [get_business_name(f) for f in X.columns],
        'Direction': ['↑ Increases churn risk' if s > 0 else '↓ Decreases churn risk' for s in customer_shap]
    }).sort_values('SHAP_Value', key=abs, ascending=False)
    
    return {
        'explanation': explanation,
        'top_risk_factors': explanation[explanation['SHAP_Value'] > 0].head(5),
        'top_retention_factors': explanation[explanation['SHAP_Value'] < 0].head(5)
    }

# Generate textual explanation summarizing risk and retention factors
def generate_customer_explanation_text(explanation: Dict[str, Any]) -> str:
    risk_factors = explanation['top_risk_factors']
    retention_factors = explanation['top_retention_factors']
    
    text = '**Why this customer may churn:**\n'
    for _, row in risk_factors.iterrows():
        text += f"  • {row['Business_Name']} = {row['Value']:.1f}\n"
        
    text += '\n**Factors keeping this customer:**\n'
    for _, row in retention_factors.iterrows():
        text += f"  • {row['Business_Name']} = {row['Value']:.1f}\n"
        
    return text

# Fetch top program features sorted by global SHAP importance
def get_global_feature_importance(model = None, X = None, max_samples: int = 500) -> pd.DataFrame:
    if model is None:
        model = load_model(MODELS_DIR / 'churn_model.joblib')
        
    if X is None:
        df = pd.read_csv(FEATURES_DIR / 'customer_features.csv')
        feature_cols = [c for c in df.columns if c not in [PK, 'Churn']]
        X = df[feature_cols]
        
    result = global_shap_analysis(model, X, max_samples)
    return result['feature_importance']

# Retrieve the specific SHAP explanation metrics for a customer row
def get_customer_explanation(customer_data: pd.Series, model = None) -> Dict[str, Any]:
    if model is None:
        model = load_model(MODELS_DIR / 'churn_model.joblib')
        
    X = pd.DataFrame([customer_data])
    shap_vals = compute_shap_values(model, X)
    return local_shap_explanation(model, X, 0, shap_vals)

# Simulate churn reduction scenarios and calculate saved customer value
@timer
def simulate_retention_impact(df: pd.DataFrame, improvement_pcts: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25]) -> pd.DataFrame:
    churn_col = 'Churn_Probability' if 'Churn_Probability' in df.columns else 'Churn'
    at_risk = df[df[churn_col] > 0.5].copy()
    total_at_risk = len(at_risk)
    
    if 'FVS' in at_risk.columns:
        at_risk = at_risk.sort_values('FVS', ascending=False)
        
    scenarios = []
    for pct in improvement_pcts:
        customers_saved = int(total_at_risk * pct)
        if 'FVS' in at_risk.columns:
            clv_retained = at_risk.head(customers_saved)['CLV'].sum()
        else:
            clv_retained = customers_saved * at_risk['CLV'].mean() if not at_risk.empty else 0
            
        revenue_protected = clv_retained
        avg_fvs_saved = at_risk.head(customers_saved)['FVS'].mean() if 'FVS' in at_risk.columns and (not at_risk.head(customers_saved).empty) else 0
        
        scenarios.append({
            'Retention_Improvement': f'{pct * 100:.0f}%',
            'Improvement_Pct': pct,
            'Customers_At_Risk': total_at_risk,
            'Customers_Saved': customers_saved,
            'CLV_Retained': round(clv_retained, 2),
            'Revenue_Protected': round(revenue_protected, 2),
            'Avg_CLV_Saved': round(clv_retained / max(customers_saved, 1), 2),
            'Avg_FVS_Saved': round(avg_fvs_saved, 1)
        })
        
    return pd.DataFrame(scenarios)

# Calculate financial return on investment for campaign scenarios
def calculate_roi(simulation: pd.DataFrame, cost_per_customer: float = 150.0) -> pd.DataFrame:
    result = simulation.copy()
    result['Campaign_Cost'] = result['Customers_Saved'] * cost_per_customer
    result['Net_Revenue'] = result['Revenue_Protected'] - result['Campaign_Cost']
    result['ROI'] = np.where(result['Campaign_Cost'] > 0, (result['Net_Revenue'] / result['Campaign_Cost'] * 100).round(1), 0.0)
    result['ROI_Label'] = result['ROI'].apply(lambda x: f'{x:.0f}%')
    return result