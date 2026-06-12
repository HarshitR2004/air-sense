import streamlit as st

# Render a single KPI metric inside a bordered container card
def kpi_card(title: str, value: str, delta: str = None, delta_type: str = 'neutral', icon: str = None, glow_color: str = '#3b82f6'):
    delta_color = 'off'
    
    if delta:
        delta_str = str(delta).strip()
        is_negative_value = delta_str.startswith('-')
        
        if delta_type == 'positive':
            delta_color = 'inverse' if is_negative_value else 'normal'
        elif delta_type == 'negative':
            delta_color = 'normal' if is_negative_value else 'inverse'
        else:
            delta_color = 'off'
            
    with st.container(border=True):
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

# Render a row of multiple KPI metrics side-by-side
def metric_row(metrics: list):
    cols = st.columns(len(metrics))
    
    for col, m in zip(cols, metrics):
        with col:
            kpi_card(
                title=m.get('title', ''),
                value=m.get('value', ''),
                delta=m.get('delta'),
                delta_type=m.get('delta_type', 'neutral'),
                icon=m.get('icon', None),
                glow_color=m.get('glow_color', '#3b82f6')
            )