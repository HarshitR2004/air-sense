import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Premium Airline Color Palette
PRIMARY_BLUE = '#3b82f6'
SECONDARY_PURPLE = '#8b5cf6'
EMERALD_GREEN = '#10b981'
AMBER_YELLOW = '#f59e0b'
ROSE_RED = '#f43f5e'
SLATE_GRAY = '#64748b'
BG_TRANSPARENT = 'rgba(0,0,0,0)'
COLOR_PALETTE = [PRIMARY_BLUE, SECONDARY_PURPLE, EMERALD_GREEN, AMBER_YELLOW, ROSE_RED, SLATE_GRAY]

# Apply a dark premium layout styling to a Plotly figure
def apply_premium_layout(fig, title: str):
    fig.update_layout(
        title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': 18, 'family': 'Inter, Roboto, sans-serif', 'color': '#f8fafc', 'weight': 'bold'}},
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font=dict(color='#cbd5e1', family='Inter, Roboto, sans-serif'),
        legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5, bgcolor=BG_TRANSPARENT, bordercolor='rgba(255, 255, 255, 0.1)', borderwidth=1),
        margin=dict(l=40, r=40, t=60, b=80),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', zerolinecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(size=11, color='#94a3b8')),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.05)', zerolinecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(size=11, color='#94a3b8'))
    )
    return fig

# Create a premium line chart
def create_line_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = PRIMARY_BLUE) -> go.Figure:
    fig = px.line(df, x=x, y=y, color_discrete_sequence=[color])
    fig.update_traces(line=dict(width=3), marker=dict(size=6))
    return apply_premium_layout(fig, title)

# Create a clean donut/pie chart
def create_donut_chart(df: pd.DataFrame, values: str, names: str, title: str, colors: list = COLOR_PALETTE) -> go.Figure:
    fig = px.pie(df, values=values, names=names, hole=0.55, color_discrete_sequence=colors)
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0f172a', width=2)))
    return apply_premium_layout(fig, title)

# Create a premium bar chart (vertical or horizontal)
def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, orientation: str = 'v', color_col: str = None, colors: list = COLOR_PALETTE, barmode: str = 'group') -> go.Figure:
    if orientation == 'h':
        fig = px.bar(df, x=x, y=y, orientation='h', color=color_col, color_discrete_sequence=colors, barmode=barmode)
    else:
        fig = px.bar(df, x=x, y=y, orientation='v', color=color_col, color_discrete_sequence=colors, barmode=barmode)
        
    fig.update_traces(marker=dict(line=dict(color='#0f172a', width=1)))
    
    if color_col in (x, y):
        fig.update_layout(showlegend=False)
        
    return apply_premium_layout(fig, title)

# Create a scatter plot chart
def create_scatter_chart(df: pd.DataFrame, x: str, y: str, color_col: str, title: str, size_col: str = None, colors: list = COLOR_PALETTE) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color_col, size=size_col, color_discrete_sequence=colors, opacity=0.75)
    fig.update_traces(marker=dict(line=dict(color='rgba(15, 23, 42, 0.5)', width=0.5)))
    return apply_premium_layout(fig, title)

# Create a premium radar chart comparing customer profiles
def create_radar_chart(categories: list, values_dict: dict, title: str) -> go.Figure:
    fig = go.Figure()
    colors = COLOR_PALETTE
    
    for i, (name, values) in enumerate(values_dict.items()):
        color = colors[i % len(colors)]
        r_values = list(values) + [values[0]]
        theta_categories = list(categories) + [categories[0]]
        
        hex_val = color.lstrip('#')
        r = int(hex_val[0:2], 16)
        g = int(hex_val[2:4], 16)
        b = int(hex_val[4:6], 16)
        fill_color = f'rgba({r}, {g}, {b}, 0.1)'
        
        fig.add_trace(go.Scatterpolar(r=r_values, theta=theta_categories, fill='toself', fillcolor=fill_color, name=name, line=dict(color=color, width=2)))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255, 255, 255, 0.08)', linecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(size=9, color='#64748b'), angle=45),
            angularaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)', linecolor='rgba(255, 255, 255, 0.1)', tickfont=dict(size=10, color='#94a3b8')),
            bgcolor=BG_TRANSPARENT
        )
    )
    return apply_premium_layout(fig, title)

# Create a radial gauge indicator chart
def create_gauge_chart(value: float, title: str, range_min: float = 0.0, range_max: float = 1.0, color: str = PRIMARY_BLUE) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#cbd5e1'}},
        gauge={
            'axis': {'range': [range_min, range_max], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255, 255, 255, 0.05)',
            'borderwidth': 2,
            'bordercolor': 'rgba(255, 255, 255, 0.1)',
            'steps': [
                {'range': [range_min, range_min + (range_max - range_min) * 0.33], 'color': 'rgba(16, 185, 129, 0.05)'},
                {'range': [range_min + (range_max - range_min) * 0.33, range_min + (range_max - range_min) * 0.66], 'color': 'rgba(245, 158, 11, 0.05)'},
                {'range': [range_min + (range_max - range_min) * 0.66, range_max], 'color': 'rgba(244, 63, 94, 0.05)'}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor=BG_TRANSPARENT, font=dict(color='#cbd5e1', family='Inter, Roboto, sans-serif'), margin=dict(l=30, r=30, t=50, b=30), height=200)
    return fig