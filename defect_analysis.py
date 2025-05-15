import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dimension_distribution_plots(defects_df, dimension_columns=None):
    """
    Create distribution plots for defect dimensions.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - dimension_columns: List of dimension columns to plot (default: standard dimensions)
    
    Returns:
    - Dictionary of plotly figures keyed by dimension name
    """
    if dimension_columns is None:
        dimension_columns = {
            'length [mm]': 'Defect Length (mm)',
            'width [mm]': 'Defect Width (mm)',
            'depth [%]': 'Defect Depth (%)'
        }
    
    figures = {}
    
    for col, title in dimension_columns.items():
        if col in defects_df.columns:
            # Filter out non-numeric values and NaNs
            valid_data = defects_df[pd.to_numeric(defects_df[col], errors='coerce').notna()]
            
            if not valid_data.empty:
                # Create histogram
                fig = px.histogram(
                    valid_data,
                    x=col,
                    nbins=20,
                    marginal="box",  # Add a box plot to show quartiles
                    title=f"Distribution of {title}",
                    labels={col: title},
                    color_discrete_sequence=['rgba(0, 128, 255, 0.6)']
                )
                
                # Add mean line
                mean_val = valid_data[col].mean()
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_val:.2f}",
                    annotation_position="top right"
                )
                
                # Format
                fig.update_layout(
                    xaxis_title=title,
                    yaxis_title="Count",
                    bargap=0.1,
                    height=400
                )
                
                figures[col] = fig
    
    return figures

def create_combined_dimensions_plot(defects_df):
    """
    Create a scatter plot showing the relationship between length, width and depth.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    
    Returns:
    - Plotly figure object
    """
    # Check if we have the necessary columns
    req_cols = ['length [mm]', 'width [mm]']
    has_depth = 'depth [%]' in defects_df.columns
    
    if not all(col in defects_df.columns for col in req_cols):
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="Required dimension columns not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Filter out NaNs and invalid values
    valid_data = defects_df.copy()
    for col in req_cols:
        valid_data = valid_data[pd.to_numeric(valid_data[col], errors='coerce').notna()]
    
    if has_depth:
        valid_data = valid_data[pd.to_numeric(valid_data['depth [%]'], errors='coerce').notna()]
    
    if valid_data.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No valid dimension data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Calculate area
    valid_data['area [mm²]'] = valid_data['length [mm]'] * valid_data['width [mm]']
    
    # Create scatter plot
    if has_depth:
        fig = px.scatter(
            valid_data,
            x='length [mm]',
            y='width [mm]',
            color='depth [%]',
            size='area [mm²]',
            hover_name='component / anomaly identification' if 'component / anomaly identification' in valid_data.columns else None,
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Defect Dimensions Relationship",
            labels={
                'length [mm]': 'Length (mm)',
                'width [mm]': 'Width (mm)',
                'depth [%]': 'Depth (%)',
                'area [mm²]': 'Area (mm²)'
            }
        )
    else:
        fig = px.scatter(
            valid_data,
            x='length [mm]',
            y='width [mm]',
            size='area [mm²]',
            hover_name='component / anomaly identification' if 'component / anomaly identification' in valid_data.columns else None,
            title="Defect Dimensions Relationship",
            labels={
                'length [mm]': 'Length (mm)',
                'width [mm]': 'Width (mm)',
                'area [mm²]': 'Area (mm²)'
            }
        )
    
    # Format
    fig.update_layout(
        height=500,
        xaxis=dict(type='log') if valid_data['length [mm]'].max() / valid_data['length [mm]'].min() > 100 else None,
        yaxis=dict(type='log') if valid_data['width [mm]'].max() / valid_data['width [mm]'].min() > 100 else None
    )
    
    return fig

def create_dimension_statistics_table(defects_df):
    """
    Create a statistics summary table for defect dimensions.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    
    Returns:
    - DataFrame with dimension statistics
    """
    dimension_cols = ['length [mm]', 'width [mm]', 'depth [%]']
    available_cols = [col for col in dimension_cols if col in defects_df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    # Calculate statistics for each dimension
    stats = []
    for col in available_cols:
        # Convert to numeric, coercing errors to NaN
        values = pd.to_numeric(defects_df[col], errors='coerce')
        
        # Skip if all values are NaN
        if values.isna().all():
            continue
            
        stat = {
            'Dimension': col,
            'Mean': values.mean(),
            'Median': values.median(),
            'Min': values.min(),
            'Max': values.max(),
            'Std Dev': values.std(),
            'Count': values.count()
        }
        stats.append(stat)
    
    return pd.DataFrame(stats)