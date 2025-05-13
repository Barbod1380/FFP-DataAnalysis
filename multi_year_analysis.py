# multi_year_analysis.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compare_defects(old_defects_df, new_defects_df, distance_tolerance=0.1):
    """
    Compare defects between two inspection years to identify common and new defects.
    
    Parameters:
    - old_defects_df: DataFrame with defects from the earlier inspection
    - new_defects_df: DataFrame with defects from the newer inspection
    - distance_tolerance: Maximum distance (in meters) to consider defects at the same location
    
    Returns:
    - results: Dictionary with comparison results and statistics
    """
    # Make copies to avoid modifying the original dataframes
    old_df = old_defects_df.copy()
    new_df = new_defects_df.copy()
    
    # Ensure required columns exist
    required_cols = ['log dist. [m]', 'component / anomaly identification']
    for col in required_cols:
        if col not in old_df.columns or col not in new_df.columns:
            raise ValueError(f"Column '{col}' missing from one of the datasets")
    
    # Add unique identifiers to track defects
    old_df['defect_id'] = range(len(old_df))
    new_df['defect_id'] = range(len(new_df))
    
    # Track matched defects
    matched_old_ids = set()
    matched_new_ids = set()
    matches = []
    
    # For each defect in the new dataset, find matching defects in the old dataset
    for _, new_defect in new_df.iterrows():
        # Find defects of the same type within the distance tolerance 
        # that haven't already been matched
        potential_matches = old_df[
            (old_df['component / anomaly identification'] == new_defect['component / anomaly identification']) & 
            (abs(old_df['log dist. [m]'] - new_defect['log dist. [m]']) <= distance_tolerance) &
            (~old_df['defect_id'].isin(matched_old_ids))  # Only consider unmatched old defects
        ]
        
        if not potential_matches.empty:
            # Find the closest match
            closest_match = potential_matches.iloc[
                (potential_matches['log dist. [m]'] - new_defect['log dist. [m]']).abs().argsort()[0]
            ]
            
            # Record the match
            matches.append({
                'new_defect_id': new_defect['defect_id'],
                'old_defect_id': closest_match['defect_id'],
                'distance_diff': abs(closest_match['log dist. [m]'] - new_defect['log dist. [m]']),
                'log_dist': new_defect['log dist. [m]'],
                'old_log_dist': closest_match['log dist. [m]'],  # Added for better traceability
                'defect_type': new_defect['component / anomaly identification']
            })
            
            matched_old_ids.add(closest_match['defect_id'])
            matched_new_ids.add(new_defect['defect_id'])
    
    # Create dataframe of matches
    matches_df = pd.DataFrame(matches) if matches else pd.DataFrame(columns=[
        'new_defect_id', 'old_defect_id', 'distance_diff', 'log_dist', 'old_log_dist', 'defect_type'
    ])
    
    # Identify new defects (those in new_df that weren't matched)
    new_defects = new_df[~new_df['defect_id'].isin(matched_new_ids)].copy()
    
    # Calculate statistics
    total_new_defects = len(new_df)
    common_defects_count = len(matches_df)
    new_defects_count = len(new_defects)
    
    # Percentage calculations
    pct_common = (common_defects_count / total_new_defects) * 100 if total_new_defects > 0 else 0
    pct_new = (new_defects_count / total_new_defects) * 100 if total_new_defects > 0 else 0
    
    # Distribution of new defect types
    if not new_defects.empty:
        defect_type_counts = new_defects['component / anomaly identification'].value_counts().reset_index()
        defect_type_counts.columns = ['defect_type', 'count']
        defect_type_counts['percentage'] = (defect_type_counts['count'] / new_defects_count) * 100
    else:
        defect_type_counts = pd.DataFrame(columns=['defect_type', 'count', 'percentage'])
    
    # Return results
    results = {
        'matches_df': matches_df,
        'new_defects': new_defects,
        'common_defects_count': common_defects_count,
        'new_defects_count': new_defects_count,
        'total_defects': total_new_defects,
        'pct_common': pct_common,
        'pct_new': pct_new,
        'defect_type_distribution': defect_type_counts
    }
    return results

def create_comparison_stats_plot(comparison_results):
    """
    Create a pie chart showing new vs. common defects
    """
    labels = ['Common Defects', 'New Defects']
    values = [
        comparison_results['common_defects_count'],
        comparison_results['new_defects_count']
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        textinfo='label+percent',
        marker=dict(colors=['#2E86C1', '#EC7063'])
    )])
    
    fig.update_layout(
        title='Distribution of Common vs. New Defects',
        font=dict(size=14),
        height=400
    )
    
    return fig

def create_new_defect_types_plot(comparison_results):
    """
    Create a bar chart showing distribution of new defect types
    """
    type_dist = comparison_results['defect_type_distribution']
    
    if type_dist.empty:
        # Create an empty figure with a message if there are no new defects
        fig = go.Figure()
        fig.add_annotation(
            text="No new defects found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Sort by count descending
    type_dist = type_dist.sort_values('count', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(
            x=type_dist['defect_type'],
            y=type_dist['count'],
            text=type_dist['percentage'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto',
            marker_color='#EC7063'
        )
    ])
    
    fig.update_layout(
        title='Distribution of New Defect Types',
        xaxis_title='Defect Type',
        yaxis_title='Count',
        font=dict(size=14),
        height=500,
        xaxis=dict(tickangle=-45)  # Rotate x labels for better readability
    )
    
    return fig

def create_defect_location_plot(comparison_results, old_defects_df, new_defects_df):
    """
    Create a scatter plot showing the location of defects along the pipeline
    Highlighting common and new defects
    """
    # Get matched and new defect IDs
    matched_new_ids = set(comparison_results['matches_df']['new_defect_id']) if not comparison_results['matches_df'].empty else set()
    
    # Prepare data for plotting
    common_defects = new_defects_df[new_defects_df['defect_id'].isin(matched_new_ids)].copy()
    new_defects = comparison_results['new_defects']
    
    # Create plot
    fig = go.Figure()
    
    # Add common defects
    if not common_defects.empty:
        fig.add_trace(go.Scatter(
            x=common_defects['log dist. [m]'],
            y=common_defects['clock_float'] if 'clock_float' in common_defects.columns else [1] * len(common_defects),
            mode='markers',
            name='Common Defects',
            marker=dict(
                color='#2E86C1',
                size=10,
                opacity=0.7
            ),
            hovertemplate=(
                "<b>Common Defect</b><br>"
                "Distance: %{x:.2f} m<br>"
                "Type: %{customdata[0]}<br>"
                "Depth: %{customdata[1]:.1f}%<extra></extra>"
            ),
            customdata=np.stack((
                common_defects['component / anomaly identification'],
                common_defects['depth [%]'].fillna(0)
            ), axis=-1)
        ))
    
    # Add new defects
    if not new_defects.empty:
        fig.add_trace(go.Scatter(
            x=new_defects['log dist. [m]'],
            y=new_defects['clock_float'] if 'clock_float' in new_defects.columns else [1] * len(new_defects),
            mode='markers',
            name='New Defects',
            marker=dict(
                color='#EC7063',
                size=10,
                opacity=0.7
            ),
            hovertemplate=(
                "<b>New Defect</b><br>"
                "Distance: %{x:.2f} m<br>"
                "Type: %{customdata[0]}<br>"
                "Depth: %{customdata[1]:.1f}%<extra></extra>"
            ),
            customdata=np.stack((
                new_defects['component / anomaly identification'],
                new_defects['depth [%]'].fillna(0)
            ), axis=-1)
        ))
    
    # Update layout
    fig.update_layout(
        title='Location of Common and New Defects Along Pipeline',
        xaxis_title='Distance Along Pipeline (m)',
        yaxis_title='Clock Position' if 'clock_float' in new_defects_df.columns else 'Position',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        hovermode='closest',
        height=500
    )
    
    return fig