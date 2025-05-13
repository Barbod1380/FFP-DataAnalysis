# multi_year_analysis.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compare_defects(old_defects_df, new_defects_df, distance_tolerance = 0.1):
    # Copy inputs
    old_df = old_defects_df.copy()
    new_df = new_defects_df.copy()
    
    # Check columns
    for col in ['log dist. [m]', 'component / anomaly identification']:
        if col not in old_df or col not in new_df:
            raise ValueError(f"Missing column: {col}")
    
    # Assign IDs
    old_df['defect_id'] = range(len(old_df))
    new_df['defect_id'] = range(len(new_df))
    
    matched_old = set()
    matched_new = set()
    matches = []
    
    for _, new_defect in new_df.iterrows():
        # Filter same‐type, within tolerance, not yet matched
        mask = (
            (old_df['component / anomaly identification']
              == new_defect['component / anomaly identification'])
            & (old_df['defect_id'].isin(matched_old) == False)
            & (old_df['log dist. [m]']
               .sub(new_defect['log dist. [m]'])
               .abs() <= distance_tolerance)
        )
        potential_matches = old_df[mask]
        
        if not potential_matches.empty:
            # Find the index label of the minimal distance
            dists = (potential_matches['log dist. [m]']
                     - new_defect['log dist. [m]']).abs()
            best_idx = dists.idxmin()
            closest_match = potential_matches.loc[best_idx]
            
            # Record it
            matches.append({
                'new_defect_id': new_defect['defect_id'],
                'old_defect_id': closest_match['defect_id'],
                'distance_diff': dists.loc[best_idx],
                'log_dist': new_defect['log dist. [m]'],
                'old_log_dist': closest_match['log dist. [m]'],
                'defect_type': new_defect['component / anomaly identification']
            })
            
            matched_old.add(closest_match['defect_id'])
            matched_new.add(new_defect['defect_id'])
    
    # Build results
    matches_df = pd.DataFrame(matches, 
        columns=['new_defect_id','old_defect_id','distance_diff',
                 'log_dist','old_log_dist','defect_type'])
    new_defects = new_df.loc[~new_df['defect_id'].isin(matched_new)].copy()
    
    total = len(new_df)
    common = len(matches_df)
    new_cnt = len(new_defects)
    
    # Stats
    pct_common = common/total*100 if total else 0
    pct_new = new_cnt/total*100 if total else 0
    
    # Distribution of “truly new” types
    if new_cnt:
        dist = (new_defects['component / anomaly identification']
                .value_counts()
                .rename_axis('defect_type')
                .reset_index(name='count'))
        dist['percentage'] = dist['count']/new_cnt*100
    else:
        dist = pd.DataFrame(columns=['defect_type','count','percentage'])
    
    return {
        'matches_df': matches_df,
        'new_defects': new_defects,
        'common_defects_count': common,
        'new_defects_count': new_cnt,
        'total_defects': total,
        'pct_common': pct_common,
        'pct_new': pct_new,
        'defect_type_distribution': dist
    }

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