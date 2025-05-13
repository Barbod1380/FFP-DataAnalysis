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
        hole=0.3,
        textinfo='label+percent',
        textposition='inside',  # Better placement of labels
        insidetextorientation='radial',  # Makes text horizontal inside slices
        marker=dict(colors=['#2E86C1', '#EC7063']),
        direction='clockwise',  # Optional: control rotation direction
        sort=False  # Keep the order of labels
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