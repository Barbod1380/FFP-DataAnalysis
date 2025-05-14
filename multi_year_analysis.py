# multi_year_analysis.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compare_defects(old_defects_df, new_defects_df, old_year=None, new_year=None, distance_tolerance=0.1):
    """
    Compare defects between two inspection years to identify common and new defects.
    
    Parameters:
    - old_defects_df: DataFrame with defects from the earlier inspection
    - new_defects_df: DataFrame with defects from the newer inspection
    - old_year: Year of the earlier inspection (optional, for growth rate calculation)
    - new_year: Year of the later inspection (optional, for growth rate calculation)
    - distance_tolerance: Maximum distance (in meters) to consider defects at the same location
    
    Returns:
    - results: Dictionary with comparison results and statistics
    """
    # Copy inputs
    old_df = old_defects_df.copy()
    new_df = new_defects_df.copy()
    
    # Check if we can calculate growth rates
    calculate_growth = False
    if old_year is not None and new_year is not None and new_year > old_year:
        calculate_growth = True
        year_diff = new_year - old_year
    
    # Check if depth data is available for growth calculations
    has_depth_data = ('depth [%]' in old_df.columns and 'depth [%]' in new_df.columns)
    has_wt_data = ('wt nom [mm]' in old_df.columns and 'wt nom [mm]' in new_df.columns)
    
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
            
            # Basic match data
            match_data = {
                'new_defect_id': new_defect['defect_id'],
                'old_defect_id': closest_match['defect_id'],
                'distance_diff': dists.loc[best_idx],
                'log_dist': new_defect['log dist. [m]'],
                'old_log_dist': closest_match['log dist. [m]'],
                'defect_type': new_defect['component / anomaly identification']
            }
            
            # Add growth data if available
            if calculate_growth and has_depth_data:
                old_depth = closest_match['depth [%]']
                new_depth = new_defect['depth [%]']
                
                # Add depth information
                match_data.update({
                    'old_depth_pct': old_depth,
                    'new_depth_pct': new_depth,
                    'depth_change_pct': new_depth - old_depth,
                    'growth_rate_pct_per_year': (new_depth - old_depth) / year_diff,
                    'is_negative_growth': (new_depth - old_depth) < 0
                })
                
                # If wall thickness data is available, convert to mm/year
                if has_wt_data:
                    old_wt = closest_match['wt nom [mm]']
                    new_wt = new_defect['wt nom [mm]']
                    
                    # Use the average wall thickness for conversion
                    avg_wt = (old_wt + new_wt) / 2
                    
                    old_depth_mm = old_depth * avg_wt / 100
                    new_depth_mm = new_depth * avg_wt / 100
                    
                    match_data.update({
                        'old_depth_mm': old_depth_mm,
                        'new_depth_mm': new_depth_mm,
                        'depth_change_mm': new_depth_mm - old_depth_mm,
                        'growth_rate_mm_per_year': (new_depth_mm - old_depth_mm) / year_diff
                    })
            
            matches.append(match_data)
            matched_old.add(closest_match['defect_id'])
            matched_new.add(new_defect['defect_id'])
    
    # Column list for empty dataframe handling
    columns = ['new_defect_id', 'old_defect_id', 'distance_diff', 
               'log_dist', 'old_log_dist', 'defect_type']
               
    if calculate_growth and has_depth_data:
        columns.extend(['old_depth_pct', 'new_depth_pct', 'depth_change_pct', 
                       'growth_rate_pct_per_year', 'is_negative_growth'])
        if has_wt_data:
            columns.extend(['old_depth_mm', 'new_depth_mm', 'depth_change_mm', 
                           'growth_rate_mm_per_year'])
    
    # Build results
    matches_df = pd.DataFrame(matches, columns=columns) if matches else pd.DataFrame(columns=columns)
    new_defects = new_df.loc[~new_df['defect_id'].isin(matched_new)].copy()
    
    total = len(new_df)
    common = len(matches_df)
    new_cnt = len(new_defects)
    
    # Stats
    pct_common = common/total*100 if total else 0
    pct_new = new_cnt/total*100 if total else 0
    
    # Distribution of "truly new" types
    if new_cnt:
        dist = (new_defects['component / anomaly identification']
                .value_counts()
                .rename_axis('defect_type')
                .reset_index(name='count'))
        dist['percentage'] = dist['count']/new_cnt*100
    else:
        dist = pd.DataFrame(columns=['defect_type', 'count', 'percentage'])
    
    # Calculate growth statistics if depth data is available
    growth_stats = None
    if calculate_growth and has_depth_data and not matches_df.empty:
        # Growth statistics
        negative_growth_count = matches_df['is_negative_growth'].sum()
        pct_negative_growth = (negative_growth_count / len(matches_df)) * 100 if len(matches_df) > 0 else 0
        
        # Filter out negative growth for positive growth stats
        positive_growth = matches_df[~matches_df['is_negative_growth']]
        
        growth_stats = {
            'total_matched_defects': len(matches_df),
            'negative_growth_count': negative_growth_count,
            'pct_negative_growth': pct_negative_growth,
            'avg_growth_rate_pct': matches_df['growth_rate_pct_per_year'].mean(),
            'avg_positive_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].mean() if len(positive_growth) > 0 else 0,
            'max_growth_rate_pct': positive_growth['growth_rate_pct_per_year'].max() if len(positive_growth) > 0 else 0
        }
        
        # Add mm-based stats if available
        if has_wt_data:
            growth_stats.update({
                'avg_growth_rate_mm': matches_df['growth_rate_mm_per_year'].mean(),
                'avg_positive_growth_rate_mm': positive_growth['growth_rate_mm_per_year'].mean() if len(positive_growth) > 0 else 0,
                'max_growth_rate_mm': positive_growth['growth_rate_mm_per_year'].max() if len(positive_growth) > 0 else 0
            })
    
    return {
        'matches_df': matches_df,
        'new_defects': new_defects,
        'common_defects_count': common,
        'new_defects_count': new_cnt,
        'total_defects': total,
        'pct_common': pct_common,
        'pct_new': pct_new,
        'defect_type_distribution': dist,
        'growth_stats': growth_stats,
        'has_depth_data': has_depth_data,
        'has_wt_data': has_wt_data,
        'calculate_growth': calculate_growth
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