# multi_year_analysis.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compare_defects(old_defects_df, new_defects_df, old_year, new_year, distance_tolerance=0.1):
    """
    Compare defects between two inspection years to identify common and new defects.
    
    Parameters:
    - old_defects_df: DataFrame with defects from the earlier inspection
    - new_defects_df: DataFrame with defects from the newer inspection
    - old_year: Year of the earlier inspection
    - new_year: Year of the later inspection
    - distance_tolerance: Maximum distance (in meters) to consider defects at the same location
    
    Returns:
    - results: Dictionary with comparison results and statistics
    """
    # Make copies to avoid modifying the original dataframes
    old_df = old_defects_df.copy()
    new_df = new_defects_df.copy()
    
    # Calculate year difference for growth rate calculation
    year_diff = new_year - old_year
    if year_diff <= 0:
        raise ValueError("New year must be greater than old year")
    
    # Ensure required columns exist
    required_cols = ['log dist. [m]', 'component / anomaly identification']
    for col in required_cols:
        if col not in old_df.columns or col not in new_df.columns:
            raise ValueError(f"Column '{col}' missing from one of the datasets")
    
    # Check if depth data is available for growth calculations
    has_depth_data = ('depth [%]' in old_df.columns and 'depth [%]' in new_df.columns)
    has_wt_data = ('wt nom [mm]' in old_df.columns and 'wt nom [mm]' in new_df.columns)
    
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
            
            # Initialize match data with basic info
            match_data = {
                'new_defect_id': new_defect['defect_id'],
                'old_defect_id': closest_match['defect_id'],
                'distance_diff': abs(closest_match['log dist. [m]'] - new_defect['log dist. [m]']),
                'log_dist': new_defect['log dist. [m]'],
                'old_log_dist': closest_match['log dist. [m]'],
                'defect_type': new_defect['component / anomaly identification']
            }
            
            # Add depth and growth information if available
            if has_depth_data:
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
            
            matched_old_ids.add(closest_match['defect_id'])
            matched_new_ids.add(new_defect['defect_id'])
    
    # Create dataframe of matches
    if matches:
        matches_df = pd.DataFrame(matches)
    else:
        # Create empty dataframe with appropriate columns
        columns = ['new_defect_id', 'old_defect_id', 'distance_diff', 'log_dist', 
                   'old_log_dist', 'defect_type']
        if has_depth_data:
            columns.extend(['old_depth_pct', 'new_depth_pct', 'depth_change_pct', 
                           'growth_rate_pct_per_year', 'is_negative_growth'])
            if has_wt_data:
                columns.extend(['old_depth_mm', 'new_depth_mm', 'depth_change_mm', 
                               'growth_rate_mm_per_year'])
        matches_df = pd.DataFrame(columns=columns)
    
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
    
    # Calculate growth statistics if depth data is available
    if has_depth_data and not matches_df.empty:
        # Growth statistics - first in percentage points
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
    else:
        growth_stats = None
    
    # Return results
    results = {
        'matches_df': matches_df,
        'new_defects': new_defects,
        'common_defects_count': common_defects_count,
        'new_defects_count': new_defects_count,
        'total_defects': total_new_defects,
        'pct_common': pct_common,
        'pct_new': pct_new,
        'defect_type_distribution': defect_type_counts,
        'growth_stats': growth_stats,
        'has_depth_data': has_depth_data,
        'has_wt_data': has_wt_data
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