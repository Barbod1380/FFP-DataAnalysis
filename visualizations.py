import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import decimal_to_clock_str

def create_unwrapped_pipeline_visualization(defects_df, joints_df):
    """
    Create an enhanced unwrapped cylinder visualization of pipeline defects.
    
    Parameters:
    - defects_df: DataFrame with defect information
    - joints_df: DataFrame with joint information
    
    Returns:
    - Plotly figure object
    """
    # Precompute marker sizes (relative scale with better range)
    min_size, max_size = 5, 30
    area = defects_df["area_mm2"].fillna(0)
    scaled_sizes = np.clip(
        (area / area.max()) * max_size,
        min_size,
        max_size
    )
    
    # Create clock_str column for hover display
    defects_df['clock_str'] = defects_df['clock_float'].apply(decimal_to_clock_str)
    
    # Get actual max depth for dynamic color scaling
    max_depth = defects_df["depth [%]"].max()
    
    # Prepare color dimensions for different visualization modes
    color_modes = {
        "Depth (%)": {
            "column": "depth [%]",
            "colorscale": "Turbo",
            "color_range": [0, max_depth]  # Dynamic scaling based on actual data
        },
        "Surface Location": {
            "column": "surface location",
            "colorscale": "Viridis",
            "is_categorical": True
        },
        "Area (mm²)": {
            "column": "area_mm2",
            "colorscale": "Plasma",
            "color_range": [0, defects_df["area_mm2"].max()]
        }
    }
    
    # Create figure with subplots (1 row, 1 column)
    fig = make_subplots(rows=1, cols=1, 
                        specs=[[{"secondary_y": True}]])
    
    # Add one scatter trace per color mode
    for i, (label, config) in enumerate(color_modes.items()):
        col = config["column"]
        
        # Skip modes with missing data
        if col not in defects_df.columns:
            continue
            
        if config.get("is_categorical", False):
            # For categorical data like surface location
            unique_values = defects_df[col].dropna().unique()
            color_map = px.colors.qualitative.Bold
            
            for j, val in enumerate(unique_values):
                mask = defects_df[col] == val
                color = color_map[j % len(color_map)]
                
                fig.add_trace(go.Scatter(
                    x=defects_df.loc[mask, "log dist. [m]"],
                    y=defects_df.loc[mask, "clock_float"],
                    mode="markers",
                    marker=dict(
                        size=scaled_sizes[mask],
                        color=color,
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    name=f"{val}",
                    legendgroup=label,
                    customdata=np.stack([
                        defects_df.loc[mask, "joint number"].astype(str),
                        defects_df.loc[mask, "component / anomaly identification"],
                        defects_df.loc[mask, "depth [%]"].fillna(0),
                        defects_df.loc[mask, "area_mm2"].fillna(0),
                        defects_df.loc[mask, "clock_str"]
                    ], axis=-1),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "<b>Distance:</b> %{x:.2f} m<br>"
                        "<b>Clock:</b> %{customdata[4]}<br>"
                        "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
                        "<b>Area:</b> %{customdata[3]:.1f} mm²<br>"
                        "<b>Joint:</b> %{customdata[0]}<br>"
                        f"<b>{label}:</b> {val}<extra></extra>"
                    ),
                    visible=(i == 0)  # only first mode visible on load
                ))
        else:
            # For continuous data like depth
            fig.add_trace(go.Scatter(
                x=defects_df["log dist. [m]"],
                y=defects_df["clock_float"],
                mode="markers",
                marker=dict(
                    size=scaled_sizes,
                    color=defects_df[col],
                    colorscale=config["colorscale"],
                    cmin=config.get("color_range", [defects_df[col].min(), defects_df[col].max()])[0],
                    cmax=config.get("color_range", [defects_df[col].min(), defects_df[col].max()])[1],
                    colorbar=dict(
                        title=label,
                        thickness=20,
                        len=0.6,
                        y=0.5,
                        yanchor="middle",
                    ),
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=label,
                legendgroup=label,
                customdata=np.stack([
                    defects_df["joint number"].astype(str),
                    defects_df["component / anomaly identification"],
                    defects_df["depth [%]"].fillna(0),
                    defects_df["area_mm2"].fillna(0),
                    defects_df["clock_str"]
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "<b>Distance:</b> %{x:.2f} m<br>"
                    "<b>Clock:</b> %{customdata[4]}<br>"
                    "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
                    "<b>Area:</b> %{customdata[3]:.1f} mm²<br>"
                    "<b>Joint:</b> %{customdata[0]}<br>"
                    f"<b>{label}:</b> %{{marker.color:.1f}}<extra></extra>"
                ),
                visible=(i == 0)  # only first mode visible on load
            ))
    
    # Add a background grid representing clock positions
    for hour in range(1, 13):
        # Horizontal lines for each hour
        fig.add_shape(
            type="line",
            x0=defects_df["log dist. [m]"].min() - 1,
            x1=defects_df["log dist. [m]"].max() + 1,
            y0=hour,
            y1=hour,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below"
        )
    
    # Add joint boundaries with improved styling
    for _, row in joints_df.iterrows():
        x0 = row["log dist. [m]"]
        joint_num = row["joint number"]
        
        # Add joint annotation - moved to top of plot
        fig.add_annotation(
            x=x0,
            y=13.5,
            text=f"Joint {joint_num}",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color="rgba(50, 50, 50, 0.9)")
        )
    
    # Create buttons for switching between color modes
    buttons = []
    for i, label in enumerate(color_modes):
        # Count total number of traces per mode (can vary for categorical)
        trace_counts = []
        running_count = 0
        
        for mode_label, config in color_modes.items():
            if config.get("is_categorical", False) and config["column"] in defects_df.columns:
                unique_count = len(defects_df[config["column"]].dropna().unique())
                trace_counts.append(unique_count)
                running_count += unique_count
            else:
                trace_counts.append(1)
                running_count += 1
        
        # Calculate which traces should be visible for this mode
        visible = [False] * running_count
        start_idx = sum(trace_counts[:i])
        end_idx = start_idx + trace_counts[i]
        for j in range(start_idx, end_idx):
            visible[j] = True
        
        buttons.append(dict(
            label=label,
            method="update",
            args=[{"visible": visible},
                  {"title": f"Pipeline Defect Map — {label} Mode"}]
        ))
    
    # Compute maximum distance in your data
    max_dist = defects_df["log dist. [m]"].max()
    
    # Create ticks at 0,100,200,… up to (and including) the largest 100-multiple ≤ max_dist
    tickvals = np.arange(0, max_dist + 1, 100)
    
    # Label them as whole‐number strings
    ticktext = [f"{int(x)}" for x in tickvals]
    
    fig.update_xaxes(
        title_text="Distance Along Pipeline (m)",
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.2)",
        zeroline=False,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        showticklabels=True
    )
    
    fig.update_yaxes(
        title_text="Clock Position (hr)",
        tickmode="array",
        tickvals=list(range(1, 14)),
        ticktext=[f"{h}:00" for h in range(1, 14)],
        range=[0.5, 13.5],
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.2)",
        zeroline=False
    )
    
    # Add pipeline schematic on secondary y-axis for context
    if not joints_df.empty:
        # Configure secondary y-axis
        fig.update_yaxes(
            range=[-0.1, 0.1],
            visible=False,
            secondary_y=True
        )
    
    # Improve layout
    fig.update_layout(
        title={
            'text': "Pipeline Defect Map — Unwrapped Cylinder View",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=100, b=80),
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.05,
            y=1.12,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(240, 240, 240, 0.8)",
            bordercolor="rgba(100, 100, 100, 0.5)",
            # Set padding and button width for better appearance
            pad={"r": 10, "t": 10}
        )],
        height=700,
        annotations=[
            # Add a clear label for the visualization mode dropdown
            dict(
                text="Color By:",
                x=0.05,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="black"),
                align="left"
            )
        ]
    )
    
    # Add annotations explaining the visualization
    fig.add_annotation(
        text="Larger markers indicate larger defect areas",
        x=0.01,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left"
    )
    
    # Add clock position reference annotation
    fig.add_annotation(
        text="Clock positions: Top=13:00, Right=3:00, Bottom=6:00, Left=9:00",
        x=0.01,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left"
    )
    
    # Add critical zone indication
    fig.add_shape(
        type="rect",
        x0=defects_df["log dist. [m]"].min() - 1,
        x1=defects_df["log dist. [m]"].max() + 1,
        y0=1,
        y1=13,
        fillcolor="rgba(255, 0, 0, 0.05)",
        line=dict(width=0),
        layer="below"
    )
    
    return fig

def create_joint_defect_visualization(defects_df, joint_number):
    """
    Create a visualization of defects for a specific joint, representing defects
    as rectangles whose fill color maps to depth (%) between the joint's min & max,
    plus an interactive hover and a matching colorbar.
    """
    # 1) Filter
    joint_defects = defects_df[defects_df['joint number'] == joint_number].copy()
    if joint_defects.empty:
        return go.Figure().update_layout(
            title=f"No defects found for Joint {joint_number}",
            xaxis_title="Distance (m)",
            yaxis_title="Clock Position",
            plot_bgcolor="white"
        )
    
    # 2) Depth range for this joint
    depths = joint_defects['depth [%]'].astype(float)
    min_depth, max_depth = depths.min(), depths.max()
    
    # Ensure we have a valid range (avoid division by zero)
    if min_depth == max_depth:
        min_depth = max(0, min_depth - 1)
        max_depth = max_depth + 1

    # 3) Geometry constants
    min_dist = joint_defects['log dist. [m]'].min()
    max_dist = joint_defects['log dist. [m]'].max()
    pipe_diameter = 1.0  # m
    meters_per_clock_unit = np.pi * pipe_diameter / 12

    fig = go.Figure()
    colorscale_name = "YlOrRd"

    # 4) Draw each defect
    for _, defect in joint_defects.iterrows():
        x_center = defect['log dist. [m]']
        clock_pos = defect['clock_float']
        length_m = defect['length [mm]'] / 1000
        width_m = defect['width [mm]'] / 1000
        depth_pct = float(defect['depth [%]'])

        # rectangle corners
        w_clock = width_m / meters_per_clock_unit
        x0, x1 = x_center - length_m/2, x_center + length_m/2
        y0, y1 = clock_pos - w_clock/2, clock_pos + w_clock/2

        # Calculate normalized depth (0-1) for color mapping
        norm_depth = (depth_pct - min_depth) / (max_depth - min_depth)
        
        # Get color from colorscale using plotly's helper
        color = px.colors.sample_colorscale(colorscale_name, [norm_depth])[0]

        # Create custom data for hover info
        custom_data = [
            defect['clock'],
            depth_pct,
            defect['length [mm]'],
            defect['width [mm]'],
            defect.get('component / anomaly identification', 'Unknown')
        ]
        
        # Add rectangle for each defect with proper hover template
        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode='lines',
            fill='toself',
            fillcolor=color,  # Apply the color from the colorscale
            line=dict(color='black', width=1),
            hoveron='fills+points',
            hoverinfo='text',
            customdata=[custom_data] * 5,  # Same data for all 5 points
            hovertemplate="<b>Defect Information</b><br>" +
                          "Distance: %{x:.3f} m<br>" +
                          "Clock: %{customdata[0]}<br>" +
                          "Depth: %{customdata[1]:.1f}%<br>" +
                          "Length: %{customdata[2]:.1f} mm<br>" +
                          "Width: %{customdata[3]:.1f} mm<br>" +
                          "Type: %{customdata[4]}<extra></extra>",
            showlegend=False
        ))

    # 5) Invisible scatter for shared colorbar
    fig.add_trace(go.Scatter(
        x=[None]*len(depths),
        y=[None]*len(depths),
        mode='markers',
        marker=dict(
            color=depths,
            colorscale=colorscale_name,
            cmin=min_depth,
            cmax=max_depth,
            showscale=True,
            colorbar=dict(
                title="Depth (%)",
                thickness=15,
                len=0.5,
                tickformat=".1f"
            ),
            opacity=0
        ),
        showlegend=False
    ))

    # 6) Clock‐hour grid lines
    for hr in range(1,13):
        fig.add_shape(
            type="line",
            x0=min_dist - 0.2, x1=max_dist + 0.2,
            y0=hr, y1=hr,
            line=dict(color="lightgray", dash="dot", width=1),
            layer="below"
        )

    # 7) Layout
    fig.update_layout(
        title=f"Defect Map for Joint {joint_number}",
        xaxis_title="Distance Along Pipeline (m)",
        yaxis_title="Clock Position (hr)",
        plot_bgcolor="white",
        xaxis=dict(
            range=[min_dist - 0.2, max_dist + 0.2],
            showgrid=True, gridcolor="rgba(200,200,200,0.2)"
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(1,13)),
            ticktext=[f"{h}:00" for h in range(1,13)],
            range=[0.5,12.5],
            showgrid=True, gridcolor="rgba(200,200,200,0.2)"
        ),
        height=600, width=1200,
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def create_growth_rate_histogram(comparison_results):
    """
    Create a histogram showing the distribution of positive corrosion growth rates
    """
    if (not comparison_results['has_depth_data'] or 
        comparison_results['matches_df'].empty):
        fig = go.Figure()
        fig.add_annotation(
            text="No growth rate data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Get the matches dataframe
    matches_df = comparison_results['matches_df']
    
    # Filter for positive growth only
    positive_growth = matches_df[~matches_df['is_negative_growth']]
    
    if positive_growth.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No positive growth data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Use mm data if available, otherwise use percentage
    if comparison_results['has_wt_data']:
        growth_col = 'growth_rate_mm_per_year'
        x_title = 'Growth Rate (mm/year)'
    else:
        growth_col = 'growth_rate_pct_per_year'
        x_title = 'Growth Rate (% points/year)'
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=positive_growth[growth_col],
        nbinsx=20,
        marker=dict(
            color='rgba(255, 100, 102, 0.7)',
            line=dict(color='rgba(255, 100, 102, 1)', width=1)
        ),
        name='Positive Growth Rates'
    ))
    
    # Add vertical line at average growth rate
    mean_growth = positive_growth[growth_col].mean()
    
    fig.add_shape(
        type="line",
        x0=mean_growth, x1=mean_growth,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=mean_growth,
        y=1,
        yref="paper",
        text=f"Mean: {mean_growth:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-30
    )
    
    # Layout
    fig.update_layout(
        title='Distribution of Positive Defect Growth Rates',
        xaxis_title=x_title,
        yaxis_title='Count',
        bargap=0.1,
        bargroupgap=0.1,
        height=500
    )
    
    return fig

def create_negative_growth_plot(comparison_results):
    """
    Create a scatter plot highlighting negative growth defects
    """
    if (not comparison_results['has_depth_data'] or 
        comparison_results['matches_df'].empty):
        fig = go.Figure()
        fig.add_annotation(
            text="No growth rate data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Get the matches dataframe
    matches_df = comparison_results['matches_df']
    
    # Split into negative and positive growth
    negative_growth = matches_df[matches_df['is_negative_growth']]
    positive_growth = matches_df[~matches_df['is_negative_growth']]
    
    if negative_growth.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No negative growth anomalies detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Use mm data if available, otherwise use percentage
    if comparison_results['has_wt_data']:
        old_depth_col = 'old_depth_mm'
        new_depth_col = 'new_depth_mm'
        growth_col = 'growth_rate_mm_per_year'
        y_title = 'Growth Rate (mm/year)'
    else:
        old_depth_col = 'old_depth_pct'
        new_depth_col = 'new_depth_pct'
        growth_col = 'growth_rate_pct_per_year'
        y_title = 'Growth Rate (% points/year)'
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add positive growth defects
    if not positive_growth.empty:
        fig.add_trace(go.Scatter(
            x=positive_growth['log_dist'],
            y=positive_growth[growth_col],
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                opacity=0.5
            ),
            name='Positive Growth',
            hovertemplate=(
                "<b>Location:</b> %{x:.2f}m<br>"
                f"<b>Growth Rate:</b> %{{y:.3f}}{y_title.split(' ')[1]}<br>"
                f"<b>Old Depth:</b> %{{customdata[0]:.2f}}{' mm' if comparison_results['has_wt_data'] else '%'}<br>"
                f"<b>New Depth:</b> %{{customdata[1]:.2f}}{' mm' if comparison_results['has_wt_data'] else '%'}<br>"
                "<b>Type:</b> %{customdata[2]}"
                "<extra></extra>"
            ),
            customdata=np.column_stack((
                positive_growth[old_depth_col],
                positive_growth[new_depth_col],
                positive_growth['defect_type']
            ))
        ))
    
    # Add negative growth defects
    fig.add_trace(go.Scatter(
        x=negative_growth['log_dist'],
        y=negative_growth[growth_col],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            opacity=0.7,
            symbol='triangle-down',
            line=dict(width=1, color='black')
        ),
        name='Negative Growth (Anomaly)',
        hovertemplate=(
            "<b>Location:</b> %{x:.2f}m<br>"
            f"<b>Growth Rate:</b> %{{y:.3f}}{y_title.split(' ')[1]}<br>"
            f"<b>Old Depth:</b> %{{customdata[0]:.2f}}{' mm' if comparison_results['has_wt_data'] else '%'}<br>"
            f"<b>New Depth:</b> %{{customdata[1]:.2f}}{' mm' if comparison_results['has_wt_data'] else '%'}<br>"
            "<b>Type:</b> %{customdata[2]}"
            "<extra></extra>"
        ),
        customdata=np.column_stack((
            negative_growth[old_depth_col],
            negative_growth[new_depth_col],
            negative_growth['defect_type']
        ))
    ))
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=min(matches_df['log_dist']),
        x1=max(matches_df['log_dist']),
        y0=0, y1=0,
        line=dict(color="black", width=1, dash="dash"),
    )
    
    # Layout
    fig.update_layout(
        title='Defect Growth Rate vs Location (Highlighting Negative Growth)',
        xaxis_title='Distance Along Pipeline (m)',
        yaxis_title=y_title,
        height=500,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig