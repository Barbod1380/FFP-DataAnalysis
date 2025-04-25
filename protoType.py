import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Pipeline Inspection Visualization", layout="wide")

# Add app title
st.title("Pipeline Inspection Data Visualization")

def process_pipeline_data(df):
    """
    Process the pipeline inspection data into two separate tables:
    1. joints_df: Contains unique joint information
    2. defects_df: Contains defect information with joint associations
    
    Parameters:
    - df: DataFrame with the raw pipeline data
    
    Returns:
    - joints_df: DataFrame with joint information
    - defects_df: DataFrame with defect information
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # 1. Replace empty strings with NaN for proper handling
    df_copy = df_copy.replace(r'^\s*$', np.nan, regex=True)
    
    # 2. Convert numeric columns to appropriate types
    numeric_columns = [
        'joint number', 
        'joint length [m]', 
        'wt nom [mm]', 
        'up weld dist. [m]', 
        'depth [%]', 
        'length [mm]', 
        'width [mm]'
    ]
    
    for col in numeric_columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # 4. Sort by log distance to ensure proper order for forward fill
    if 'log dist. [m]' in df_copy.columns:
        df_copy = df_copy.sort_values('log dist. [m]')
    
    # 5. Create joints_df with only the specified columns
    joints_df = df_copy[df_copy['joint number'].notna()][['log dist. [m]', 'joint number', 'joint length [m]', 'wt nom [mm]']].copy()
    
    # 6. Drop duplicate joint numbers if any
    joints_df = joints_df.drop_duplicates(subset=['joint number'])
    joints_df = joints_df.reset_index().drop(columns = ['index'])
    
    # 7. Create defects_df - records with length and width values
    # First, forward fill joint number to associate defects with joints
    df_copy['joint number'] = df_copy['joint number'].fillna(method='ffill')
    
    # Filter for records that have both length and width values
    defects_df = df_copy[
        df_copy['length [mm]'].notna() & 
        df_copy['width [mm]'].notna()
    ].copy()
    
    # Select only the specified columns
    defect_columns = [
        'log dist. [m]',
        'component / anomaly identification',
        'joint number',
        'up weld dist. [m]',
        'clock',
        'depth [%]',
        'length [mm]',
        'width [mm]',
        'surface location'
    ]
    
    # Check which columns exist in the data
    available_columns = [col for col in defect_columns if col in df_copy.columns]
    
    # Select only available columns
    defects_df = defects_df[available_columns]
    defects_df = defects_df.reset_index().drop(columns = ['index'])
    
    return joints_df, defects_df


# Define the parse_clock function
def parse_clock(clock_str):
    try:
        hours, minutes = map(int, clock_str.split(":"))
        return hours + minutes / 60
    except Exception:
        return np.nan

# Define the decimal_to_clock_str function
def decimal_to_clock_str(decimal_hours):
    """
    Convert decimal hours to clock format string.
    Example: 5.9 → "5:54"
    
    Parameters:
    - decimal_hours: Clock position in decimal format
    
    Returns:
    - String in clock format "H:MM"
    """
    if pd.isna(decimal_hours):
        return "Unknown"
    
    # Ensure the value is between 1 and 12
    if decimal_hours < 1:
        decimal_hours += 12
    elif decimal_hours > 12:
        decimal_hours = decimal_hours % 12
        if decimal_hours == 0:
            decimal_hours = 12
    
    hours = int(decimal_hours)
    minutes = int((decimal_hours - hours) * 60)
    
    return f"{hours}:{minutes:02d}"

# Define the create_unwrapped_pipeline_visualization function
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

# Define the joint-specific visualization function
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

# Main app layout
st.write("Upload a pipeline inspection CSV file to visualize defects")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    # Process the pipeline data
    joints_df, defects_df = process_pipeline_data(df)
    
    # Process clock and area data
    defects_df["clock_float"] = defects_df["clock"].apply(parse_clock)
    defects_df["area_mm2"] = defects_df["length [mm]"] * defects_df["width [mm]"]
    defects_df["joint number"] = defects_df["joint number"].astype("Int64")
    
    # Display the tables
    st.header("Data Preview")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Joints Table (Top 10 Records)")
        st.dataframe(joints_df.head(10))
    
    with col2:
        st.subheader("Defects Table (Top 10 Records)")
        st.dataframe(defects_df.head(10))
    
    # Visualization section
    st.header("Visualization")
    
    # Visualization type selection
    viz_type = st.radio(
        "Select Visualization Type",
        ["Complete Pipeline", "Joint-by-Joint"],
        horizontal=True
    )
    
    if viz_type == "Complete Pipeline":
        # Button to show visualization
        if st.button("Show Complete Pipeline Visualization"):
            st.subheader("Pipeline Defect Map")
            fig = create_unwrapped_pipeline_visualization(defects_df, joints_df)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Joint selection
        available_joints = sorted(joints_df["joint number"].unique())
        
        # Format the joint numbers with their distance for better context
        joint_options = {}
        for joint in available_joints:
            joint_row = joints_df[joints_df["joint number"] == joint].iloc[0]
            distance = joint_row["log dist. [m]"]
            joint_options[f"Joint {joint} (at {distance:.1f}m)"] = joint
        
        selected_joint_label = st.selectbox(
            "Select Joint to Visualize",
            options=list(joint_options.keys())
        )
        
        selected_joint = joint_options[selected_joint_label]
        
        # Button to show joint visualization
        if st.button("Show Joint Visualization"):
            st.subheader(f"Defect Map for {selected_joint_label}")
            fig = create_joint_defect_visualization(defects_df, selected_joint)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a CSV file to begin analysis.")