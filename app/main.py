"""
Main module for the Pipeline Analysis Streamlit application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime

# Import app modules
from app.components import (
    load_css, show_step_indicator, get_logo_base64, 
    info_box, custom_metric, create_welcome_screen, create_sidebar,
    create_data_download_links
)
from app.config import (
    APP_TITLE, APP_SUBTITLE, ENCODING_OPTIONS, 
    DEFAULT_PIPE_DIAMETER, DEFAULT_KNN_NEIGHBORS,
    DEFAULT_DISTANCE_TOLERANCE, DEFAULT_CLOCK_TOLERANCE
)

# Import core modules
from core.column_mapping import (
    suggest_column_mapping, apply_column_mapping, 
    get_missing_required_columns, STANDARD_COLUMNS, REQUIRED_COLUMNS
)
from core.data_processing import process_pipeline_data
from core.multi_year_analysis import compare_defects

# Import analysis modules
from analysis.defect_analysis import (
    create_dimension_distribution_plots, create_dimension_statistics_table,
    create_combined_dimensions_plot, create_joint_summary
)
from analysis.growth_analysis import (
    correct_negative_growth_rates, create_growth_summary_table, 
    create_highest_growth_table
)

# Import visualization modules
from visualization.pipeline_viz import create_unwrapped_pipeline_visualization
from visualization.joint_viz import create_joint_defect_visualization
from visualization.comparison_viz import (
    create_comparison_stats_plot, create_new_defect_types_plot, 
    create_negative_growth_plot, create_growth_rate_histogram, create_multi_dimensional_growth_plot
)

# Import utility functions
from utils.format_utils import float_to_clock, parse_clock

def initialize_session_state():
    """Initialize or update session state variables."""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}  # Will store {year: {'joints_df': df1, 'defects_df': df2}}
    if 'current_year' not in st.session_state:
        st.session_state.current_year = None
    if 'file_upload_key' not in st.session_state:
        st.session_state.file_upload_key = 0  # For forcing file uploader to clear
    if 'active_step' not in st.session_state:
        st.session_state.active_step = 1
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'corrected_results' not in st.session_state:
        st.session_state.corrected_results = None
    if 'comparison_years' not in st.session_state:
        st.session_state.comparison_years = None
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    # Add tab state management
    if 'comparison_viz_tab' not in st.session_state:
        st.session_state.comparison_viz_tab = 0
    if 'correction_dimension' not in st.session_state:
        st.session_state.correction_dimension = 'depth'
    if 'growth_analysis_dimension' not in st.session_state:
        st.session_state.growth_analysis_dimension = 'depth'

def load_csv_with_encoding(file):
    """
    Try to load a CSV file with different encodings.
    
    Parameters:
    - file: Uploaded file object
    
    Returns:
    - Tuple of (DataFrame, encoding)
    
    Raises:
    - ValueError if the file cannot be loaded with any encoding
    """
    for encoding in ENCODING_OPTIONS:
        try:
            # Reset file pointer to the beginning
            file.seek(0)
            
            # Try to read with current encoding
            df = pd.read_csv(
                file, 
                encoding=encoding,
                sep=None,  # Auto-detect separator
                engine='python',  # More flexible engine
                on_bad_lines='warn'  # Continue despite bad lines
            )
            
            # Check and convert clock column if needed
            if 'clock' in df.columns:
                # Check if any values are numeric (floating point)
                if df['clock'].dtype.kind in 'fi' or any(isinstance(x, (int, float)) for x in df['clock'].dropna()):
                    st.info("Converting numeric clock values to HH:MM format")
                    # Convert numeric values to clock format
                    df['clock'] = df['clock'].apply(
                        lambda x: float_to_clock(float(x)) if pd.notna(x) and isinstance(x, (int, float)) else x
                    )
                
                # For string values that don't look like clock format (HH:MM)
                clock_pattern = re.compile(r'^\d{1,2}:\d{2}$')
                non_standard = df['clock'].apply(
                    lambda x: pd.notna(x) and isinstance(x, str) and not clock_pattern.match(x)
                ).any()
                
                if non_standard:
                    info_box("Some clock values may not be in standard HH:MM format", box_type="warning")
            return df, encoding
            
        except Exception as e:
            continue  # Try next encoding
    
    # If all encodings fail
    raise ValueError(f"Failed to load the file with any of the encodings: {', '.join(ENCODING_OPTIONS)}")

def process_data_section(df, selected_year):
    """
    Display the data processing section for a newly uploaded file.
    
    Parameters:
    - df: DataFrame with the uploaded data
    - selected_year: Selected year for the data
    
    Returns:
    - Processed flag (True if data was processed)
    """
    # Column mapping process in a collapsible section
    with st.expander("Column Mapping", expanded=True):
        st.markdown('<div class="section-header">Map Columns for Standardization</div>', unsafe_allow_html=True)
        
        # Update active step
        st.session_state.active_step = 2
        
        st.info("""
            **Column Mapping Instructions:**
            Match your file's columns to standard column names. Required fields are marked with *.
            This mapping ensures consistent analysis across different data formats.
        """)
        
        # Get suggested column mapping
        suggested_mapping = suggest_column_mapping(df)
        
        # Create UI for mapping confirmation
        st.write("Confirm the mapping between your file's columns and standard columns:")
        
        confirmed_mapping = {}
        all_columns = [None] + df.columns.tolist()
        
        # Create three columns for the mapping UI to save space
        col1, col2, col3 = st.columns(3)
        
        # Split the standard columns into three groups
        third = len(STANDARD_COLUMNS) // 3
        remaining = len(STANDARD_COLUMNS) % 3
        
        # Calculate split points for columns
        if remaining == 1:
            # First column gets one extra
            split1 = third + 1
            split2 = split1 + third
        elif remaining == 2:
            # First and second columns get one extra each
            split1 = third + 1
            split2 = split1 + third + 1
        else:
            # Even distribution
            split1 = third
            split2 = split1 + third
        
        # First column of mappings
        with col1:
            for std_col in STANDARD_COLUMNS[:split1]:
                suggested = suggested_mapping.get(std_col)
                index = 0 if suggested is None else all_columns.index(suggested)
                
                is_required = std_col in REQUIRED_COLUMNS
                label = f"{std_col}" + (" *" if is_required else "")
                
                selected = st.selectbox(
                    label,
                    options=all_columns,
                    index=index,
                    key=f"map_{selected_year}_{std_col}"
                )
                confirmed_mapping[std_col] = selected
        
        # Second column of mappings
        with col2:
            for std_col in STANDARD_COLUMNS[split1:split2]:
                suggested = suggested_mapping.get(std_col)
                index = 0 if suggested is None else all_columns.index(suggested)
                
                is_required = std_col in REQUIRED_COLUMNS
                label = f"{std_col}" + (" *" if is_required else "")
                
                selected = st.selectbox(
                    label,
                    options=all_columns,
                    index=index,
                    key=f"map_{selected_year}_{std_col}_col2"
                )
                confirmed_mapping[std_col] = selected
        
        # Third column of mappings
        with col3:
            for std_col in STANDARD_COLUMNS[split2:]:
                suggested = suggested_mapping.get(std_col)
                index = 0 if suggested is None else all_columns.index(suggested)
                
                is_required = std_col in REQUIRED_COLUMNS
                label = f"{std_col}" + (" *" if is_required else "")
                
                selected = st.selectbox(
                    label,
                    options=all_columns,
                    index=index,
                    key=f"map_{selected_year}_{std_col}_col3"
                )
                confirmed_mapping[std_col] = selected
        
        # Add note about required fields
        st.markdown('<div style="margin-top:10px;font-size:0.8em;">* Required fields</div>', unsafe_allow_html=True)
        
        # Check for missing required columns
        missing_cols = get_missing_required_columns(confirmed_mapping)
        if missing_cols:
            info_box(f"Missing required columns: {', '.join(missing_cols)}. You may proceed, but functionality may be limited.", "warning")

    # Add this after the column mapping expander
    with st.expander("Pipeline Specifications", expanded=True):
        st.markdown('<div class="section-header">Enter Pipeline Parameters</div>', unsafe_allow_html=True)
        
        # Pipe diameter input with a reasonable default and validation
        pipe_diameter = st.number_input(
            "Pipe Diameter (m)",
            min_value=0.1,
            max_value=3.0,  # Reasonable range for pipeline diameters
            value=DEFAULT_PIPE_DIAMETER,
            step=0.1,
            format="%.2f",
            help="Enter the pipeline diameter in meters"
        )

    # Process and add button
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    process_col1, process_col2 = st.columns([1, 3])
    with process_col1:
        process_button = st.button(
            f"Process {selected_year} Data", 
            key=f"process_data_{selected_year}",
            use_container_width=True
        )
    
    if process_button:
        # Update active step
        st.session_state.active_step = 3
        
        with st.spinner(f"Processing {selected_year} data..."):
            # Apply the mapping to rename columns
            standardized_df = apply_column_mapping(df, confirmed_mapping)
            
            # Process the pipeline data
            joints_df, defects_df = process_pipeline_data(standardized_df)
            
            # Add progress bar for processing
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulating work
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Process clock and area data
            if 'clock' in defects_df.columns:
                # First ensure all clock values are in string format
                defects_df['clock'] = defects_df['clock'].astype(str)
                
                # Check if string values don't match the expected format
                clock_pattern = re.compile(r'^\d{1,2}:\d{2}$')
                non_standard = defects_df['clock'].apply(
                    lambda x: pd.notna(x) and not clock_pattern.match(x) and x != 'nan'
                ).any()
                
                if non_standard:
                    info_box("Some clock values may not be in standard HH:MM format. These will be handled as NaN.", "warning")
                    # Try to fix non-standard formats
                    defects_df['clock'] = defects_df['clock'].apply(
                        lambda x: float_to_clock(float(x)) if pd.notna(x) and x != 'nan' and not clock_pattern.match(x) else x
                    )
                
                # Now convert to float for visualization
                defects_df["clock_float"] = defects_df["clock"].apply(parse_clock)
            
            if 'length [mm]' in defects_df.columns and 'width [mm]' in defects_df.columns:
                defects_df["area_mm2"] = defects_df["length [mm]"] * defects_df["width [mm]"]
            
            if 'joint number' in defects_df.columns:
                defects_df["joint number"] = defects_df["joint number"].astype("Int64")
            
            # Store in session state
            st.session_state.datasets[selected_year] = {
                'joints_df': joints_df,
                'defects_df': defects_df,
                'pipe_diameter': pipe_diameter  # Store the pipe diameter
            }
            st.session_state.current_year = selected_year
            
            # Force the file uploader to reset
            st.session_state.file_upload_key += 1
            
            # Show success message and then rerun
            st.success(f"Successfully processed {selected_year} data")
            st.rerun()
            
        return True
    
    return False

def single_year_analysis_tab():
    """
    Display the single year analysis tab with various analysis options.
    """
    st.markdown('<h2 class="section-header">Single Year Analysis</h2>', unsafe_allow_html=True)
    
    # Select year to analyze
    years = sorted(st.session_state.datasets.keys())
    col1, col2 = st.columns([2, 2])
    
    with col1:
        selected_analysis_year = st.selectbox(
            "Select Year to Analyze",
            options=years,
            index=years.index(st.session_state.current_year) if st.session_state.current_year in years else 0,
            key="year_selector_single_analysis"
        )
    
    # Get the selected dataset
    joints_df = st.session_state.datasets[selected_analysis_year]['joints_df']
    defects_df = st.session_state.datasets[selected_analysis_year]['defects_df']
    
    # Display dataset summary in custom metrics
    with col2:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(custom_metric("Joints", f"{len(joints_df)}"), unsafe_allow_html=True)
        with col_b:
            st.markdown(custom_metric("Defects", f"{len(defects_df)}"), unsafe_allow_html=True)
        with col_c:
            if 'depth [%]' in defects_df.columns:
                max_depth = defects_df['depth [%]'].max()
                st.markdown(custom_metric("Max Depth", f"{max_depth:.1f}%"), unsafe_allow_html=True)
            else:
                st.markdown(custom_metric("Max Depth", "N/A"), unsafe_allow_html=True)
    
    # Create tabs for different analysis types within a card container
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    analysis_tabs = st.tabs(["Data Preview", "Defect Dimensions", "Visualizations"])
    
    # Tab 1: Data Preview
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_analysis_year} Joints")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(joints_df.head(5), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add download buttons
            href = create_data_download_links(joints_df, "joints", selected_analysis_year)
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            st.subheader(f"{selected_analysis_year} Defects")
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(defects_df.head(5), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add download buttons 
            href = create_data_download_links(defects_df, "defects", selected_analysis_year)
            st.markdown(href, unsafe_allow_html=True)
    
    # Tab 2: Defect Dimensions Analysis
    with analysis_tabs[1]:
        st.subheader("Defect Dimension Analysis")
        
        # Display dimension statistics table
        st.markdown("<div class='section-header'>Dimension Statistics</div>", unsafe_allow_html=True)
        stats_df = create_dimension_statistics_table(defects_df)
        if not stats_df.empty:
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(stats_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            info_box("No dimension data available for analysis.", "warning")
        
        # Create distribution plots
        dimension_figs = create_dimension_distribution_plots(defects_df)
        
        if dimension_figs:
            st.markdown("<div class='section-header' style='margin-top:20px;'>Dimension Distributions</div>", unsafe_allow_html=True)
            # Create columns for the plots
            cols = st.columns(min(len(dimension_figs), 3))
            
            # Display each dimension distribution
            for i, (col_name, fig) in enumerate(dimension_figs.items()):
                with cols[i % len(cols)]:
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Display combined dimensions plot
            st.markdown("<div class='section-header' style='margin-top:20px;'>Defect Dimensions Relationship</div>", unsafe_allow_html=True)
            combined_fig = create_combined_dimensions_plot(defects_df)
            st.plotly_chart(combined_fig, use_container_width=True)
        else:
            info_box("No dimension data available for plotting distributions.", "warning")
    
    # Tab 3: Pipeline Visualizations
    with analysis_tabs[2]:
        st.subheader("Pipeline Visualization")
        
        # Visualization type selection with improved UI
        viz_col1, viz_col2 = st.columns([2, 2])
        
        with viz_col1:
            viz_type = st.radio(
                "Select Visualization Type",
                ["Complete Pipeline", "Joint-by-Joint"],
                horizontal=True,
                key="viz_type_single_analysis"
            )

        # Add this in the Complete Pipeline visualization section
        if viz_type == "Complete Pipeline":
            # Add filtering options
            with st.expander("Filter Defects", expanded=False):
                st.subheader("Filter Defects by Properties")
                
                # Create columns for different filter types
                filter_col1, filter_col2 = st.columns(2)
                
                # Initialize filter variables
                apply_depth_filter = False
                apply_length_filter = False
                apply_width_filter = False
                
                with filter_col1:
                    # Depth filter
                    if 'depth [%]' in defects_df.columns:
                        # Get min/max values with safety checks for non-numeric values
                        depth_values = pd.to_numeric(defects_df['depth [%]'], errors='coerce')
                        depth_min = float(depth_values.min())
                        depth_max = float(depth_values.max())
                        
                        apply_depth_filter = st.checkbox("Filter by Depth", key="filter_depth")
                        if apply_depth_filter:
                            min_depth, max_depth = st.slider(
                                "Depth Range (%)",
                                min_value=depth_min,
                                max_value=depth_max,
                                value=(depth_min, depth_max),
                                step=0.5,
                                key="depth_range"
                            )
                    
                    # Length filter
                    if 'length [mm]' in defects_df.columns:
                        # Get min/max values with safety checks
                        length_values = pd.to_numeric(defects_df['length [mm]'], errors='coerce')
                        length_min = float(length_values.min())
                        length_max = float(length_values.max() + 10)  # Add small margin
                        
                        apply_length_filter = st.checkbox("Filter by Length", key="filter_length")
                        if apply_length_filter:
                            min_length, max_length = st.slider(
                                "Length Range (mm)",
                                min_value=length_min,
                                max_value=length_max,
                                value=(length_min, length_max),
                                step=5.0,
                                key="length_range"
                            )
                
                with filter_col2:
                    # Width filter
                    if 'width [mm]' in defects_df.columns:
                        # Get min/max values with safety checks
                        width_values = pd.to_numeric(defects_df['width [mm]'], errors='coerce')
                        width_min = float(width_values.min())
                        width_max = float(width_values.max() + 10)  # Add small margin
                        
                        apply_width_filter = st.checkbox("Filter by Width", key="filter_width")
                        if apply_width_filter:
                            min_width, max_width = st.slider(
                                "Width Range (mm)",
                                min_value=width_min,
                                max_value=width_max,
                                value=(width_min, width_max),
                                step=5.0,
                                key="width_range"
                            )
                    
                    # Add defect type filter if available
                    if 'component / anomaly identification' in defects_df.columns:
                        defect_types = ['All Types'] + sorted(defects_df['component / anomaly identification'].unique().tolist())
                        selected_defect_type = st.selectbox(
                            "Filter by Defect Type",
                            options=defect_types,
                            key="defect_type_filter"
                        )
            
            # Button to show visualization with filtering
            if st.button("Generate Pipeline Visualization", key="show_pipeline_single_analysis", use_container_width=True):
                st.markdown("<div class='section-header'>Pipeline Defect Map</div>", unsafe_allow_html=True)
                
                # Show a spinner during calculation
                with st.spinner("Generating pipeline visualization..."):
                    # Apply filters to the defects dataframe
                    filtered_defects = defects_df.copy()
                    
                    filter_applied = False
                    filter_description = []
                    
                    # Apply depth filter if checked
                    if apply_depth_filter and 'depth [%]' in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects['depth [%]'], errors='coerce') >= min_depth) & 
                            (pd.to_numeric(filtered_defects['depth [%]'], errors='coerce') <= max_depth)
                        ]
                        filter_applied = True
                        filter_description.append(f"Depth: {min_depth}% to {max_depth}%")
                    
                    # Apply length filter if checked
                    if apply_length_filter and 'length [mm]' in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects['length [mm]'], errors='coerce') >= min_length) & 
                            (pd.to_numeric(filtered_defects['length [mm]'], errors='coerce') <= max_length)
                        ]
                        filter_applied = True
                        filter_description.append(f"Length: {min_length}mm to {max_length}mm")
                    
                    # Apply width filter if checked
                    if apply_width_filter and 'width [mm]' in filtered_defects.columns:
                        filtered_defects = filtered_defects[
                            (pd.to_numeric(filtered_defects['width [mm]'], errors='coerce') >= min_width) & 
                            (pd.to_numeric(filtered_defects['width [mm]'], errors='coerce') <= max_width)
                        ]
                        filter_applied = True
                        filter_description.append(f"Width: {min_width}mm to {max_width}mm")
                    
                    # Apply defect type filter if selected
                    if 'component / anomaly identification' in filtered_defects.columns and selected_defect_type != 'All Types':
                        filtered_defects = filtered_defects[
                            filtered_defects['component / anomaly identification'] == selected_defect_type
                        ]
                        filter_applied = True
                        filter_description.append(f"Type: {selected_defect_type}")
                    
                    # Create and display statistics about filtered data
                    if filter_applied:
                        original_count = len(defects_df)
                        filtered_count = len(filtered_defects)
                        
                        # Join filter descriptions
                        filter_text = ", ".join(filter_description)
                        
                        # Show filter summary
                        st.info(f"Showing {filtered_count} defects out of {original_count} total defects ({filtered_count/original_count*100:.1f}%)\nFilters applied: {filter_text}")
                    
                    # Generate the visualization with filtered data
                    fig = create_unwrapped_pipeline_visualization(filtered_defects, joints_df)
                    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                    
                    # Display guide
                    st.info("""
                    **Visualization Guide:**
                    - Each point represents a defect
                    - X-axis shows distance along pipeline in meters
                    - Y-axis shows clock position
                    - Color indicates defect depth percentage
                    """)

        else:  # Joint-by-joint visualization
            # Joint selection with improved UI
            available_joints = sorted(joints_df["joint number"].unique())
            
            # Format joint numbers with distance
            joint_options = {}
            for joint in available_joints:
                joint_row = joints_df[joints_df["joint number"] == joint].iloc[0]
                distance = joint_row["log dist. [m]"]
                joint_options[f"Joint {joint} (at {distance:.1f}m)"] = joint
            
            joint_col1, joint_col2 = st.columns([3, 1])

            with joint_col1:
                selected_joint_label = st.selectbox(
                    "Select Joint to Visualize",
                    options=list(joint_options.keys()),
                    key="joint_selector_single_analysis"
                )

            with joint_col2:
                view_mode = st.radio(
                    "View Mode",
                    ["2D Unwrapped"],
                    key="joint_view_mode"
                )
            
            selected_joint = joint_options[selected_joint_label]

            # Button to show joint visualization
            if st.button("Generate Joint Visualization", key="show_joint_single_analysis", use_container_width=True):
                st.markdown(f"<div class='section-header'>Defect Map for {selected_joint_label}</div>", unsafe_allow_html=True)
                
                # Get joint summary
                joint_summary = create_joint_summary(defects_df, joints_df, selected_joint)
                
                # Create summary panel with metrics
                summary_cols = st.columns(4)
                
                with summary_cols[0]:
                    st.markdown(custom_metric("Defect Count", joint_summary["defect_count"]), unsafe_allow_html=True)
                
                with summary_cols[1]:
                    # Format defect types as a string
                    if joint_summary["defect_types"]:
                        defect_types_str = ", ".join([f"{count} {type_}" for type_, count in joint_summary["defect_types"].items()])
                        if len(defect_types_str) < 30:
                            st.markdown(custom_metric("Defect Types", defect_types_str), unsafe_allow_html=True)
                        else:
                            st.markdown(custom_metric("Defect Types", f"{len(joint_summary['defect_types'])} types"), unsafe_allow_html=True)
                            st.write(defect_types_str)
                    else:
                        st.markdown(custom_metric("Defect Types", "None"), unsafe_allow_html=True)
                
                with summary_cols[2]:
                    length_value = joint_summary["joint_length"]
                    if length_value != "N/A":
                        length_value = f"{length_value:.2f}m"
                    st.markdown(custom_metric("Joint Length", length_value), unsafe_allow_html=True)
                
                with summary_cols[3]:
                    st.markdown(custom_metric("Severity Rank", joint_summary["severity_rank"]), unsafe_allow_html=True)
                
                # Add a divider for clarity
                st.markdown("<hr style='margin:20px 0;border-color:#e0e0e0;'>", unsafe_allow_html=True)
                
                # Show the visualization with better handling
                with st.spinner("Generating joint visualization..."):
                    fig = create_joint_defect_visualization(defects_df, selected_joint)
                    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close the card container

def display_comparison_visualization_tabs(comparison_results, earlier_year, later_year):
    """Display the consolidated visualization tabs for comparison results."""
    
    # Create visualization tabs
    viz_tabs = st.tabs([
        "New vs Common", "New Defect Types", "Negative Growth Correction", "Growth Rate Analysis"
    ])
    
    with viz_tabs[0]:
        # Pie chart of common vs new defects
        pie_fig = create_comparison_stats_plot(comparison_results)
        st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})
    
    with viz_tabs[1]:
        # Bar chart of new defect types
        bar_fig = create_new_defect_types_plot(comparison_results)
        st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Negative Growth Correction tab
    with viz_tabs[2]:
        st.subheader("Growth Analysis with Correction")
        
        # Add dimension selection to this tab
        st.markdown("#### Select Dimension for Analysis")
        
        # Get available dimensions
        available_dimensions = []
        if comparison_results.get('has_depth_data', False):
            available_dimensions.append('depth')
        if comparison_results.get('has_length_data', False):
            available_dimensions.append('length')
        if comparison_results.get('has_width_data', False):
            available_dimensions.append('width')
        
        if not available_dimensions:
            st.warning("No growth data available for any dimension.")
        else:
            # Initialize dimension state if not exists
            if st.session_state.correction_dimension not in available_dimensions:
                st.session_state.correction_dimension = available_dimensions[0]
            
            # Simple dimension selection with proper session state management
            selected_dimension = st.selectbox(
                "Choose dimension to analyze",
                options=available_dimensions,
                index=available_dimensions.index(st.session_state.correction_dimension) if st.session_state.correction_dimension in available_dimensions else 0,
                key=f"correction_dimension_{earlier_year}_{later_year}",
                help="Select which defect dimension to analyze for growth patterns"
            )
            
            # Update session state
            st.session_state.correction_dimension = selected_dimension
            
            # Show growth plot for selected dimension
            st.markdown(f"#### {selected_dimension.title()} Growth Data")
            
            # Show the selected dimension plot
            original_plot = create_negative_growth_plot(comparison_results, dimension=selected_dimension)
            st.plotly_chart(original_plot, use_container_width=True, config={'displayModeBar': False})
            
            # Only show correction controls for depth dimension
            if selected_dimension == 'depth':
                # Check if depth data is available for correction
                if not (comparison_results.get('has_depth_data', False) and 'is_negative_growth' in comparison_results['matches_df'].columns):
                    st.warning("No depth growth data available for correction. Make sure both datasets have depth measurements.")
                else:
                    # Display negative depth growth summary
                    neg_count = comparison_results['matches_df']['is_negative_growth'].sum()
                    total_count = len(comparison_results['matches_df'])
                    pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                    
                    st.markdown("#### Negative Depth Growth Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(custom_metric("Total Defects", f"{total_count}"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(custom_metric("Negative Growth", f"{neg_count}"), unsafe_allow_html=True)
                    with col3:
                        st.markdown(custom_metric("Percentage", f"{pct:.1f}%"), unsafe_allow_html=True)
                    
                    if neg_count > 0:
                        st.info("Negative depth growth rates are likely measurement errors and can be corrected using similar defects in the same joint.")
                    else:
                        st.success("No negative depth growth detected - no correction needed!")
                    
                    # Show corrected results if available
                    if st.session_state.corrected_results is not None:
                        corrected_results = st.session_state.corrected_results
                        correction_info = corrected_results.get('correction_info', {})
                        
                        if correction_info.get("success", False):
                            st.markdown("#### Correction Results")
                            st.success(f"Successfully corrected {correction_info['corrected_count']} out of {correction_info['total_negative']} negative depth growth defects.")
                            
                            if correction_info['uncorrected_count'] > 0:
                                st.warning(f"Could not correct {correction_info['uncorrected_count']} defects in {len(correction_info['uncorrected_joints'])} joints due to insufficient similar defects.")
                            
                            # Show corrected growth plot
                            st.markdown("#### Corrected Depth Growth Data")
                            corrected_plot = create_negative_growth_plot(corrected_results, dimension='depth')
                            st.plotly_chart(corrected_plot, use_container_width=True, config={'displayModeBar': False})
                            
                            # Legend
                            st.markdown("""
                            **Legend:**
                            - Blue circles: Positive growth (unchanged)
                            - Red triangles: Negative growth (uncorrected)
                            - Green diamonds: Corrected growth (formerly negative)
                            """)
                    
                    # Show KNN correction controls only if there are negative growth defects
                    if neg_count > 0:
                        # Check if joint numbers are available for KNN correction
                        has_joint_num = comparison_results.get('has_joint_num', False)
                        if not has_joint_num:
                            st.warning("""
                            **Joint numbers not available for correction**
                            
                            The KNN correction requires the 'joint number' column to be present in your defect data.
                            Please ensure both datasets have this column properly mapped.
                            """)
                        else:
                            # KNN correction controls
                            st.markdown("#### Apply KNN Correction to Depth")
                            
                            k_neighbors = st.slider(
                                "Number of Similar Defects (K) for Correction",
                                min_value=1,
                                max_value=5,
                                value=DEFAULT_KNN_NEIGHBORS,
                                key=f"k_neighbors_{earlier_year}_{later_year}",
                                help="Number of similar defects with positive growth to use for estimating corrected depth growth rates"
                            )
                            
                            # Correction form
                            with st.form(key=f"depth_correction_form_{earlier_year}_{later_year}"):
                                st.write("Click the button below to apply KNN correction to negative depth growth:")
                                submit_correction = st.form_submit_button("Apply Depth Correction", use_container_width=True)
                                
                                if submit_correction:
                                    with st.spinner("Correcting negative depth growth rates using KNN..."):
                                        try:
                                            corrected_results = st.session_state.comparison_results.copy()
                                            
                                            # Apply the correction
                                            corrected_df, correction_info = correct_negative_growth_rates(
                                                st.session_state.comparison_results['matches_df'], 
                                                k=k_neighbors
                                            )
                                            
                                            corrected_results['matches_df'] = corrected_df
                                            corrected_results['correction_info'] = correction_info
                                            
                                            # Update growth stats if correction was successful
                                            if correction_info.get("updated_growth_stats"):
                                                corrected_results['growth_stats'] = correction_info['updated_growth_stats']
                                            
                                            st.session_state.corrected_results = corrected_results
                                            
                                            if correction_info.get("success", False):
                                                st.success(f"Successfully corrected {correction_info['corrected_count']} out of {correction_info['total_negative']} negative depth growth defects.")
                                                
                                                if correction_info['uncorrected_count'] > 0:
                                                    st.warning(f"Could not correct {correction_info['uncorrected_count']} defects in {len(correction_info['uncorrected_joints'])} joints due to insufficient similar defects.")
                                                
                                                st.rerun()
                                            else:
                                                st.error(f"Could not apply correction: {correction_info.get('error', 'Unknown error')}")
                                        except Exception as e:
                                            st.error(f"Error during correction: {str(e)}")
                                            st.info("This could be due to missing sklearn library or incompatible data. Please check that your data has all required fields: joint number, length, width, and depth.")
            else:
                # For length and width dimensions, show analysis but no correction
                st.info(f"""
                **{selected_dimension.title()} Growth Analysis**
                
                You are viewing {selected_dimension} growth analysis. The plot above shows how {selected_dimension} measurements 
                changed between inspections. 
                
                **Note**: KNN correction is only available for depth measurements. Switch to 'depth' dimension 
                to access correction features.
                """)
                
                # Show basic stats for length/width
                matches_df = comparison_results['matches_df']
                
                if selected_dimension == 'length' and comparison_results.get('has_length_data', False):
                    if 'is_negative_length_growth' in matches_df.columns:
                        neg_count = matches_df['is_negative_length_growth'].sum()
                        total_count = len(matches_df)
                        pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                        
                        st.markdown("#### Length Growth Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(custom_metric("Total Defects", f"{total_count}"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(custom_metric("Negative Growth", f"{neg_count}"), unsafe_allow_html=True)
                        with col3:
                            st.markdown(custom_metric("Percentage", f"{pct:.1f}%"), unsafe_allow_html=True)
                
                elif selected_dimension == 'width' and comparison_results.get('has_width_data', False):
                    if 'is_negative_width_growth' in matches_df.columns:
                        neg_count = matches_df['is_negative_width_growth'].sum()
                        total_count = len(matches_df)
                        pct = (neg_count / total_count) * 100 if total_count > 0 else 0
                        
                        st.markdown("#### Width Growth Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(custom_metric("Total Defects", f"{total_count}"), unsafe_allow_html=True)
                        with col2:
                            st.markdown(custom_metric("Negative Growth", f"{neg_count}"), unsafe_allow_html=True)
                        with col3:
                            st.markdown(custom_metric("Percentage", f"{pct:.1f}%"), unsafe_allow_html=True)
    
    # Growth Rate Analysis tab
    with viz_tabs[3]:
        st.subheader("Growth Rate Analysis")
        
        # Add dimension selection to this tab
        st.markdown("#### Select Dimension for Growth Rate Analysis")
        
        # Initialize session state for this tab's dimension selection if not exists
        if st.session_state.growth_analysis_dimension not in ['depth', 'length', 'width']:
            st.session_state.growth_analysis_dimension = 'depth'
        
        growth_dimension = st.selectbox(
            "Choose dimension for growth rate analysis",
            options=['depth', 'length', 'width'],
            index=['depth', 'length', 'width'].index(st.session_state.growth_analysis_dimension),
            key=f"growth_dimension_{earlier_year}_{later_year}",
            help="Select which defect dimension to analyze for growth rate statistics"
        )
        st.session_state.growth_analysis_dimension = growth_dimension
        
        # Use the comparison_results parameter directly, check for corrected results in session state
        results_to_use = (
            st.session_state.corrected_results 
            if st.session_state.get("corrected_results") is not None 
            else st.session_state.get("comparison_results")
        )
        
        if results_to_use is None:
            st.info("No comparison data availablesss.")
        else:
            matches_df = results_to_use.get('matches_df', pd.DataFrame())
            
            if matches_df.empty:
                st.warning("No comparison data available in the results.")
            else:
                # Define dimension-specific column names and check if they exist in the dataframe
                dimension_columns = {
                    'depth': {
                        'negative_flag': 'is_negative_growth',
                        'growth_rate_cols': ['growth_rate_mm_per_year', 'growth_rate_pct_per_year']
                    },
                    'length': {
                        'negative_flag': 'is_negative_length_growth', 
                        'growth_rate_cols': ['length_growth_rate_mm_per_year']
                    },
                    'width': {
                        'negative_flag': 'is_negative_width_growth',
                        'growth_rate_cols': ['width_growth_rate_mm_per_year']
                    }
                }
                
                # Check if the selected dimension has the required columns
                dim_config = dimension_columns.get(growth_dimension)
                if not dim_config:
                    st.warning(f"Invalid dimension selected: {growth_dimension}")
                else:
                    negative_flag = dim_config['negative_flag']
                    growth_rate_cols = dim_config['growth_rate_cols']
                    
                    # Find which growth rate column exists in the dataframe
                    available_growth_col = None
                    for col in growth_rate_cols:
                        if col in matches_df.columns:
                            available_growth_col = col
                            break
                    
                    # Check if we have the minimum required columns
                    if negative_flag not in matches_df.columns or available_growth_col is None:
                        st.warning(f"""
                        **No {growth_dimension} growth data available**
                        
                        Required columns missing from comparison results:
                        - Negative flag: {'✅' if negative_flag in matches_df.columns else '❌'} `{negative_flag}`
                        - Growth rate: {'✅' if available_growth_col else '❌'} `{' or '.join(growth_rate_cols)}`
                        
                        Make sure both datasets have {growth_dimension} measurements and valid year values.
                        
                        Available columns in matches_df: {list(matches_df.columns)}
                        """)
                    else:
                        # Show correction status if applicable
                        if growth_dimension == 'depth' and 'correction_info' in results_to_use and results_to_use['correction_info'].get('success', False):
                            st.success("Showing analysis with corrected depth growth rates. The negative growth defects have been adjusted based on similar defects.")
                        
                        # Display growth rate statistics
                        st.markdown(f"#### {growth_dimension.title()} Growth Statistics")
                        
                        # Determine the unit based on the column name
                        if 'mm_per_year' in available_growth_col:
                            unit = 'mm/year'
                        elif 'pct_per_year' in available_growth_col:
                            unit = '%/year'
                        else:
                            unit = 'units/year'
                        
                        # Calculate statistics dynamically (ensures they show immediately after comparison)
                        negative_count = matches_df[negative_flag].sum()
                        total_count = len(matches_df)
                        pct_negative = (negative_count / total_count) * 100 if total_count > 0 else 0
                        
                        # Calculate positive growth statistics
                        positive_growth = matches_df[~matches_df[negative_flag]]
                        avg_growth = positive_growth[available_growth_col].mean() if len(positive_growth) > 0 else 0
                        max_growth = positive_growth[available_growth_col].max() if len(positive_growth) > 0 else 0
                        
                        # Display statistics
                        stats_cols = st.columns(3)
                        
                        with stats_cols[0]:
                            st.markdown(
                                custom_metric(
                                    f"Avg {growth_dimension.title()} Growth Rate", 
                                    f"{avg_growth:.3f} {unit}"
                                ), 
                                unsafe_allow_html=True
                            )
                        
                        with stats_cols[1]:
                            st.markdown(
                                custom_metric(
                                    f"Max {growth_dimension.title()} Growth Rate", 
                                    f"{max_growth:.3f} {unit}"
                                ), 
                                unsafe_allow_html=True
                            )
                        
                        with stats_cols[2]:
                            st.markdown(
                                custom_metric(
                                    "Negative Growth", 
                                    f"{negative_count} ({pct_negative:.1f}%)"
                                ), 
                                unsafe_allow_html=True
                            )
                        
                        # Show histogram for selected dimension
                        st.markdown(f"#### {growth_dimension.title()} Growth Rate Distribution")
                        try:
                            growth_hist_fig = create_growth_rate_histogram(results_to_use, dimension=growth_dimension)
                            st.plotly_chart(growth_hist_fig, use_container_width=True, config={'displayModeBar': False})
                        except Exception as e:
                            st.warning(f"Could not generate histogram: {str(e)}. Data is available but visualization failed.")

def multi_year_comparison_tab():
    """
    Display the multi-year comparison tab with analysis across different years.
    """
    st.markdown('<h2 class="section-header">Multi-Year Comparison</h2>', unsafe_allow_html=True)
    
    if len(st.session_state.datasets) < 2:
        st.info("""
            **Multiple datasets required**
            Please upload at least two datasets from different years to enable comparison.
            Use the sidebar to add more inspection data.
        """
        )
    else:
        # Year selection for comparison with improved UI
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        available_years = sorted(st.session_state.datasets.keys())
        
        st.markdown("<div class='section-header'>Select Years to Compare</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            earlier_year = st.selectbox(
                "Earlier Inspection Year", 
                options=available_years[:-1],  # All but the last year
                index=0,
                key="earlier_year_comparison"
            )
        
        with col2:
            # Filter for years after the selected earlier year
            later_years = [year for year in available_years if year > earlier_year]
            later_year = st.selectbox(
                "Later Inspection Year", 
                options=later_years,
                index=0,
                key="later_year_comparison"
            )
        
        # Get the datasets
        earlier_defects = st.session_state.datasets[earlier_year]['defects_df']
        later_defects = st.session_state.datasets[later_year]['defects_df']
        earlier_joints = st.session_state.datasets[earlier_year]['joints_df']
        later_joints = st.session_state.datasets[later_year]['joints_df']
        
        # Add parameter settings with better UI
        st.markdown("<div class='section-header' style='margin-top:20px;'>Comparison Parameters</div>", unsafe_allow_html=True)
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            # Distance tolerance for matching defects with tooltip
            tolerance = st.slider(
                "Distance Tolerance (m)", 
                min_value=0.001, 
                max_value=0.1, 
                value=DEFAULT_DISTANCE_TOLERANCE, 
                step=0.001,
                format="%.3f",
                key="distance_tolerance_slider"
            )
            st.markdown(
                """
                <div style="font-size:0.8em;color:#7f8c8d;margin-top:-10px;">
                Maximum distance between defects to consider them the same feature
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with param_col2:
            # Clock tolerance for matching defects
            clock_tolerance = st.slider(
                "Clock Position Tolerance (minutes)",
                min_value=0,
                max_value=60,
                value=DEFAULT_CLOCK_TOLERANCE,
                step=5,
                key="clock_tolerance_slider"
            )
            st.markdown(
                """
                <div style="font-size:0.8em;color:#7f8c8d;margin-top:-10px;">
                Maximum difference in clock position to consider defects the same feature
                </div>
                """, 
                unsafe_allow_html=True
            )

        
        # Button to perform comparison
        if st.button("Compare Defects", key="compare_defects_button", use_container_width=True):
            with st.spinner(f"Comparing defects between {earlier_year} and {later_year}..."):
                try:
                    # Store the years for later reference
                    st.session_state.comparison_years = (earlier_year, later_year)
                    
                    # Perform the comparison
                    comparison_results = compare_defects(
                        earlier_defects, 
                        later_defects,
                        old_joints_df=earlier_joints,
                        new_joints_df=later_joints,
                        old_year=int(earlier_year),
                        new_year=int(later_year),
                        distance_tolerance=tolerance,
                        clock_tolerance_minutes=clock_tolerance,
                        correct_negative_growth=False  # Don't correct yet
                    )
                    
                    # Store the comparison results in session state for other tabs
                    st.session_state.comparison_results = comparison_results
                    # Reset corrected results when new comparison is made
                    st.session_state.corrected_results = None
                    
                    # Display summary statistics
                    st.markdown("<div class='section-header' style='margin-top:20px;'>Comparison Summary</div>", unsafe_allow_html=True)
                    
                    # Create metrics in 4 columns with improved styling
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.markdown(custom_metric("Total Defects", comparison_results['total_defects']), unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown(custom_metric("Common Defects", comparison_results['common_defects_count']), unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown(custom_metric("New Defects", comparison_results['new_defects_count']), unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        st.markdown(custom_metric("% New Defects", f"{comparison_results['pct_new']:.1f}%"), unsafe_allow_html=True)
                    
                    # Call the consolidated visualization function
                    display_comparison_visualization_tabs(comparison_results, earlier_year, later_year)
                    
                    # Display tables of common and new defects in an expander
                    with st.expander("Detailed Defect Lists", expanded=False):
                        if not comparison_results['matches_df'].empty:
                            st.markdown("<div class='section-header'>Common Defects</div>", unsafe_allow_html=True)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(comparison_results['matches_df'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        if not comparison_results['new_defects'].empty:
                            st.markdown("<div class='section-header' style='margin-top:20px;'>New Defects</div>", unsafe_allow_html=True)
                            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                            st.dataframe(comparison_results['new_defects'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    info_box(
                        f"""
                        <strong>Error comparing defects:</strong> {str(e)}<br><br>
                        Make sure both datasets have the required columns and compatible data formats.
                        """, 
                        "warning"
                    )
        
        # Show stored comparison results if available
        elif st.session_state.comparison_results is not None:
            comparison_results = st.session_state.comparison_results
            # Check if the years match our current selection
            if st.session_state.comparison_years == (earlier_year, later_year):
                # Display summary statistics
                st.markdown("<div class='section-header' style='margin-top:20px;'>Comparison Summary</div>", unsafe_allow_html=True)
                
                # Create metrics in 4 columns with improved styling
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.markdown(custom_metric("Total Defects", comparison_results['total_defects']), unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown(custom_metric("Common Defects", comparison_results['common_defects_count']), unsafe_allow_html=True)
                
                with metric_cols[2]:
                    st.markdown(custom_metric("New Defects", comparison_results['new_defects_count']), unsafe_allow_html=True)
                
                with metric_cols[3]:
                    st.markdown(custom_metric("% New Defects", f"{comparison_results['pct_new']:.1f}%"), unsafe_allow_html=True)
                
                # Call the consolidated visualization function
                display_comparison_visualization_tabs(comparison_results, earlier_year, later_year)
                
                # Display tables of common and new defects in an expander
                with st.expander("Detailed Defect Lists", expanded=False):
                    if not comparison_results['matches_df'].empty:
                        st.markdown("<div class='section-header'>Common Defects</div>", unsafe_allow_html=True)
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(comparison_results['matches_df'], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if not comparison_results['new_defects'].empty:
                        st.markdown("<div class='section-header' style='margin-top:20px;'>New Defects</div>", unsafe_allow_html=True)
                        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                        st.dataframe(comparison_results['new_defects'], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Years don't match, ask user to re-run comparison
                st.info("You've changed the years for comparison. Please click 'Compare Defects' to analyze the new year combination.")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close the card container

def run_app():
    """Main function to run the Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Page configuration
    st.set_page_config(
        page_title="Pipeline Inspection Analysis", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    load_css()
    
    # Logo and title
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="{get_logo_base64()}" class="logo" alt="Logo">
        </div>
        <h1 class="custom-title">{APP_TITLE}</h1>
        <p class="custom-subtitle">{APP_SUBTITLE}</p>
        """, 
        unsafe_allow_html=True
    )
    
    # Create sidebar and get uploaded file
    uploaded_file, selected_year = create_sidebar(st.session_state)
    
    # Main content
    if uploaded_file is not None:
        # Create progress indicator for the workflow
        show_step_indicator(st.session_state.active_step)

        # Create a container for the column mapping process
        with st.container():
            # Load the data with robust encoding handling
            try:
                df, successful_encoding = load_csv_with_encoding(uploaded_file)
                if successful_encoding != 'utf-8':
                    info_box(f"File loaded with {successful_encoding} encoding. Some special characters may display differently.", "info")
            except ValueError as e:
                info_box(str(e), "warning")
                st.stop()
            
            # Display file info in a card-like container
            with st.expander("File Preview", expanded=True):
                st.markdown('<div class="section-header">File Information</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Filename:** {uploaded_file.name}")
                with col2:
                    st.markdown(f"**Rows:** {df.shape[0]}")
                with col3:
                    st.markdown(f"**Columns:** {df.shape[1]}")
                
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(df.head(100), height=200, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Process data
            processed = process_data_section(df, selected_year)
            
            # If processed, rerun to update state and show analysis tabs
            if processed:
                st.rerun()
    
    # Create tabs for different analysis modes if datasets exist
    if st.session_state.datasets:
        # Reset progress indicator since we're now in analysis mode
        st.session_state.active_step = 3
        
        tab1, tab2 = st.tabs(["Single Year Analysis", "Multi-Year Comparison"])
        
        # Tab 1: Single Year Analysis
        with tab1:
            single_year_analysis_tab()
        
        # Tab 2: Multi-Year Comparison
        with tab2:
            multi_year_comparison_tab()
    else:
        # If no datasets are loaded, show a welcome message
        create_welcome_screen()

if __name__ == "__main__":
    run_app()