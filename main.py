# Enhanced main.py with robust encoding handling
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import functions from modules
from data_processing import process_pipeline_data
from utils import parse_clock
from visualizations import create_unwrapped_pipeline_visualization, create_joint_defect_visualization, create_growth_rate_histogram, create_negative_growth_plot
from column_mapping import (
    suggest_column_mapping, 
    apply_column_mapping, 
    get_missing_required_columns, 
    STANDARD_COLUMNS,
    REQUIRED_COLUMNS
)
from multi_year_analysis import (
    compare_defects, 
    create_comparison_stats_plot, 
    create_new_defect_types_plot,
)

# Function to load CSV with multiple encoding attempts
def load_csv_with_encoding(file):
    """
    Try to load a CSV file with different encodings.
    Returns the DataFrame and the successful encoding.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
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
            
            return df, encoding
            
        except Exception as e:
            continue  # Try next encoding
    
    # If all encodings fail
    raise ValueError(f"Failed to load the file with any of the encodings: {', '.join(encodings)}")

# Initialize session state for multiple datasets
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}  # Will store {year: {'joints_df': df1, 'defects_df': df2}}
if 'current_year' not in st.session_state:
    st.session_state.current_year = None
if 'file_upload_key' not in st.session_state:
    st.session_state.file_upload_key = 0  # For forcing file uploader to clear

# Page configuration
st.set_page_config(page_title="Pipeline Inspection Analysis", layout="wide")

# Application title and description
st.title("Pipeline Inspection Multi-Year Analysis")
st.write("Upload inspection data from multiple years to analyze pipeline condition changes over time.")

# Create a sidebar for file management
with st.sidebar:
    st.header("Data Management")
    
    # Display currently loaded datasets
    if st.session_state.datasets:
        st.subheader("Loaded Datasets")
        for year in sorted(st.session_state.datasets.keys()):
            st.success(f"✓ {year} data loaded")
    
    # Add new dataset section
    st.subheader("Add New Dataset")
    
    # Year selection for new data
    current_year = datetime.now().year
    year_options = list(range(current_year - 30, current_year + 1))
    selected_year = st.selectbox(
        "Select Inspection Year", 
        options=year_options,
        index=len(year_options) - 1,  # Default to current year
        key="year_selector"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        f"Upload {selected_year} Inspection CSV", 
        type="csv",
        key=f"file_uploader_{st.session_state.file_upload_key}"
    )
    
    # Button to clear all data
    if st.button("Clear All Datasets"):
        st.session_state.datasets = {}
        st.session_state.current_year = None
        st.session_state.file_upload_key += 1  # Force file uploader to reset
        st.rerun()

# Main content area
if uploaded_file is not None:
    # Create a container for the column mapping process
    with st.container():
        # Load the data with robust encoding handling
        try:
            df, successful_encoding = load_csv_with_encoding(uploaded_file)
            if successful_encoding != 'utf-8':
                st.info(f"File loaded with {successful_encoding} encoding. Some special characters may display differently.")
        except ValueError as e:
            st.error(str(e))
            st.stop()
        
        # Display file info in a collapsible section
        with st.expander("File Preview", expanded=True):
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
            st.dataframe(df.head(3))
        
        # Column mapping process in a collapsible section
        with st.expander("Column Mapping", expanded=True):
            st.subheader(f"Map Columns for {selected_year} Data")
            
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
                        key=f"map_{selected_year}_{std_col}"
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
                        key=f"map_{selected_year}_{std_col}"
                    )
                    confirmed_mapping[std_col] = selected
            
            # Add note about required fields
            st.write("* Required fields")
            
            # Check for missing required columns
            missing_cols = get_missing_required_columns(confirmed_mapping)
            if missing_cols:
                st.warning(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("You may proceed, but functionality may be limited.")
        
        # Process and add button
        process_col1, process_col2 = st.columns([1, 3])
        with process_col1:
            process_button = st.button(f"Process {selected_year} Data")
        
        if process_button:
            with st.spinner(f"Processing {selected_year} data..."):
                # Apply the mapping to rename columns
                standardized_df = apply_column_mapping(df, confirmed_mapping)
                
                # Process the pipeline data
                joints_df, defects_df = process_pipeline_data(standardized_df)
                
                # Process clock and area data
                if 'clock' in defects_df.columns:
                    defects_df["clock_float"] = defects_df["clock"].apply(parse_clock)
                
                if 'length [mm]' in defects_df.columns and 'width [mm]' in defects_df.columns:
                    defects_df["area_mm2"] = defects_df["length [mm]"] * defects_df["width [mm]"]
                
                if 'joint number' in defects_df.columns:
                    defects_df["joint number"] = defects_df["joint number"].astype("Int64")
                
                # Store in session state
                st.session_state.datasets[selected_year] = {
                    'joints_df': joints_df,
                    'defects_df': defects_df
                }
                st.session_state.current_year = selected_year
                
                # Force the file uploader to reset
                st.session_state.file_upload_key += 1
                
                st.success(f"Successfully processed {selected_year} data")
                st.rerun()

# Create tabs for different analysis modes
if st.session_state.datasets:
    tab1, tab2 = st.tabs(["Single Year Analysis", "Multi-Year Comparison"])
    
    # Tab 1: Single Year Analysis (similar to original functionality)
    with tab1:
        st.header("Single Year Analysis")
        
        # Select year to analyze
        years = sorted(st.session_state.datasets.keys())
        selected_analysis_year = st.selectbox(
            "Select Year to Analyze",
            options=years,
            index=years.index(st.session_state.current_year) if st.session_state.current_year in years else 0
        )
        
        # Get the selected dataset
        joints_df = st.session_state.datasets[selected_analysis_year]['joints_df']
        defects_df = st.session_state.datasets[selected_analysis_year]['defects_df']
        
        # Display data preview
        with st.expander("Data Preview", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_analysis_year} Joints (Top 5 Records)")
                st.dataframe(joints_df.head(5))
            
            with col2:
                st.subheader(f"{selected_analysis_year} Defects (Top 5 Records)")
                st.dataframe(defects_df.head(5))
        
        # Visualization section
        st.subheader("Visualization")
        
        # Visualization type selection
        viz_type = st.radio(
            "Select Visualization Type",
            ["Complete Pipeline", "Joint-by-Joint"],
            horizontal=True
        )
        
        if viz_type == "Complete Pipeline":
            # Button to show visualization
            if st.button("Show Complete Pipeline Visualization"):
                st.subheader(f"Pipeline Defect Map ({selected_analysis_year})")
                fig = create_unwrapped_pipeline_visualization(defects_df, joints_df)
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Joint selection
            available_joints = sorted(joints_df["joint number"].unique())
            
            # Format joint numbers with distance
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
                st.subheader(f"Defect Map for {selected_joint_label} ({selected_analysis_year})")
                fig = create_joint_defect_visualization(defects_df, selected_joint)
                st.plotly_chart(fig, use_container_width=True)
    
    # Modified section for the Multi-Year Comparison tab
    with tab2:
        st.header("Multi-Year Comparison")
        
        if len(st.session_state.datasets) < 2:
            st.warning("Please upload at least two datasets from different years to enable comparison.")
        else:
            # Year selection for comparison
            available_years = sorted(st.session_state.datasets.keys())
            
            col1, col2 = st.columns(2)
            
            with col1:
                earlier_year = st.selectbox(
                    "Select Earlier Year", 
                    options=available_years[:-1],  # All but the last year
                    index=0,
                    key="earlier_year"
                )
            
            with col2:
                # Filter for years after the selected earlier year
                later_years = [year for year in available_years if year > earlier_year]
                later_year = st.selectbox(
                    "Select Later Year", 
                    options=later_years,
                    index=0,
                    key="later_year"
                )
            
            # Get the datasets
            earlier_defects = st.session_state.datasets[earlier_year]['defects_df']
            later_defects = st.session_state.datasets[later_year]['defects_df']
            
            # Distance tolerance for matching defects
            tolerance = st.slider(
                "Distance Tolerance (m)", 
                min_value=0.001, 
                max_value=0.1, 
                value=0.001, 
                step=0.001,
                format="%.3f",
                help="Maximum distance between defects to consider them at the same location"
            )
            
            # Button to perform comparison
            if st.button("Compare Defects"):
                with st.spinner(f"Comparing defects between {earlier_year} and {later_year}..."):
                    try:
                        # Perform the comparison with year values for growth rate calculation
                        comparison_results = compare_defects(
                            earlier_defects, 
                            later_defects,
                            old_year=int(earlier_year),
                            new_year=int(later_year),
                            distance_tolerance=tolerance
                        )
                        # Display summary statistics
                        st.subheader("Comparison Summary")
                        
                        # Create metrics in 4 columns
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric(label="Total Defects", value=comparison_results['total_defects'])
                        
                        with metric_col2:
                            st.metric(label="Common Defects", value=comparison_results['common_defects_count'])
                        
                        with metric_col3:
                            st.metric(label="New Defects", value=comparison_results['new_defects_count'])
                        
                        with metric_col4:
                            st.metric(label="% New Defects", value=f"{comparison_results['pct_new']:.1f}%")
                        
                        # Create visualization tabs
                        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                            "New vs Common", "New Defect Types", "Growth Rate", "Negative Growth"
                        ])
                        
                        with viz_tab1:
                            # Pie chart of common vs new defects
                            pie_fig = create_comparison_stats_plot(comparison_results)
                            st.plotly_chart(pie_fig, use_container_width=True)
                        
                        with viz_tab2:
                            # Bar chart of new defect types
                            bar_fig = create_new_defect_types_plot(comparison_results)
                            st.plotly_chart(bar_fig, use_container_width=True)
                        
                        with viz_tab3:
                            # Histogram of defect growth rates
                            if comparison_results['has_depth_data']:
                                # Display growth rate statistics
                                growth_stats = comparison_results['growth_stats']
                                
                                stats_col1, stats_col2, stats_col3 = st.columns(3)
                                
                                with stats_col1:
                                    if comparison_results['has_wt_data']:
                                        st.metric(
                                            label="Avg Growth Rate", 
                                            value=f"{growth_stats['avg_positive_growth_rate_mm']:.3f} mm/yr"
                                        )
                                    else:
                                        st.metric(
                                            label="Avg Growth Rate", 
                                            value=f"{growth_stats['avg_positive_growth_rate_pct']:.3f} %/yr"
                                        )
                                
                                with stats_col2:
                                    if comparison_results['has_wt_data']:
                                        st.metric(
                                            label="Max Growth Rate", 
                                            value=f"{growth_stats['max_growth_rate_mm']:.3f} mm/yr"
                                        )
                                    else:
                                        st.metric(
                                            label="Max Growth Rate", 
                                            value=f"{growth_stats['max_growth_rate_pct']:.3f} %/yr"
                                        )
                                
                                with stats_col3:
                                    st.metric(
                                        label="Negative Growth Defects", 
                                        value=f"{growth_stats['negative_growth_count']} ({growth_stats['pct_negative_growth']:.1f}%)"
                                    )
                                
                                # Show histogram
                                growth_hist_fig = create_growth_rate_histogram(comparison_results)
                                st.plotly_chart(growth_hist_fig, use_container_width=True)
                            else:
                                st.info("Depth data not available in one or both datasets. Cannot calculate growth rates.")
                        
                        with viz_tab4:
                            # Plot highlighting negative growth defects
                            if comparison_results['has_depth_data']:
                                negative_growth_fig = create_negative_growth_plot(comparison_results)
                                st.plotly_chart(negative_growth_fig, use_container_width=True)
                                
                                # Add explanation
                                st.info("""
                                **Negative Growth Explanation**:
                                Defects showing negative growth rates (red triangles) indicate areas where the defect 
                                depth appears to have decreased between inspections. This is physically unlikely and 
                                usually indicates:
                                
                                - Measurement errors in one or both inspections
                                - Different inspection tools or calibration between surveys
                                - Possible repair work that wasn't documented
                                
                                These areas should be flagged for verification and further investigation.
                                """)
                            else:
                                st.info("Depth data not available in one or both datasets. Cannot analyze negative growth.")
                        
                        # Display tables of common and new defects in an expander
                        with st.expander("Detailed Defect Lists", expanded=False):
                            if not comparison_results['matches_df'].empty:
                                st.subheader("Common Defects")
                                st.dataframe(comparison_results['matches_df'])
                            
                            if not comparison_results['new_defects'].empty:
                                st.subheader("New Defects")
                                st.dataframe(comparison_results['new_defects'])
                    
                    except Exception as e:
                        st.error(f"Error comparing defects: {str(e)}")
                        st.info("Make sure both datasets have the required columns and compatible data formats.")
        
else:
    st.info("Please upload at least one dataset using the sidebar to begin analysis.")