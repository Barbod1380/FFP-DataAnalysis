# enhanced_main.py - With improved styling and organization
import streamlit as st
import pandas as pd
import re
import time
import numpy as np
import base64
from datetime import datetime

# Import functions from modules
from data_processing import process_pipeline_data
from utils import *
from visualizations import *
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
    create_defect_location_plot,
    create_growth_rate_histogram,
    create_negative_growth_plot
)
from defect_analysis import *

# ===== Custom CSS to enhance the app's appearance =====
def load_css():
    """Apply custom CSS styling to the app."""
    css = """
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom title and header styling */
    .custom-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        text-align: center;
        border-bottom: 2px solid #3498db;
    }
    
    .custom-subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Card-like container styling */
    .card-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2980b9;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 15px;
    }
    
    .warning-box {
        background-color: #fff5e6;
        border-left: 4px solid #e67e22;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 15px;
    }
    
    .success-box {
        background-color: #e8f8ef;
        border-left: 4px solid #2ecc71;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        margin-bottom: 15px;
    }
    
    /* Data table styling */
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    
    .styled-table thead tr {
        background-color: #2980b9;
        color: #ffffff;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }

    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }

    /* Sidebar specific styling */
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    
    /* Button styling */
    .custom-button {
        background-color: #3498db;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s;
        border: none;
        margin: 5px 0;
    }
    
    .custom-button:hover {
        background-color: #2980b9;
    }
    
    /* Custom metric styling */
    .custom-metric {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #2980b9;
    }
    
    .metric-label {
        font-size: 14px;
        color: #7f8c8d;
    }
    
    /* Tab styling - override Streamlit's default */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f1f1;
        border-radius: 4px 4px 0 0;
        padding-left: 20px;
        padding-right: 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2980b9;
        color: white;
    }
    
    /* Progress bar step indicator */
    .step-progress {
        display: flex;
        justify-content: space-between;
        margin-bottom: 30px;
        position: relative;
    }
    
    .step-progress:before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 2px;
        background: #e0e0e0;
        transform: translateY(-50%);
        z-index: 1;
    }
    
    .step {
        width: 30px;
        height: 30px;
        background-color: white;
        border: 2px solid #e0e0e0;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        z-index: 2;
        position: relative;
    }
    
    .step.active {
        background-color: #3498db;
        border-color: #3498db;
        color: white;
    }
    
    .step.completed {
        background-color: #2ecc71;
        border-color: #2ecc71;
        color: white;
    }
    
    .step-label {
        position: absolute;
        top: 35px;
        font-size: 12px;
        width: 100px;
        text-align: center;
        left: -35px;
    }
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .logo {
        width: 80px;
        height: 80px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 12px;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #e0e0e0;
    }

    /* Custom badges */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }

    .status-badge.green {
        background-color: #e8f8ef;
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }

    .status-badge.yellow {
        background-color: #fff5e6;
        color: #e67e22;
        border: 1px solid #e67e22;
    }

    .status-badge.red {
        background-color: #fae5e5;
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }

    .tooltip .tooltip-text {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    /* Make dataframes prettier */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Function to create a custom card
def card(title, content):
    card_html = f"""
    <div class="card-container">
        <div class="section-header">{title}</div>
        {content}
    </div>
    """
    return st.markdown(card_html, unsafe_allow_html=True)

# Function to create custom metrics with better styling
def custom_metric(label, value, description=None):
    metric_html = f"""
    <div class="custom-metric">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {f"<div style='font-size:12px;color:#95a5a6;'>{description}</div>" if description else ""}
    </div>
    """
    return metric_html

# Function to create a status badge
def status_badge(text, status):
    badge_html = f"""
    <span class="status-badge {status}">{text}</span>
    """
    return badge_html

# Function to create custom info/warning boxes
def info_box(text, box_type="info"):
    box_class = f"{box_type}-box"
    box_html = f"""
    <div class="{box_class}">
        {text}
    </div>
    """
    return st.markdown(box_html, unsafe_allow_html=True)  

# Function to create step progress indicator
def show_step_indicator(active_step):
    """Display a simple step indicator that won't show raw HTML"""
    steps = ["Upload File", "Map Columns", "Process Data"]
    cols = st.columns(len(steps))
    
    for i, (col, step_label) in enumerate(zip(cols, steps), 1):
        with col:
            if i < active_step:
                emoji = "✅"  # Completed
                color = "green"
            elif i == active_step:
                emoji = "🔵"  # Active
                color = "blue"
            else:
                emoji = "⚪"  # Not started
                color = "gray"
            
            st.markdown(f"### {emoji} **Step {i}**", unsafe_allow_html=True)
            st.caption(step_label)

# Logo - create a pipeline logo (placeholder)
def get_logo_base64():
    # This is just a placeholder. In a real app, you would use your actual logo.
    return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgIDxjaXJjbGUgZmlsbD0iIzM0OThkYiIgY3g9IjEwMCIgY3k9IjEwMCIgcj0iMTAwIi8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iNzAiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSI1Ii8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iMTEwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjIwIiByeD0iNSIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iNTUiIGN5PSI4MCIgcj0iOCIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iMTQwIiBjeT0iMTIwIiByPSI4Ii8+CiAgPC9nPgo8L3N2Zz4K"

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
    raise ValueError(f"Failed to load the file with any of the encodings: {', '.join(encodings)}")

# Initialize session state for multiple datasets
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}  # Will store {year: {'joints_df': df1, 'defects_df': df2}}
if 'current_year' not in st.session_state:
    st.session_state.current_year = None
if 'file_upload_key' not in st.session_state:
    st.session_state.file_upload_key = 0  # For forcing file uploader to clear
if 'active_step' not in st.session_state:
    st.session_state.active_step = 1

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
    <h1 class="custom-title">Pipeline Inspection Analysis</h1>
    <p class="custom-subtitle">Upload inspection data to analyze pipeline condition and track defects over time</p>
    """, 
    unsafe_allow_html=True
)

# Create a sidebar for file management
with st.sidebar:
    st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
    
    # Display currently loaded datasets
    if st.session_state.datasets:
        st.markdown("<strong>Loaded Datasets</strong>", unsafe_allow_html=True)
        for year in sorted(st.session_state.datasets.keys()):
            st.markdown(
                f'<div style="padding:8px 10px;margin-bottom:5px;background-color:#e8f8ef;border-radius:5px;">'
                f'<span style="color:#2ecc71;margin-right:8px;">✓</span>{year} data loaded'
                f'</div>', 
                unsafe_allow_html=True
            )
    
    st.markdown('<div style="margin-top:20px;"><strong>Add New Dataset</strong></div>', unsafe_allow_html=True)
    
    # Year selection for new data
    current_year = datetime.now().year
    year_options = list(range(current_year - 30, current_year + 1))
    selected_year = st.selectbox(
        "Select Inspection Year", 
        options=year_options,
        index=len(year_options) - 1,  # Default to current year
        key="year_selector_sidebar"
    )
    
    # File uploader with improved styling
    st.markdown(f'<div style="margin:10px 0 5px 0;"><strong>Upload {selected_year} Inspection CSV</strong></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your CSV file here",
        type="csv",
        key=f"file_uploader_{st.session_state.file_upload_key}",
        label_visibility="collapsed"
    )
    
    # Button to clear all data with better styling
    st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
    if st.button("Clear All Datasets", key="clear_all_datasets_btn", use_container_width=True):
        st.session_state.datasets = {}
        st.session_state.current_year = None
        st.session_state.file_upload_key += 1  # Force file uploader to reset
        st.session_state.active_step = 1
        st.rerun()
    
    # Add a footer with app info
    st.markdown(
        """
        <div class="footer">
            <p>Pipeline Inspection Analysis Tool<br>Version 1.0</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Main content area
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
                value=1.0,  # Default value
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

                if 'surface location' in standardized_df.columns:
                    standardized_df['surface location'] = standardized_df['surface location'].apply(standardize_surface_location)
                
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

# Create tabs for different analysis modes
if st.session_state.datasets:
    # Reset progress indicator since we're now in analysis mode
    st.session_state.active_step = 3
    
    tab1, tab2 = st.tabs(["Single Year Analysis", "Multi-Year Comparison"])
    
    # Tab 1: Single Year Analysis
    with tab1:
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
                csv = joints_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="joints_{selected_analysis_year}.csv" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.8em;padding:5px 10px;">Download Joints CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                st.subheader(f"{selected_analysis_year} Defects")
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(defects_df.head(5), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add download buttons 
                csv = defects_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="defects_{selected_analysis_year}.csv" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.8em;padding:5px 10px;">Download Defects CSV</a>'
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


            else:
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
    
    # Tab 2: Multi-Year Comparison
    with tab2:
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
            
            # Add parameter settings with better UI
            st.markdown("<div class='section-header' style='margin-top:20px;'>Comparison Parameters</div>", unsafe_allow_html=True)
            
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                # Distance tolerance for matching defects with tooltip
                tolerance = st.slider(
                    "Distance Tolerance (m)", 
                    min_value=0.001, 
                    max_value=0.1, 
                    value=0.01, 
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
            
            # Button to perform comparison
            if st.button("Compare Defects", key="compare_defects_button", use_container_width=True):
                with st.spinner(f"Comparing defects between {earlier_year} and {later_year}..."):
                    try:
                        # Perform the comparison
                        comparison_results = compare_defects(
                            earlier_defects, 
                            later_defects,
                            old_year=int(earlier_year),
                            new_year=int(later_year),
                            distance_tolerance=tolerance
                        )
                        
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
                        
                        # Create visualization tabs
                        viz_tabs = st.tabs([
                            "New vs Common", "New Defect Types", "Growth Rate", "Negative Growth"
                        ])
                        
                        with viz_tabs[0]:
                            # Pie chart of common vs new defects
                            pie_fig = create_comparison_stats_plot(comparison_results)
                            st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})
                        
                        with viz_tabs[1]:
                            # Bar chart of new defect types
                            bar_fig = create_new_defect_types_plot(comparison_results)
                            st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False})
                        
                        with viz_tabs[2]:
                            # Histogram of defect growth rates
                            if comparison_results.get('has_depth_data', False) and comparison_results.get('calculate_growth', False):
                                # Display growth rate statistics
                                growth_stats = comparison_results['growth_stats']
                                
                                stats_cols = st.columns(3)
                                
                                with stats_cols[0]:
                                    if comparison_results.get('has_wt_data', False):
                                        st.markdown(
                                            custom_metric(
                                                "Avg Growth Rate", 
                                                f"{growth_stats['avg_positive_growth_rate_mm']:.3f} mm/yr"
                                            ), 
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            custom_metric(
                                                "Avg Growth Rate", 
                                                f"{growth_stats['avg_positive_growth_rate_pct']:.3f} %/yr"
                                            ), 
                                            unsafe_allow_html=True
                                        )
                                
                                with stats_cols[1]:
                                    if comparison_results.get('has_wt_data', False):
                                        st.markdown(
                                            custom_metric(
                                                "Max Growth Rate", 
                                                f"{growth_stats['max_growth_rate_mm']:.3f} mm/yr"
                                            ), 
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            custom_metric(
                                                "Max Growth Rate", 
                                                f"{growth_stats['max_growth_rate_pct']:.3f} %/yr"
                                            ), 
                                            unsafe_allow_html=True
                                        )
                                
                                with stats_cols[2]:
                                    st.markdown(
                                        custom_metric(
                                            "Negative Growth", 
                                            f"{growth_stats['negative_growth_count']} ({growth_stats['pct_negative_growth']:.1f}%)"
                                        ), 
                                        unsafe_allow_html=True
                                    )
                                
                                # Show histogram
                                growth_hist_fig = create_growth_rate_histogram(comparison_results)
                                st.plotly_chart(growth_hist_fig, use_container_width=True, config={'displayModeBar': False})
                            else:
                                info_box("Growth rate analysis not available. Requires depth data in both datasets and valid year values.", "info")
                        
                        with viz_tabs[3]:
                            # Plot highlighting negative growth defects
                            if comparison_results.get('has_depth_data', False) and comparison_results.get('calculate_growth', False):
                                negative_growth_fig = create_negative_growth_plot(comparison_results)
                                st.plotly_chart(negative_growth_fig, use_container_width=True, config={'displayModeBar': False})
                                
                                # Add explanation
                                st.info(
                                    """
                                    **Negative Growth Explanation:**
                                    Defects showing negative growth rates (red triangles) indicate areas where the defect 
                                    depth appears to have decreased between inspections. This is physically unlikely and 
                                    usually indicates:
                                        - Measurement errors in one or both inspections
                                        - Different inspection tools or calibration between surveys
                                        - Possible repair work that wasn't documented
                                    These areas should be flagged for verification and further investigation.
                                    """
                                )
                            else:
                                info_box("Negative growth analysis not available. Requires depth data in both datasets and valid year values.", "info")
                        
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
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close the card container
else:
    # If no datasets are loaded, show a welcome message
    st.markdown(
        """
        <div class="card-container" style="text-align:center;padding:40px;background-color:#f8f9fa;">
            <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgIDxjaXJjbGUgZmlsbD0iI2NjY2NjYyIgY3g9IjEwMCIgY3k9IjEwMCIgcj0iMTAwIi8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iNzAiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSI1Ii8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iMTEwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjIwIiByeD0iNSIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iNTUiIGN5PSI4MCIgcj0iOCIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iMTQwIiBjeT0iMTIwIiByPSI4Ii8+CiAgPC9nPgo8L3N2Zz4K" style="width:120px;margin-bottom:20px;">
            <h2 style="color:#7f8c8d;margin-bottom:20px;">Welcome to Pipeline Inspection Analysis</h2>
            <p style="color:#95a5a6;margin-bottom:30px;">Upload at least one dataset using the sidebar to begin analysis.</p>
            <div style="color:#3498db;font-size:2em;"><i class="fas fa-arrow-left"></i> Start by uploading a CSV file</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add a quick guide
    st.markdown(
        """
        <div class="card-container" style="margin-top:20px;">
            <div class="section-header">Quick Guide</div>
            <ol style="padding-left:20px;">
                <li><strong>Upload Data:</strong> Use the sidebar to upload pipeline inspection CSV files</li>
                <li><strong>Map Columns:</strong> Match your file's columns to standard names</li>
                <li><strong>Analyze:</strong> View statistics and visualizations for your pipeline data</li>
                <li><strong>Compare:</strong> Upload multiple years to track defect growth over time</li>
            </ol>
            <div class="section-header" style="margin-top:20px;">Supported Features</div>
            <ul style="padding-left:20px;">
                <li>Statistical analysis of defect dimensions</li>
                <li>Unwrapped pipeline visualizations</li>
                <li>Joint-by-joint defect analysis</li>
                <li>Multi-year comparison with growth rate calculation</li>
                <li>New defect identification</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )