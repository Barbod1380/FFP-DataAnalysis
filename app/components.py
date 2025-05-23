"""
UI components for the Pipeline Analysis application.
"""
import streamlit as st
import pandas as pd
import base64
from datetime import datetime

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

def card(title, content):
    """
    Create a custom card with a title and content.
    
    Parameters:
    - title: Card title
    - content: HTML content for the card
    
    Returns:
    - Streamlit markdown element
    """
    card_html = f"""
    <div class="card-container">
        <div class="section-header">{title}</div>
        {content}
    </div>
    """
    return st.markdown(card_html, unsafe_allow_html=True)

def custom_metric(label, value, description=None):
    """
    Create a custom metric display with a value and label.
    
    Parameters:
    - label: Metric name
    - value: Metric value
    - description: Optional description text
    
    Returns:
    - HTML string for the metric
    """
    metric_html = f"""
    <div class="custom-metric">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {f"<div style='font-size:12px;color:#95a5a6;'>{description}</div>" if description else ""}
    </div>
    """
    return metric_html

def status_badge(text, status):
    """
    Create a colored status badge.
    
    Parameters:
    - text: Badge text
    - status: Badge color/status (green, yellow, red)
    
    Returns:
    - HTML string for the badge
    """
    badge_html = f"""
    <span class="status-badge {status}">{text}</span>
    """
    return badge_html

def info_box(text, box_type="info"):
    """
    Create an info, warning, or success box.
    
    Parameters:
    - text: Box content
    - box_type: Box style (info, warning, success)
    
    Returns:
    - Streamlit markdown element
    """
    box_class = f"{box_type}-box"
    box_html = f"""
    <div class="{box_class}">
        {text}
    </div>
    """
    return st.markdown(box_html, unsafe_allow_html=True)

def show_step_indicator(active_step):
    """
    Display a step progress indicator.
    
    Parameters:
    - active_step: Current active step (1-based index)
    """
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

def get_logo_base64():
    """
    Get a base64-encoded SVG logo for the application.
    
    Returns:
    - Base64-encoded SVG string
    """
    # Pipeline logo placeholder
    return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgIDxjaXJjbGUgZmlsbD0iIzM0OThkYiIgY3g9IjEwMCIgY3k9IjEwMCIgcj0iMTAwIi8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iNzAiIHdpZHRoPSIxMjAiIGhlaWdodD0iMjAiIHJ4PSI1Ii8+CiAgICA8cmVjdCBmaWxsPSIjZmZmIiB4PSI0MCIgeT0iMTEwIiB3aWR0aD0iMTIwIiBoZWlnaHQ9IjIwIiByeD0iNSIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iNTUiIGN5PSI4MCIgcj0iOCIvPgogICAgPGNpcmNsZSBmaWxsPSIjZTc0YzNjIiBjeD0iMTQwIiBjeT0iMTIwIiByPSI4Ii8+CiAgPC9nPgo8L3N2Zz4K"

def create_welcome_screen():
    """
    Create a welcome screen for when no data is loaded.
    
    Returns:
    - Streamlit markdown element
    """
    welcome_html = f"""
    <div class="card-container" style="text-align:center;padding:40px;background-color:#f8f9fa;">
        <img src="{get_logo_base64()}" style="width:120px;margin-bottom:20px;">
        <h2 style="color:#7f8c8d;margin-bottom:20px;">Welcome to Pipeline Inspection Analysis</h2>
        <p style="color:#95a5a6;margin-bottom:30px;">Upload at least one dataset using the sidebar to begin analysis.</p>
        <div style="color:#3498db;font-size:2em;"><i class="fas fa-arrow-left"></i> Start by uploading a CSV file</div>
    </div>
    """
    st.markdown(welcome_html, unsafe_allow_html=True)
    
    # Add a quick guide
    guide_html = """
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
    """
    st.markdown(guide_html, unsafe_allow_html=True)

def create_sidebar(session_state):
    """
    Create the application sidebar for data management.
    
    Parameters:
    - session_state: Streamlit session state
    """
    with st.sidebar:
        st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)
        
        # Display currently loaded datasets
        if session_state.datasets:
            st.markdown("<strong>Loaded Datasets</strong>", unsafe_allow_html=True)
            for year in sorted(session_state.datasets.keys()):
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
            key=f"file_uploader_{session_state.file_upload_key}",
            label_visibility="collapsed"
        )
        
        # Button to clear all data with better styling
        st.markdown('<div style="margin-top:30px;"></div>', unsafe_allow_html=True)
        if st.button("Clear All Datasets", key="clear_all_datasets_btn", use_container_width=True):
            session_state.datasets = {}
            session_state.current_year = None
            session_state.file_upload_key += 1  # Force file uploader to reset
            session_state.active_step = 1
            session_state.comparison_results = None
            session_state.corrected_results = None
            session_state.comparison_years = None
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
        
        return uploaded_file, selected_year

def create_data_download_links(df, prefix, year):
    """
    Create download links for dataframes.
    
    Parameters:
    - df: DataFrame to download
    - prefix: Prefix for the filename
    - year: Year to include in the filename
    
    Returns:
    - HTML string with the download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{prefix}_{year}.csv" class="custom-button" style="display:inline-block;text-decoration:none;margin-top:10px;font-size:0.8em;padding:5px 10px;">Download {prefix} CSV</a>'
    return href