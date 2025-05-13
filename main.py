# Modified section of main.py
import streamlit as st
import pandas as pd
import numpy as np

# Import functions from modules
from data_processing import process_pipeline_data
from utils import parse_clock
from visualizations import create_unwrapped_pipeline_visualization, create_joint_defect_visualization
from column_mapping import (
    suggest_column_mapping, 
    apply_column_mapping, 
    get_missing_required_columns, 
    STANDARD_COLUMNS,
    REQUIRED_COLUMNS
)

# Initialize session state variables if they don't exist
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = False
if 'joints_df' not in st.session_state:
    st.session_state.joints_df = None
if 'defects_df' not in st.session_state:
    st.session_state.defects_df = None

# Set page title
st.title("Pipeline Inspection Data Visualization")

# File uploader section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Reset processed state if a new file is uploaded
if uploaded_file is not None and not st.session_state.processed_data:
    # Load the data
    df = pd.read_csv(uploaded_file)
    
    # Display a preview of the uploaded file
    st.subheader("Uploaded File Preview")
    st.dataframe(df.head(3))
    
    # Get suggested column mapping
    suggested_mapping = suggest_column_mapping(df)
    
    # Create UI for mapping confirmation
    st.subheader("Confirm Column Mapping")
    st.write("Please confirm the mapping between your file's columns and our standard columns:")
    
    confirmed_mapping = {}
    all_columns = [None] + df.columns.tolist()
    
    # Create two columns for the mapping UI to save space
    col1, col2 = st.columns(2)
    
    # Split the standard columns into two groups for the two columns
    half = len(STANDARD_COLUMNS) // 2
    
    # First column of mappings
    with col1:
        for std_col in STANDARD_COLUMNS[:half]:
            suggested = suggested_mapping.get(std_col)
            # Find the index of the suggested column in the dropdown list
            index = 0 if suggested is None else all_columns.index(suggested)
            
            # Highlight required columns
            is_required = std_col in REQUIRED_COLUMNS
            label = f"Map '{std_col}' to:" + (" *" if is_required else "")
            
            selected = st.selectbox(
                label,
                options=all_columns,
                index=index,
                key=f"map_{std_col}"
            )
            confirmed_mapping[std_col] = selected
    
    # Second column of mappings
    with col2:
        for std_col in STANDARD_COLUMNS[half:]:
            suggested = suggested_mapping.get(std_col)
            # Find the index of the suggested column in the dropdown list
            index = 0 if suggested is None else all_columns.index(suggested)
            
            # Highlight required columns
            is_required = std_col in REQUIRED_COLUMNS
            label = f"Map '{std_col}' to:" + (" *" if is_required else "")
            
            selected = st.selectbox(
                label,
                options=all_columns,
                index=index,
                key=f"map_{std_col}"
            )
            confirmed_mapping[std_col] = selected
    
    # Add note about required fields
    st.write("* Required fields")
    
    # Check for missing required columns
    missing_cols = get_missing_required_columns(confirmed_mapping)
    if missing_cols:
        st.warning(f"The following required columns are missing: {', '.join(missing_cols)}")
        st.info("You may still proceed, but some functionality may be limited.")
    
    # Continue button
    if st.button("Confirm and Process"):
        # Apply the mapping to rename columns
        standardized_df = apply_column_mapping(df, confirmed_mapping)
        
        # Proceed with data processing
        joints_df, defects_df = process_pipeline_data(standardized_df)
        
        # Process clock and area data (if columns exist)
        if 'clock' in defects_df.columns:
            defects_df["clock_float"] = defects_df["clock"].apply(parse_clock)
        
        if 'length [mm]' in defects_df.columns and 'width [mm]' in defects_df.columns:
            defects_df["area_mm2"] = defects_df["length [mm]"] * defects_df["width [mm]"]
        
        if 'joint number' in defects_df.columns:
            defects_df["joint number"] = defects_df["joint number"].astype("Int64")
        
        # Store processed data in session state
        st.session_state.joints_df = joints_df
        st.session_state.defects_df = defects_df
        st.session_state.processed_data = True
        
        # Rerun to show the visualization section
        st.experimental_rerun()

# Only show the visualization section if we have processed data
if st.session_state.processed_data and st.session_state.joints_df is not None and st.session_state.defects_df is not None:
    joints_df = st.session_state.joints_df
    defects_df = st.session_state.defects_df
    
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
    
    # Add a button to reset and upload a new file
    if st.button("Process New File"):
        st.session_state.processed_data = False
        st.session_state.joints_df = None
        st.session_state.defects_df = None
        st.experimental_rerun()
else:
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis.")