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

# File uploader section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
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
        
        # Continue with the rest of your original application...
        # Visualization section
        st.header("Visualization")
        
        # ... (rest of the original code)
else:
    st.info("Please upload a CSV file to begin analysis.")