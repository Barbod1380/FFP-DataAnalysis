import streamlit as st
import pandas as pd
import numpy as np

# Import functions from modules
from data_processing import process_pipeline_data
from utils import parse_clock
from visualizations import create_unwrapped_pipeline_visualization, create_joint_defect_visualization

# Set page configuration
st.set_page_config(page_title="Pipeline Inspection Visualization", layout="wide")

# Add app title
st.title("Pipeline Inspection Data Visualization")

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