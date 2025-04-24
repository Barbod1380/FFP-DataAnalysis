import streamlit as st
import pandas as pd
import plotly.express as px

# Parse clock-position ("H:MM") → numeric hours
def parse_clock_to_float(val):
    try:
        h, m = val.split(":")
        return float(h) + float(m) / 60.0
    except:
        return None

# Page config
st.set_page_config(page_title="FFS Pipeline Analyzer", layout="wide")

# Title and Description
st.title("🛠️ FFS Pipeline Defect Analyzer")
st.markdown("""
Welcome to the **Fitness-for-Service (FFS)** pipeline defect analysis tool.  
Upload a `.csv` file containing your pipeline defect data to begin.
""")

# Upload CSV File
uploaded_file = st.file_uploader("📂 Upload your pipeline data (.csv)", type="csv")

# If file is uploaded, show basic info
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head(10), use_container_width=True)

        # Convert columns to numeric safely
        numeric_cols = ["length [mm]", "width [mm]", "depth [%]", "ERF B31G"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert clock string to float hour format
        df["clock_float"] = df["clock"].astype(str).apply(parse_clock_to_float)

        # Compute defect area in mm²
        df["area_mm2"] = df["length [mm]"] * df["width [mm]"]

        # Identify joint boundaries
        joint_starts = df.loc[df["joint number"].notna(), "log dist. [m]"].unique()

        # Build the scatter plot
        fig = px.scatter(
            df,
            x="log dist. [m]",
            y="clock_float",
            size="area_mm2",
            hover_data=["component / anomaly identification", "depth [%]", "ERF B31G"],
            title="Defect Map — Unwrapped Pipe Surface",
            labels={
                "log dist. [m]": "Distance along pipe (m)",
                "clock_float": "Clock position (hours)"
            }
        )

        # Add vertical lines at joint starts
        ymin, ymax = df["clock_float"].min(), df["clock_float"].max()
        for pos in joint_starts:
            fig.add_shape(
                type="line",
                x0=pos, x1=pos,
                y0=ymin, y1=ymax,
                line=dict(color="black", dash="dash"),
                opacity=0.5,
            )

        # Render the chart
        st.subheader("🗺 Defect Scatter — Flat Map")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")