import streamlit as st
import pandas as pd
import plotly.express as px

# … your existing setup code …

# 1️⃣ After reading the CSV into `df`:

# Parse clock-position ("H:MM") → numeric hours
def parse_clock_to_float(val):
    try:
        h, m = val.split(":")
        return float(h) + float(m) / 60.0
    except:
        return None

df["clock_float"] = df["clock"].astype(str).apply(parse_clock_to_float)

# Compute defect area in mm²
df["area_mm2"] = df["length [mm]"] * df["width [mm]"]

# 2️⃣ Identify joint boundaries:
#    Wherever 'joint number' is non-null → mark a new joint start
joint_starts = df.loc[df["joint number"].notna(), "log dist. [m]"].unique()

# 3️⃣ Build the scatter plot
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

# 4️⃣ Add vertical lines at each joint start
ymin, ymax = df["clock_float"].min(), df["clock_float"].max()
for pos in joint_starts:
    fig.add_shape(
        type="line",
        x0=pos, x1=pos,
        y0=ymin, y1=ymax,
        line=dict(color="black", dash="dash"),
        opacity=0.5,
    )

# 5️⃣ Render in Streamlit
st.subheader("🗺 Defect Scatter — Flat Map")
st.plotly_chart(fig, use_container_width=True)
