import streamlit as st
import pandas as pd

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
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")