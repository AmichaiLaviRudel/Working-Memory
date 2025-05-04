import pandas as pd
import streamlit as st
import time

# Path to CSV file
NPXL_MONITORING_PATH = r"G:\My Drive\Study\Lab\Projects\Code_temp\users_data\Amichai\NPXL recordings _experimental_data.csv"

# Load the experimental data
project_data = pd.read_csv(NPXL_MONITORING_PATH, delimiter=',', low_memory=False)

# Streamlit App
st.title("Neuropixels Data Management")

# Display the editable data table
st_project_data = st.data_editor(
    data=project_data,
    height=400,
    column_config={
        "spike glx file": st.column_config.TextColumn(help="File Name"),
        "status": st.column_config.SelectboxColumn(
            label=None,
            help="Status of the recordings processing",
            default="2p",
            options=["2p", "Shared", "Hive", "CAT", "KS", "Phy", "Tprime"]
        )
    }
)
# Add a save button
if st.button("Save Changes"):
    # Save the modified data to your CSV/database
    st_project_data.to_csv(NPXL_MONITORING_PATH, index=False)  # Adjust filename as needed
    st.toast("Changes saved successfully!",  icon='🎉')
    time.sleep(1.5)
    st.rerun()

st.divider()
st.subheader("Analysis")