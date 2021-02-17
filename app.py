import streamlit as st
import pandas as pd

st.markdown(
    """
<style>
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: white;
    font-family: monospace;
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #0c0080;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main .csv footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}

</style>
""",
    unsafe_allow_html=True,
)





st.sidebar.markdown("# automl v1")

st.sidebar.markdown("## Upload data")
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv", key="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    st.sidebar.write(f"{df.shape[0]} rows and {df.shape[1]} columns")


mode = st.sidebar.radio(
    label="",
    options=["Exploratory Data Analysis", "Model training", "Model evaluation"],
    index=0,
)

if mode == "Exploratory Data Analysis":
    if uploaded_csv:
        st.write(df)

if mode == "Model training":
    if uploaded_csv:
        y_var = st.selectbox(
            "Choose target variable for prediction",
            df.columns,
        )
        st.write("Target variable selected:", y_var)
        
        x_vars = st.multiselect(
            "Choose dependent variables to be used for modeling",
            df.columns,
        )
        st.write("Dependent (x) variables selected:", x_vars)

if mode == "Model evaluation":
    if uploaded_csv:
        st.write("WIP")
