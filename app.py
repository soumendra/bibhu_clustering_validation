import streamlit as st



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
    if uploaded_csv_judgment:
        st.write("WIP")

if mode == "Model evaluation":
    if uploaded_csv_judgment:
        st.write("WIP")
