
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("AI Data Analysis Dashboard")
st.write("Upload your dataset to analyze your data.")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values by Column")
    st.bar_chart(df.isnull().sum())

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) > 0:
        st.subheader("Data Visualization")

        selected_column = st.selectbox(
            "Select a column to visualize",
            numeric_columns
        )

        fig, ax = plt.subplots()
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_column}")
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin analysis.")
