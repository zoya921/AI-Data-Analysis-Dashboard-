
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(
    page_title="AI Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Data Analysis Dashboard")
st.caption("Explore data • Clean datasets • Discover insights • Generate reports")

# 1. DATASET UPLOAD
st.sidebar.header("Dataset Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

try:
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".xlsx"):
            original_df = pd.read_excel(uploaded_file)
        else:
            original_df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        original_df = pd.read_csv("sample_sales.csv")
        dataset_name = "Sample Sales Dataset"

    if original_df.empty or len(original_df.columns) == 0:
        st.error("The uploaded dataset is empty.")
        st.stop()

except Exception as error:
    st.error(f"Could not load dataset: {error}")
    st.stop()

# 2. DATA CLEANING
st.sidebar.header("Data Cleaning")

handle_missing = st.sidebar.selectbox(
    "Handle missing values",
    ["Keep unchanged", "Fill numeric with mean", "Fill with zero",
     "Drop rows with missing values"]
)

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows", value=True
)

df = original_df.copy()
duplicate_count = int(df.duplicated().sum())

if remove_duplicates:
    df = df.drop_duplicates()

if handle_missing == "Fill numeric with mean":
    for column in df.select_dtypes(include="number").columns:
        df[column] = df[column].fillna(df[column].mean())
elif handle_missing == "Fill with zero":
    df = df.fillna(0)
elif handle_missing == "Drop rows with missing values":
    df = df.dropna()

# 3. DATASET OVERVIEW
st.subheader("Dataset Overview")
st.caption(f"Current dataset: {dataset_name}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows", f"{df.shape[0]:,}")
c2.metric("Total Columns", df.shape[1])
c3.metric("Missing Values", int(df.isnull().sum().sum()))
c4.metric("Duplicates Removed", duplicate_count if remove_duplicates else 0)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Preview", "Cleaning Summary", "Statistics",
    "Visualizations", "AI Insights", "Report"
])

with tab1:
    st.subheader("Explore Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Column Information")
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(column_info, use_container_width=True)

    search = st.text_input("Search all columns")
    if search:
        mask = df.astype(str).apply(
            lambda column: column.str.contains(
                search, case=False, na=False
            )
        ).any(axis=1)
        st.write(f"Matching records: {int(mask.sum())}")
        st.dataframe(df[mask], use_container_width=True)

    st.subheader("Filter Records")
    filter_column = st.selectbox(
        "Select a column to filter",
        ["None"] + df.columns.tolist()
    )

    if filter_column != "None":
        if pd.api.types.is_numeric_dtype(df[filter_column]):
            minimum = float(df[filter_column].min())
            maximum = float(df[filter_column].max())

            if minimum < maximum:
                low, high = st.slider(
                    "Choose value range",
                    min_value=minimum,
                    max_value=maximum,
                    value=(minimum, maximum)
                )
                filtered_df = df[
                    df[filter_column].between(low, high)
                ]
            else:
                filtered_df = df
        else:
            options = df[filter_column].dropna().unique().tolist()
            chosen = st.multiselect(
                "Choose values",
                options,
                default=options
            )
            filtered_df = df[df[filter_column].isin(chosen)]

        st.write(f"Filtered records: {len(filtered_df)}")
        st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.subheader("Data Cleaning Summary")
    st.write(f"Original rows: {len(original_df)}")
    st.write(f"Rows after cleaning: {len(df)}")
    st.write(f"Duplicate rows found: {duplicate_count}")
    st.write(f"Missing cells remaining: {int(df.isnull().sum().sum())}")
    st.write(f"Missing-value method: {handle_missing}")

    st.subheader("Missing Values by Column")
    st.bar_chart(df.isnull().sum())

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        st.subheader("Optional Normalization")
        normalize = st.checkbox("Normalize numeric columns to 0–1")

        if normalize:
            normalized_df = df.copy()
            for column in numeric_columns:
                minimum = normalized_df[column].min()
                maximum = normalized_df[column].max()
                if pd.notna(minimum) and maximum != minimum:
                    normalized_df[column] = (
                        (normalized_df[column] - minimum)
                        / (maximum - minimum)
                    )
            st.dataframe(normalized_df, use_container_width=True)
            st.download_button(
                "Download Normalized Data",
                normalized_df.to_csv(index=False).encode("utf-8"),
                "normalized_data.csv",
                "text/csv"
            )

with tab3:
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    numeric_df = df.select_dtypes(include="number")
    if len(numeric_df.columns) >= 2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(
            numeric_df.corr(), annot=True, cmap="coolwarm",
            fmt=".2f", ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numeric columns are needed for correlation.")

with tab4:
    st.subheader("Interactive Visualizations")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        ["Histogram", "Bar Chart", "Line Chart",
         "Pie Chart", "Scatter Plot", "Correlation Heatmap"]
    )

    if chart_type == "Histogram" and numeric_columns:
        column = st.selectbox("Numeric column", numeric_columns)
        fig, ax = plt.subplots()
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)
        plt.close(fig)

    elif chart_type == "Bar Chart":
        if categorical_columns:
            category = st.selectbox("Category column", categorical_columns)
            counts = df[category].value_counts().head(15)
            st.bar_chart(counts)
        else:
            st.info("No categorical columns available.")

    elif chart_type == "Line Chart":
        if numeric_columns:
            column = st.selectbox("Value column", numeric_columns)
            st.line_chart(df[column].reset_index(drop=True))
        else:
            st.info("No numeric columns available.")

    elif chart_type == "Pie Chart":
        if categorical_columns:
            category = st.selectbox("Pie category", categorical_columns)
            counts = df[category].value_counts().head(8)
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            ax.set_title(f"Distribution of {category}")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No categorical columns available.")

    elif chart_type == "Scatter Plot":
        if len(numeric_columns) >= 2:
            x_col = st.selectbox("X-axis", numeric_columns)
            y_col = st.selectbox(
                "Y-axis", [c for c in numeric_columns if c != x_col]
            )
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("At least two numeric columns are required.")

    elif chart_type == "Correlation Heatmap":
        if len(numeric_columns) >= 2:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.heatmap(
                df[numeric_columns].corr(),
                annot=True, cmap="coolwarm", fmt=".2f", ax=ax
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("At least two numeric columns are required.")

with tab5:
    st.subheader("Automated Data Insights")
    st.caption(
        "Rule-based insights generated from dataset statistics; "
        "these are not outputs from a trained AI model."
    )

    st.write(
        f"The dataset contains {len(df)} records and "
        f"{len(df.columns)} columns."
    )

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    if not numeric_df.empty:
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if series.empty:
                continue

            st.write(
                f"**{column}:** mean = {series.mean():.2f}, "
                f"median = {series.median():.2f}, "
                f"minimum = {series.min():.2f}, "
                f"maximum = {series.max():.2f}"
            )

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = int(
                ((series < q1 - 1.5 * iqr) |
                 (series > q3 + 1.5 * iqr)).sum()
            ) if iqr > 0 else 0

            if outliers:
                st.warning(
                    f"{column}: {outliers} potential outlier(s) "
                    "detected using the IQR method."
                )

    for column in categorical_df.columns:
        modes = df[column].mode(dropna=True)
        if not modes.empty:
            st.write(
                f"Most common value in **{column}**: "
                f"{modes.iloc[0]}"
            )

    if "Sales" in df.columns and pd.api.types.is_numeric_dtype(df["Sales"]):
        st.subheader("Business Recommendations")
        if df["Sales"].notna().any():
            st.write(
                "Review high-sales categories and products to identify "
                "opportunities for stock planning and marketing."
            )
            if "Category" in df.columns:
                category_sales = df.groupby("Category")["Sales"].sum()
                if not category_sales.empty:
                    st.write(
                        f"Highest total-sales category: "
                        f"**{category_sales.idxmax()}**."
                    )

with tab6:
    st.subheader("Download Analysis Report")

    report = [
        "AI-POWERED DATA ANALYSIS DASHBOARD",
        f"Dataset: {dataset_name}",
        "",
        "DATASET OVERVIEW",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Missing values remaining: {int(df.isnull().sum().sum())}",
        f"Duplicate rows originally found: {duplicate_count}",
        "",
        "COLUMN NAMES",
        ", ".join(map(str, df.columns)),
        "",
        "STATISTICAL SUMMARY",
        df.describe(include="all").to_string(),
        "",
        "KEY FINDINGS"
    ]

    numeric_df = df.select_dtypes(include="number")
    for column in numeric_df.columns:
        values = df[column].dropna()
        if not values.empty:
            report.append(
                f"{column}: mean={values.mean():.2f}, "
                f"min={values.min():.2f}, max={values.max():.2f}"
            )

    report.extend([
        "",
        "RECOMMENDATIONS",
        "Review missing data and potential outliers before making decisions.",
        "Compare category performance and investigate meaningful trends."
    ])

    report_text = "\n".join(report)
    st.text_area("Report Preview", report_text, height=350)

    st.download_button(
        "Download Analysis Report",
        report_text,
        "data_analysis_report.txt",
        "text/plain"
    )

    st.download_button(
        "Download Cleaned Dataset",
        df.to_csv(index=False).encode("utf-8"),
        "cleaned_dataset.csv",
        "text/csv"
    )

st.sidebar.caption("AI-assisted analytics | Educational project")
