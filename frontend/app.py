import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.processing import impute_missing, remove_outliers, apply_weights, margin_of_error, generate_pdf

st.set_page_config(page_title="MoSPI Data Processing Prototype", layout="wide")
st.title("📊 MoSPI AI-Augmented Data Processing App (Enhanced Prototype)")

file = st.file_uploader("Upload your CSV/Excel survey data", type=["csv", "xlsx"])
if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Preview", "🛠 Cleaning", "📊 Analysis", "📑 Report"])

    with tab1:
        st.subheader("Dataset Overview")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.dataframe(df.head())

        st.subheader("Missing Values")
        st.write(df.isna().sum())

        st.subheader("Basic Stats")
        st.write(df.describe(include="all"))

    with tab2:
        st.sidebar.header("Cleaning Configuration")
        impute_method = st.sidebar.selectbox("Missing Value Imputation", ["Mean", "Median", "None"])
        outlier_method = st.sidebar.selectbox("Outlier Detection", ["None", "IQR", "Z-score"])

        if impute_method != "None":
            df = impute_missing(df, impute_method)

        if outlier_method != "None":
            df = remove_outliers(df, outlier_method)

        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(df.head())

    with tab3:
        st.sidebar.header("Weight Application")
        weight_col = st.sidebar.selectbox("Select weight column (optional)", [None] + list(df.columns))

        try:
            weighted_series = apply_weights(df, weight_col)
            weighted_df = pd.DataFrame(weighted_series,
                                       columns=["Weighted Mean" if weight_col and weight_col != "None" else "Mean"]).reset_index()
            weighted_df.columns = ["Variable", weighted_df.columns[1]]
            st.subheader("📊 Weighted Summary" if weight_col and weight_col != "None" else "📊 Mean Summary")
            st.write(weighted_df)
        except ValueError as e:
            st.warning(str(e))
            weighted_df = pd.DataFrame(df.select_dtypes(include="number").mean(), columns=["Mean"]).reset_index()
            weighted_df.columns = ["Variable", "Mean"]
            st.write(weighted_df)

        margins = {col: margin_of_error(df[col]) for col in df.select_dtypes(include="number").columns}
        margin_df = pd.DataFrame(list(margins.items()), columns=["Variable", "Margin of Error"])
        st.subheader("📏 Margin of Error")
        st.write(margin_df)

        st.subheader("📊 Visualizations")
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            col_choice = st.selectbox("Choose a numeric column", num_cols)

            fig, ax = plt.subplots()
            sns.histplot(df[col_choice], kde=True, ax=ax)
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.boxplot(x=df[col_choice], ax=ax)
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    with tab4:
        st.subheader("📑 Report Generation")
        if st.button("Generate PDF Report"):
            pdf = generate_pdf(weighted_df, margin_df, file.name,
                               methods={"impute": impute_method, "outlier": outlier_method})
            pdf_bytes = pdf.output(dest="S").encode("latin1")
            pdf_output = io.BytesIO(pdf_bytes)

            st.download_button(
                label="⬇ Download Report",
                data=pdf_output,
                file_name="mospi_report.pdf",
                mime="application/pdf"
            )
