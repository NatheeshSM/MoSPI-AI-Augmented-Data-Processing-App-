import pandas as pd
import numpy as np
from scipy import stats
from fpdf import FPDF
from datetime import datetime

def impute_missing(df, method):
    for col in df.select_dtypes(include=np.number).columns:
        if method == "Mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == "Median":
            df[col].fillna(df[col].median(), inplace=True)
    return df

def remove_outliers(df, method):
    for col in df.select_dtypes(include=np.number).columns:
        if method == "IQR":
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        elif method == "Z-score":
            col_std = df[col].std(ddof=0)
            if col_std == 0 or pd.isna(col_std):
                continue
            z = (df[col] - df[col].mean()) / col_std
            df = df[z.abs() < 3]
    return df

def apply_weights(df, weight_col):
    if weight_col is None or weight_col == "None":
        return df.select_dtypes(include=np.number).mean()

    weights = pd.to_numeric(df[weight_col], errors="coerce")
    valid = weights.notna()
    if valid.sum() == 0:
        raise ValueError(f"No valid numeric weights found in column '{weight_col}'.")

    num_df = df.select_dtypes(include=np.number).loc[valid]
    w = weights.loc[valid].astype(float)

    weighted_sum = num_df.mul(w, axis=0).sum()
    total_weight = w.sum()
    if total_weight == 0:
        raise ValueError("Sum of weights is zero; cannot compute weighted mean.")

    return weighted_sum / total_weight

def margin_of_error(series, confidence=0.95):
    n = series.count()
    if n < 2:
        return np.nan
    std_err = stats.sem(series, nan_policy='omit')
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return h

def generate_pdf(summary_df, margin_df, file_name, methods):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "MoSPI Data Processing Report", ln=True, align="C")
    pdf.set_font("Arial", size=10)

    pdf.ln(5)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, f"File: {file_name}", ln=True)
    pdf.cell(200, 10, f"Imputation: {methods['impute']} | Outlier Method: {methods['outlier']}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Summary Statistics", ln=True)

    col_width = max(40, int(190 / max(1, len(summary_df.columns))))
    for col in summary_df.columns:
        pdf.cell(col_width, 10, f"{col}", border=1)
    pdf.ln()
    for _, row in summary_df.iterrows():
        for item in row:
            pdf.cell(col_width, 10, f"{item}", border=1)
        pdf.ln()

    pdf.ln(10)
    pdf.cell(200, 10, "Margin of Error", ln=True)
    for _, row in margin_df.iterrows():
        pdf.cell(100, 10, f"{row['Variable']}", border=1)
        pdf.cell(90, 10, f"{row['Margin of Error']:.4f}", border=1)
        pdf.ln()

    return pdf
