"""
Load consolidated features from Excel, compute Pearson correlation matrix,
save to Excel, and generate a correlation heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    # Load both sheets into a common dataframe
    df1 = pd.read_excel("consolidated_features.xlsx", sheet_name="Sheet1")
    df2 = pd.read_excel("consolidated_features.xlsx", sheet_name="Sheet2")

    # Standardize column names (Sheet2 has 'LMS' vs Sheet1 'lms')
    df2 = df2.rename(columns={"LMS": "lms"}) if "LMS" in df2.columns else df2

    # Concatenate both sheets
    df = pd.concat([df1, df2], ignore_index=True)

    # Drop rows where Feature_Value is NULL/NaN
    df = df.dropna(subset=["Feature_Value"])

    # Ensure Feature_Value is numeric (coerce non-numeric to NaN and drop)
    df["Feature_Value"] = pd.to_numeric(df["Feature_Value"], errors="coerce")
    df = df.dropna(subset=["Feature_Value"])

    # Determine unique order of features based on VIN column
    # Sort by VIN and get feature order as they first appear
    df_sorted = df.sort_values("VIN")
    feature_order = df_sorted["Feature_Name"].drop_duplicates().tolist()

    # Pivot: VIN as rows, Feature_Name as columns, Feature_Value as values
    pivot_df = df.pivot_table(
        index="VIN",
        columns="Feature_Name",
        values="Feature_Value",
        aggfunc="first",  # in case of duplicates, take first
    )

    # Reorder columns according to feature order from VIN-sorted data
    pivot_df = pivot_df.reindex(
        columns=[c for c in feature_order if c in pivot_df.columns]
    )

    pivot_output_excel = "pivoted_features.xlsx"
    pivot_df.to_excel(pivot_output_excel)
    print(f"Pivoted features saved to {pivot_output_excel}")


    # Compute Pearson correlation matrix (pairwise deletion for NaN)
    corr_matrix = pivot_df.corr(method="pearson", min_periods=2)

    # Save correlation matrix to Excel
    output_excel = "correlation_matrix.xlsx"
    corr_matrix.to_excel(output_excel)
    print(f"Correlation matrix saved to {output_excel}")

    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title("Pearson Correlation Matrix of Features")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Correlation heatmap saved to correlation_heatmap.png")


if __name__ == "__main__":
    main()
