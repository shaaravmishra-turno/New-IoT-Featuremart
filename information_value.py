"""
Calculate Information Value (IV) for each feature using roll forward as target.
Uses scorecardpy woebin for IV calculation. NULL values are kept as a separate bucket.
Variables not present for a VIN are assigned NULL for that observation.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

import scorecardpy as sc


def _patched_check_empty_bins(dtm, binning):
    """Pandas-compatible check_empty_bins: drop column before reassign to avoid Categorical conflict."""
    from scorecardpy.woebin import n0, n1

    bin_list = np.unique(dtm.bin.astype(str)).tolist()
    if "nan" in bin_list:
        bin_list.remove("nan")
    binleft = set(
        [re.match(r"\[(.+),(.+)\)", i).group(1) for i in bin_list]
    ).difference(set(["-inf", "inf"]))
    binright = set(
        [re.match(r"\[(.+),(.+)\)", i).group(2) for i in bin_list]
    ).difference(set(["-inf", "inf"]))
    if binleft != binright:
        bstbrks = sorted(list(map(float, ["-inf"] + list(binright) + ["inf"])))
        labels = [
            "[{},{})".format(bstbrks[i], bstbrks[i + 1])
            for i in range(len(bstbrks) - 1)
        ]
        dtm = dtm.drop(columns=["bin"])
        dtm["bin"] = pd.cut(dtm["value"], bstbrks, right=False, labels=labels).astype(
            str
        )
        binning = (
            dtm.groupby(["variable", "bin"], group_keys=False)["y"]
            .agg([n0, n1])
            .reset_index()
            .rename(columns={"n0": "good", "n1": "bad"})
        )
    return binning


def main():
    # Load both sheets into a common dataframe
    df1 = pd.read_excel("consolidated_features.xlsx", sheet_name="Sheet1")
    df2 = pd.read_excel("consolidated_features.xlsx", sheet_name="Sheet2")

    # Standardize column names (Sheet2 has 'LMS' vs Sheet1 'lms')
    df2 = df2.rename(columns={"LMS": "lms"}) if "LMS" in df2.columns else df2

    # Concatenate both sheets
    df = pd.concat([df1, df2], ignore_index=True)

    # Do NOT drop NULL - keep them for separate bucket in IV calculation
    # Variables not present for a VIN will be NULL after pivot

    # Determine unique order of features based on VIN column
    df_sorted = df.sort_values("VIN")
    feature_order = df_sorted["Feature_Name"].drop_duplicates().tolist()

    # Pivot: VIN as rows, Feature_Name as columns, Feature_Value as values
    # VINs without a feature get NULL (NaN) for that column
    pivot_df = df.pivot_table(
        index="VIN",
        columns="Feature_Name",
        values="Feature_Value",
        aggfunc="first",
    )

    # Reorder columns according to feature order from VIN-sorted data
    pivot_df = pivot_df.reindex(
        columns=[c for c in feature_order if c in pivot_df.columns]
    )

    pivot_output_excel = "pivoted_features_information_value.xlsx"
    pivot_df.to_excel(pivot_output_excel)
    print(f"Pivoted features saved to {pivot_output_excel}")

    # Get roll forward per VIN (target variable)
    vin_target = df.groupby("VIN")["roll forward"].first().reset_index()
    dt = pivot_df.reset_index().merge(vin_target, on="VIN", how="inner")

    # Ensure target is integer (0/1)
    dt["roll forward"] = dt["roll forward"].fillna(0).astype(int)
    feat_cols_all = [c for c in dt.columns if c not in ["VIN", "roll forward"]]

    # Calculate IV using scorecardpy woebin
    # Patch check_empty_bins for pandas Categorical compatibility
    # replace_blank=False, ignore_datetime_cols=False for pandas compatibility
    # check_cate_num=False to avoid interactive prompt
    # no_cores=1 avoids multiprocessing issues with pandas Categorical
    with patch("scorecardpy.woebin.check_empty_bins", _patched_check_empty_bins):
        bins = sc.woebin(
            dt,
            y="roll forward",
            replace_blank=False,
            ignore_datetime_cols=False,
            check_cate_num=False,
            no_cores=1,
        )

    # Extract IV from woebin output (total_iv is same for all rows per variable)
    iv_results = []
    for var_name, bin_df in bins.items():
        try:
            iv_val = bin_df["total_iv"].iloc[0] if len(bin_df) > 0 else None
            iv_results.append({"Feature_Name": var_name, "Information_Value": iv_val})
        except Exception as e:
            iv_results.append(
                {"Feature_Name": var_name, "Information_Value": None, "error": str(e)}
            )

    # Add features dropped by woebin (constant columns) with IV=0
    binned_vars = set(bins.keys())
    for col in feat_cols_all:
        if col not in binned_vars:
            iv_results.append({"Feature_Name": col, "Information_Value": 0.0})

    iv_df = pd.DataFrame(iv_results)

    # Sort by IV descending
    iv_df = iv_df.sort_values("Information_Value", ascending=False).reset_index(
        drop=True
    )

    # Concatenate bin details for all features
    bins_list = []
    for var_name, bin_df in bins.items():
        bins_list.append(bin_df)
    bins_df = pd.concat(bins_list, ignore_index=True) if bins_list else pd.DataFrame()

    # Save to Excel with two sheets
    output_excel = "information_value.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        iv_df.to_excel(writer, sheet_name="IV_Summary", index=False)
        if len(bins_df) > 0:
            bins_df.to_excel(writer, sheet_name="Bin_Details", index=False)
    print(f"Information Value results saved to {output_excel} (IV_Summary + Bin_Details)")

    # Create IV bar chart
    iv_plot = iv_df.dropna(subset=["Information_Value"]).head(50)
    if len(iv_plot) > 0:
        plt.figure(figsize=(12, 10))
        plt.barh(
            range(len(iv_plot)),
            iv_plot["Information_Value"].values,
            color="steelblue",
            alpha=0.8,
        )
        plt.yticks(range(len(iv_plot)), iv_plot["Feature_Name"].values, fontsize=8)
        plt.gca().invert_yaxis()
        plt.xlabel("Information Value")
        plt.title("Information Value by Feature (Top 50, roll forward as target)")
        plt.tight_layout()
        plt.savefig("information_value_chart.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("IV visualization saved to information_value_chart.png")


if __name__ == "__main__":
    main()