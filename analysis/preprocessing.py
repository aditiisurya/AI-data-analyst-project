import pandas as pd
import numpy as np

def run_preprocessing_pipeline(dfs_dict):
    """
    Cleans datasets and returns a tuple: (cleaned_dfs_dict, report_metrics)
    """
    cleaned_dict = {}
    reports = {}

    for name, df in dfs_dict.items():
        # Copy to avoid mutating original session data prematurely
        df_clean = df.copy()

        # Initial metrics
        rows_before = len(df_clean)
        missing_before = df_clean.isna().sum().sum()
        duplicates_before = df_clean.duplicated().sum()

        # Step 1: Drop Duplicates
        df_clean = df_clean.drop_duplicates()
        rows_after_dedup = len(df_clean)
        duplicates_removed = rows_before - rows_after_dedup

        # Step 2: Handle Missing Values
        missing_handled = 0
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                missing_count = df_clean[col].isna().sum()
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Fill numeric with mean
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    # Fill categorical with mode
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                missing_handled += missing_count

        # Step 3: Standardize Column Names
        df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')

        # Step 4: Detect Column Types
        col_types = {
            "Numeric": len(df_clean.select_dtypes(include=[np.number]).columns),
            "Categorical": len(df_clean.select_dtypes(include=['object', 'category']).columns),
            "Datetime": len(df_clean.select_dtypes(include=['datetime']).columns)
        }

        # Final metrics
        rows_after = len(df_clean)

        report = {
            "Rows Before": rows_before,
            "Rows After": rows_after,
            "Duplicates Removed": duplicates_removed,
            "Missing Handled": missing_handled,
            "Column Types": col_types
        }

        cleaned_dict[name] = df_clean
        reports[name] = report

    return cleaned_dict, reports
