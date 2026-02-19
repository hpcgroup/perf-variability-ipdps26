#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import pandas as pd

APP = "AMG2023"
NODES = 64
base_path = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar", f"{APP}_logs", f"{NODES}nodes")

def parse_user_alloc_dict_trimmed(user_alloc_str):
    """
    Parse the user_alloc_dict_trimmed field in CSV into a Python dictionary.
    For example: "{'xgao': 4, 'yshaidu': 8}"
    """
    if not isinstance(user_alloc_str, str):
        return {}
    try:
        # Use ast.literal_eval to parse the string form of dict
        data = ast.literal_eval(user_alloc_str)
        if isinstance(data, dict):
            return data
        else:
            return {}
    except (SyntaxError, ValueError):
        return {}

def build_user_columns(df):
    """
    Based on the user_alloc_dict_trimmed column, expand all users that have appeared 
    into separate columns (with user_ prefix). If a user does not appear in a certain job, 
    the value for that column will be 0.
    """
    all_users = set()
    
    # 1) Collect all users that have appeared
    for val in df['user_alloc_dict_trimmed']:
        user_dict = parse_user_alloc_dict_trimmed(val)
        for user in user_dict.keys():
            all_users.add(user)
    
    # Convert to list, fix column order
    all_users = sorted(all_users)
    
    # 2) Add user columns to the original DataFrame
    df_expanded = df.copy()
    for user in all_users:
        col_name = f"user_{user}"
        df_expanded[col_name] = 0  # Initialize to 0
    
    # 3) Fill in the concurrent node count for each row
    for idx, row in df_expanded.iterrows():
        user_dict = parse_user_alloc_dict_trimmed(row['user_alloc_dict_trimmed'])
        for user, alloc_nodes in user_dict.items():
            col_name = f"user_{user}"
            df_expanded.at[idx, col_name] = alloc_nodes
    
    return df_expanded

def main():
    csv_path = os.path.join(base_path, "neighborhood_performance_analysis.csv")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # 1) Read CSV
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[INFO] CSV is empty.")
        return
    
    # If there's no user_alloc_dict_trimmed column, no need to expand
    if 'user_alloc_dict_trimmed' not in df.columns:
        print("[WARN] 'user_alloc_dict_trimmed' column not found in CSV. No user columns will be built.")
        df_expanded = df
    else:
        # 2) Build "wide table" - add user_xxx columns for each user
        df_expanded = build_user_columns(df)
        
    #   Find all columns starting with user_
    user_cols = [c for c in df_expanded.columns if c.startswith("user_") and not c.startswith("user_alloc_dict")]

    #   Sum these columns by row
    df_expanded['all_users_alloc_nodes'] = df_expanded[user_cols].sum(axis=1)
    print(df_expanded[['all_users_alloc_nodes']])
    # 3) Prepare for univariate correlation: only need app_total_time_ms and other columns
    #   Find all numeric columns in the DataFrame (including user_xxx, min_group_ratio, etc.)
    numeric_cols = []
    for col in df_expanded.columns:
        # Simple check if convertible to number
        if pd.api.types.is_numeric_dtype(df_expanded[col]):
            numeric_cols.append(col)
    
    if 'app_total_time_ms' not in numeric_cols:
        print("[WARN] 'app_total_time_ms' is not in numeric columns, cannot compute correlation.")
        return
    # print(df_expanded[numeric_cols])
    # Sum columns from the 6th column onwards
    

    # 4) Only compute correlation with app_total_time_ms
    # corr_series = df_expanded[numeric_cols].corrwith(df_expanded['app_total_time_ms'], method='spearman')
    corr_series = df_expanded[numeric_cols].corrwith(df_expanded['app_total_time_ms'])
    
    # Calculate sum of each numeric column
    max_series = df_expanded[numeric_cols].max()
    print("Sum of numeric columns:")
    print(max_series)
    
    # 5) Print or save results
    #   corr_series is a Series with column names as index, values are Pearson correlation coefficients with app_total_time_ms
    #   The correlation with itself (app_total_time_ms) will be 1, which can be removed
    corr_series = corr_series.drop(labels=['app_total_time_ms'], errors='ignore')
    
    # Here we can sort by absolute value, from most correlated to least correlated
    corr_series_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
    
    # Create a DataFrame with both correlation and sum values
    result_df = pd.DataFrame({
        "correlation": corr_series_sorted,
        "max": max_series.reindex(corr_series_sorted.index)
    })
    result_df = result_df.reset_index().rename({'index': 'username'}, axis=1)
    
    # If you want to save the results to CSV
    corr_out_path = os.path.join(base_path, "neighborhood_user_time_correlation.csv")
    result_df.to_csv(corr_out_path, index=False)
    print(f"\nCorrelation and sum results saved to: {corr_out_path}")

if __name__ == "__main__":
    main()
