import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import re # Import regex module

# system = 'frontier'
system = 'perlmutter'
os.path.join(os.getcwd(), system)
base_dir = os.path.join(os.getcwd(), system, 'counter_corr')
out_dir = base_dir

# Adjust font size for potentially more x-axis labels
font_prop_axis = font_manager.FontProperties(fname=os.path.join(os.getcwd(), "gillsans.ttf"), size=20)
font_prop_title = font_manager.FontProperties(fname=os.path.join(os.getcwd(), "gillsans.ttf"), size=20)


def plot_heatmap(df, title, pdf_name, png_name):
    """
    Draw a heatmap based on the given DataFrame and save as pdf and png.
    Rows: file names (or application names)
    Columns: counters (or counter_aggregation)
    Values: spearman correlation
    """
    if df.empty:
        print(f"[Warning] DataFrame is empty, cannot draw {title}")
        return

    plt.figure(figsize=(12, 8))
    # sns.set(font_scale=0.8)  # Adjust text size
    ax = sns.heatmap(
        df,
        cmap="coolwarm",   # Color map from cool to warm
        vmin=-1, vmax=1,   # Fix value range to [-1, 1]
        annot=False,       # Set to True to display values
        linewidths=0.5
    )
    ax.set_title(title, fontproperties=font_prop_title, pad=15) # Increased padding
    # Rotate x-axis labels by 90 degrees for better readability with many columns
    plt.xticks(rotation=90, ha='right', fontproperties=font_prop_axis)
    plt.yticks(rotation=0, fontproperties=font_prop_axis) # Keep y-axis horizontal
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust layout to prevent title overlap
    # plt.show()
    pdf_path = os.path.join(out_dir, pdf_name)
    # png_path = os.path.join(out_dir, png_name)
    print(f"Saving heatmap to: {pdf_path}")
    plt.savefig(pdf_path)
    # print(f"Saving heatmap to: {png_path}")
    # plt.savefig(png_path, dpi=300)
    plt.close()

def main():
    # 1. Find all files matching counter_corr/*.csv
    counter_corr_dir = base_dir
    # Use absolute path for glob to avoid issues with os.chdir
    csv_files = glob.glob(os.path.join(counter_corr_dir, "*.csv"))
    if not csv_files:
        print(f"No matching CSV files found in {counter_corr_dir}. Please check file names or path.")
        return

    # Store {counter: spearman} mapping for each file
    file_to_counter_spearman = {}
    # Collect all possible counters
    all_counters = set()
    # Define target applications and aggregation types
    # target_apps = ["AMG2023", "DeepCAM", "MILC", "nanoGPT"]
    # target_apps = ["AMG2023", "DeepCAM", "nanoGPT", "MILC"]
    target_apps = ["AMG2023", "MILC", "DeepCAM", "nanoGPT"]
    agg_types = ["mean", "max"]
    app_agg_pairs_found = set() # To track which app/agg pairs we found files for

    # 2. Iterate through all CSV files, read counter and spearman correlations
    print(f"Found {len(csv_files)} CSV files. Processing...")
    for csv_file_path in csv_files:
        csv_file_name = os.path.basename(csv_file_path)
        # Parse filename: expecting format like APPNAME_AGGTYPE.csv
        match = re.match(r"^(.*)_(mean|max)\.csv$", csv_file_name, re.IGNORECASE)
        if not match:
            print(f"Skipping file with unexpected name format: {csv_file_name}")
            continue

        app_name, agg_type = match.groups()
        agg_type = agg_type.lower() # Ensure lowercase

        # Check if it's one of the target apps
        if app_name not in target_apps:
            # print(f"Skipping file for non-target app: {csv_file_name}")
            continue

        app_agg_pairs_found.add((app_name, agg_type))

        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            print(f"Error reading file {csv_file_name}: {e}")
            continue

        # Check if required columns exist
        required_cols = {"counter", "spearman"}
        if not required_cols.issubset(df.columns):
            print(f"File {csv_file_name} doesn't contain required columns {required_cols}, skipping.")
            continue

        # Create {counter: spearman} mapping, set NaN values to 0
        counter_spearman_map = {}
        for _, row in df.iterrows():
            counter_name = row["counter"]
            spearman_val = row["spearman"]
            if pd.isna(spearman_val):
                spearman_val = 0.0 # Ensure float
            counter_spearman_map[counter_name] = spearman_val

            # Add this counter to the global set
            all_counters.add(counter_name)

        # Store with filename (without extension) as key
        file_key = csv_file_name[:-4] # e.g., "AMG2023_max"
        file_to_counter_spearman[file_key] = counter_spearman_map

    if not file_to_counter_spearman:
        print("No valid data processed from CSV files for target applications.")
        return

    # 3. Construct the restructured DataFrame
    all_counters = sorted(list(all_counters))
    # Create new column names: counter_aggtype
    new_columns = sorted([f"{c}_{agg}" for c in all_counters for agg in agg_types])

    # Initialize DataFrame with NaNs
    heatmap_df_restructured = pd.DataFrame(np.nan, index=target_apps, columns=new_columns)

    # Fill the matrix with data
    print("Populating restructured heatmap data...")
    for file_key, c_map in file_to_counter_spearman.items():
        match = re.match(r"^(.*)_(mean|max)$", file_key, re.IGNORECASE)
        if not match: continue # Should not happen based on earlier check, but good practice
        app_name, agg_type = match.groups()
        agg_type = agg_type.lower()

        if app_name in target_apps:
            for c_name, s_val in c_map.items():
                new_col_name = f"{c_name}_{agg_type}"
                if new_col_name in heatmap_df_restructured.columns:
                    heatmap_df_restructured.loc[app_name, new_col_name] = s_val
                # else: # This case means a counter appeared in a file but wasn't added to all_counters initially - should not happen
                #     print(f"[Warning] Column '{new_col_name}' not found in DataFrame structure.")

    # Fill remaining NaNs (where a counter_agg combination didn't exist) with 0
    heatmap_df_restructured.fillna(0.0, inplace=True)

    # Remove rows for apps where no data (mean or max) was found
    rows_to_drop = [app for app in target_apps if not any(pair[0] == app for pair in app_agg_pairs_found)]
    if rows_to_drop:
        print(f"Dropping rows with no data found: {rows_to_drop}")
        heatmap_df_restructured = heatmap_df_restructured.drop(index=rows_to_drop)

    if heatmap_df_restructured.empty:
        print("Resulting DataFrame is empty after processing and filtering.")
        return

    # 4. Remove specified columns (adjusting for _mean/_max suffix)
    base_columns_to_remove = [
        'lpe_rndzv_puts_0',
        'lpe_rndzv_puts_offloaded_0',
        'rh:nack_no_matching_conn',
        'rh:nack_no_target_trs',
        'rh:tct_timeouts',
        'rh:sct_in_use',
        'atu_cache_evictions',
        'rh:connections_cancelled'
    ]
    if system == 'frontier':
        base_columns_to_remove += [
            'hni_tx_paused_0',
            'hni_tx_paused_1',
        ]
    elif system == 'perlmutter':
        base_columns_to_remove += [
            # 'hni_tx_paused_0', # Keep these for Perlmutter example
            # 'hni_tx_paused_1',
            'pct_trs_rsp_nack_drops',
            'pct_mst_hit_on_som',
            'rh:nack_resource_busy',
            'rh:nack_sequence_error',
            'rh:nacks',
            'hni_rx_paused_1',
            'hni_tx_paused_1',
            'parbs_tarb_pi_non_posted_cong_rate',
            'parbs_tarb_pi_posted_cong_rate',
            'rh:pkts_cancelled_o',
            'parbs_tarb_pi_non_posted_pkts'
            
        ]

    # Generate the full list of columns to remove (with _mean and _max)
    columns_to_remove_restructured = []
    for base_col in base_columns_to_remove:
        columns_to_remove_restructured.append(f"{base_col}_mean")
        columns_to_remove_restructured.append(f"{base_col}_max")

    print(f"Attempting to remove {len(columns_to_remove_restructured)} columns (mean/max versions)...")
    # Check which columns actually exist before trying to drop
    existing_cols_to_remove = [col for col in columns_to_remove_restructured if col in heatmap_df_restructured.columns]
    if existing_cols_to_remove:
         print(f"Removing columns: {existing_cols_to_remove}")
         heatmap_df_restructured = heatmap_df_restructured.drop(columns=existing_cols_to_remove, errors='ignore')
    else:
         print("None of the specified columns to remove were found in the DataFrame.")
         
    # print(heatmap_df_restructured.columns)
    # print(heatmap_df_restructured['parbs_tarb_pi_non_posted_blocked_cnt_mean'])

    order_cols = [
        'lpe_net_match_request_0_mean',
        'lpe_net_match_request_0_max',
        'parbs_tarb_pi_posted_pkts_mean',
        'lpe_net_match_overflow_0_mean',
        'parbs_tarb_pi_posted_blocked_cnt_mean',
        'parbs_tarb_pi_non_posted_blocked_cnt_mean',
        'hni_rx_paused_0_mean',
        'hni_tx_paused_0_mean',
        'hni_tx_paused_0_max',

        'parbs_tarb_pi_non_posted_blocked_cnt_max',
        'atu_cache_hit_base_page_size_0_mean',
        'atu_cache_hit_base_page_size_0_max',
        'rh:sct_timeouts_mean',
        'rh:sct_timeouts_max',
        'atu_cache_hit_derivative1_page_size_0_mean',
        'atu_cache_hit_derivative1_page_size_0_max',
        'parbs_tarb_pi_posted_pkts_max',
        
        
        'lpe_net_match_overflow_0_max',
        'parbs_tarb_pi_posted_blocked_cnt_max',
        'lpe_net_match_priority_0_mean',
        'rh:spt_timeouts_max',
        'rh:spt_timeouts_mean',
        'lpe_net_match_priority_0_max',
        'hni_rx_paused_0_max',
        
    ]
    
    heatmap_df_restructured = heatmap_df_restructured[order_cols]

    # ------ 5. Draw heatmap (restructured) ------
    if system == 'frontier':
        title = "Frontier"
    elif system == 'perlmutter':
        title = "Correlation of NIC counters with application runtime (Perlmutter)"
    plot_heatmap(
        df=heatmap_df_restructured,
        title=title,
        pdf_name=f"spearman_heatmap_all_{system}.pdf",
        png_name=f"spearman_heatmap_all_{system}.png"
    )
    print("Script finished.")

if __name__ == "__main__":
    main()