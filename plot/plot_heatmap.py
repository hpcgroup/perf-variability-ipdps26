import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

font_prop = font_manager.FontProperties(fname=os.path.join(os.getcwd(), "gillsans.ttf"), size=20)


def plot_heatmap(df_perlmutter, title_perlmutter, df_frontier, title_frontier, pdf_name, png_name, out_dir):
    """
    Draw two heatmaps (Perlmutter and Frontier) stacked vertically and save as pdf and png.
    Rows: file names
    Columns: counters
    Values: spearman correlation
    """
    # Check if DataFrames are empty
    if df_perlmutter.empty and df_frontier.empty:
        print(f"[Warning] Both DataFrames are empty, cannot draw heatmaps")
        return

    # Create a figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 12), constrained_layout=True)

    # Plot Perlmutter data on top subplot
    if not df_perlmutter.empty:
        sns.heatmap(
            df_perlmutter,
            cmap="coolwarm",   # Color map from cool to warm
            vmin=-1, vmax=1,   # Fix value range to [-1, 1]
            annot=False,       # Set to True to display values
            linewidths=0.5,
            ax=ax1
        )
        ax1.set_title(title_perlmutter, fontproperties=font_prop)
        # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45,
        #                     ha='right', font_properties=font_prop)
        ax1.set_xticklabels([])
        ax1.set_yticklabels(ax1.get_yticklabels(), font_properties=font_prop)
    else:
        ax1.text(0.5, 0.5, f"No data for {title_perlmutter}",
                 horizontalalignment='center', verticalalignment='center',
                 fontproperties=font_prop)
        ax1.set_title(title_perlmutter, fontproperties=font_prop)

    # Plot Frontier data on bottom subplot
    if not df_frontier.empty:
        sns.heatmap(
            df_frontier,
            cmap="coolwarm",   # Color map from cool to warm
            vmin=-1, vmax=1,   # Fix value range to [-1, 1]
            annot=False,       # Set to True to display values
            linewidths=0.5,
            ax=ax2
        )
        ax2.set_title(title_frontier, fontproperties=font_prop)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45,
                            ha='right', font_properties=font_prop)
        ax2.set_yticklabels(ax2.get_yticklabels(), font_properties=font_prop)
    else:
        ax2.text(0.5, 0.5, f"No data for {title_frontier}",
                 horizontalalignment='center', verticalalignment='center',
                 fontproperties=font_prop)
        ax2.set_title(title_frontier, fontproperties=font_prop)

    # Save the figure
    plt.savefig(os.path.join(out_dir, pdf_name))
    # plt.savefig(os.path.join(out_dir, png_name), dpi=300)
    plt.close()
    print(f"Heatmap saved: {pdf_name}")

def create_heatmap_df(system):
    base_dir = os.getcwd()
    counter_corr_dir = os.path.join(base_dir, system, 'counter_corr')

    # 1. Find all files matching counter_corr/*.csv
    os.chdir(counter_corr_dir)
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No matching CSV files found. Please check file names or path.")
        return

    # Store {counter: spearman} mapping for each file
    file_to_counter_spearman = {}
    # Collect all possible counters
    all_counters = set()

    # 2. Iterate through all CSV files, read counter and spearman correlations
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")
            continue

        # Check if required columns exist
        required_cols = {"counter", "spearman"}
        if not required_cols.issubset(df.columns):
            print(
                f"File {csv_file} doesn't contain required columns {required_cols}, skipping.")
            continue

        # Create {counter: spearman} mapping, set NaN values to 0
        counter_spearman_map = {}
        for _, row in df.iterrows():
            counter_name = row["counter"]
            spearman_val = row["spearman"]
            if pd.isna(spearman_val):
                spearman_val = 0
            counter_spearman_map[counter_name] = spearman_val

            # Add this counter to the global set
            all_counters.add(counter_name)

        # Store with filename as key
        file_to_counter_spearman[csv_file[:-4]] = counter_spearman_map

    # 3. Construct DataFrame where rows are filenames, columns are counters, default value 0
    all_counters = sorted(list(all_counters))
    sorted_files = sorted(file_to_counter_spearman.keys())
    heatmap_df = pd.DataFrame(0, index=sorted_files,
                              columns=all_counters, dtype=float)

    # Fill the matrix with data
    for f_name, c_map in file_to_counter_spearman.items():
        for c_name, s_val in c_map.items():
            heatmap_df.loc[f_name, c_name] = s_val

    # Remove specified columns from heatmap_df
    # columns_to_remove = [
    #     'lpe_rndzv_puts_0',
    #     'lpe_rndzv_puts_offloaded_0',
    #     'rh:nack_no_matching_conn',
    #     'rh:nack_no_target_trs',
    #     'rh:tct_timeouts',
    #     'rh:sct_in_use',
    #     'atu_cache_evictions',
    #     'rh:connections_cancelled'
    # ]
    # if system == 'frontier':
    #     columns_to_remove += [
    #         'hni_tx_paused_0',
    #         'hni_tx_paused_1',
    #         # 'rh:pct_trs_rsp_nack_drops',
    #         'rh:connections_cancelled',
    #         'pct_trs_rsp_nack_drops',
    #     ]
    # elif system == 'perlmutter':
    #     columns_to_remove += [
    #         'hni_tx_paused_0',
    #         'hni_tx_paused_1',
    #         # 'pct_mst_hit_on_som',
    #         # 'rh:nack_resource_busy',
    #         # 'rh:nack_sequence_error',
    #         # 'rh:nacks',
    #     ]
    # heatmap_df = heatmap_df.drop(columns=columns_to_remove, errors='ignore')

    # Define the columns to keep - if they don't exist, they'll be created with zeros
    columns_to_keep = [
        'atu_cache_hit_base_page_size_0',
        'atu_cache_hit_derivative1_page_size_0',
        'hni_rx_paused_0',
        'hni_rx_paused_1',
        'lpe_net_match_overflow_0',
        'lpe_net_match_priority_0',
        'lpe_net_match_request_0',
        'parbs_tarb_pi_non_posted_blocked_cnt',
        'parbs_tarb_pi_non_posted_cong_rate',
        'parbs_tarb_pi_non_posted_pkts',
        'parbs_tarb_pi_posted_blocked_cnt',
        'parbs_tarb_pi_posted_cong_rate',
        'parbs_tarb_pi_posted_pkts',
        'pct_trs_rsp_nack_drops',
        'pct_mst_hit_on_som',
        'rh:nack_resource_busy',
        'rh:nack_sequence_error',
        'rh:nacks',
        'rh:pkts_cancelled_o',
        'rh:sct_timeouts',
        'rh:spt_timeouts'
    ]

    # Create new DataFrame with only specified columns, adding missing ones with zeros
    filtered_df = pd.DataFrame(index=heatmap_df.index)
    for col in columns_to_keep:
        if col in heatmap_df.columns:
            filtered_df[col] = heatmap_df[col]
        else:
            filtered_df[col] = 0.0

    # Replace the original heatmap_df with our filtered version
    heatmap_df = filtered_df

    os.chdir(base_dir)
    return heatmap_df


def main():
    base_dir = os.getcwd()
    out_dir = base_dir

    heatmap_df_perlmutter = create_heatmap_df('perlmutter')
    heatmap_df_frontier = create_heatmap_df('frontier')

    # ------ 4. Draw heatmap (all files) ------
    plot_heatmap(
        df_perlmutter=heatmap_df_perlmutter,
        title_perlmutter="Perlmutter",
        df_frontier=heatmap_df_frontier,
        title_frontier="Frontier",
        pdf_name=f"spearman_heatmap_all.pdf",
        png_name=f"spearman_heatmap_all.png",
        out_dir=out_dir,
    )

    # # ------ 5. Draw heatmap for allreduce only ------
    # # File is considered allreduce if its name contains "allreduce" (including mpiallreduce/ncclallreduce)
    # allreduce_files = [f for f in sorted_files if "allreduce" in f.lower()]
    # heatmap_df_allreduce = heatmap_df.loc[allreduce_files]

    # plot_heatmap(
    #     df=heatmap_df_allreduce,
    #     title="Spearman Correlation Heatmap (Allreduce)",
    #     pdf_name="spearman_heatmap_allreduce.pdf",
    #     png_name="spearman_heatmap_allreduce.png"
    # )

    # # ------ 6. Draw heatmap for other files ------
    # other_files = [f for f in sorted_files if "allreduce" not in f.lower()]
    # heatmap_df_others = heatmap_df.loc[other_files]

    # plot_heatmap(
    #     df=heatmap_df_others,
    #     title="Spearman Correlation Heatmap (Others)",
    #     pdf_name="spearman_heatmap_others.pdf",
    #     png_name="spearman_heatmap_others.png"
    # )

if __name__ == "__main__":
    main()
