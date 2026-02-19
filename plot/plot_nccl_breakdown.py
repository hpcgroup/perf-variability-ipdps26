#!/usr/bin/env python3

import os
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
#######################
font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=14.2)
font_dis_prop = font_manager.FontProperties(fname=font_path, size=11)

system = 'perlmutter'
NAME = "Perlmutter"
# system = 'frontier'
# NAME = "Frontier"

# APP = 'nanoGPT'
APP = 'deepCAM'

colors = ['#D55E00', '#0072B2', '#009E73', '#800080', '#CC79A7', '#E69F00', '#56B4E9']
hatches = ['x', 'xxx', '\\\\', '||','///', '+', 'o', '.', '*', '-', 'ooo', '+++', 'xx', '/', '\\', '***']

if APP == "deepCAM":
    COLOR_HATCH_MAP = {
    "compute":   {"color": colors[0], "hatch": hatches[0]},     # Light Blue, no hatch
    "others":    {"color": colors[1], "hatch": hatches[1]},
    "AllGather":   {"color": colors[3], "hatch": hatches[3]},   # Light yellow, cross hatch
    "mem":      {"color": colors[4], "hatch": hatches[2]},   # Orange, diagonal lines
    "AllReduce": {"color": colors[2], "hatch": hatches[4]},
    }
else:
    COLOR_HATCH_MAP = {
        "compute":   {"color": colors[0], "hatch": hatches[0]},     # Light Blue, no hatch 
        "others":   {"color": colors[1], "hatch": hatches[1]},
        "SendRecv":   {"color": colors[3], "hatch": hatches[3]},
        "mem": {"color": colors[4], "hatch": hatches[2]},
        "ReduceScatter": {"color": colors[5], "hatch": hatches[5]},
        "AllReduce": {"color": colors[2], "hatch": hatches[4]},
        "AllGather": {"color": colors[6], "hatch": hatches[6]},
    }

def plot_six_stacked_bars(df_pivot):

    # Define categories explicitly to ensure 'compute' is handled separately
    all_categories = df_pivot.columns.tolist()
    all_categories.remove("total_time") # Remove total_time column name

    compute_category = "compute"
    other_categories = [cat for cat in all_categories if cat != compute_category]

    # 1) Sort by total_time to find best/worst/quartiles
    df_pivot_sorted = df_pivot.sort_values(by="total_time") # Use a new variable to keep original df_pivot
    best_ts = df_pivot_sorted.index[0]
    worst_ts = df_pivot_sorted.index[-1]
    q1_idx = int(len(df_pivot_sorted) * 0.25)
    q2_idx = int(len(df_pivot_sorted) * 0.5)
    q3_idx = int(len(df_pivot_sorted) * 0.75)
    # Ensure indices are valid
    q1_idx = min(q1_idx, len(df_pivot_sorted) - 1)
    q2_idx = min(q2_idx, len(df_pivot_sorted) - 1)
    q3_idx = min(q3_idx, len(df_pivot_sorted) - 1)
    q1_ts = df_pivot_sorted.index[q1_idx]
    q2_ts = df_pivot_sorted.index[q2_idx]
    q3_ts = df_pivot_sorted.index[q3_idx]

    # 2) Calculate the "average" (mean) across all runs for all categories
    avg_values = df_pivot_sorted[all_categories].mean(axis=0)

    # Construct a DataFrame with average, best, q1, q2, q3, worst rows
    df_plot = pd.DataFrame({
        "Average": avg_values,
        "Best": df_pivot_sorted.loc[best_ts, all_categories],
        "Q1": df_pivot_sorted.loc[q1_ts, all_categories],
        "Q2": df_pivot_sorted.loc[q2_ts, all_categories],
        "Q3": df_pivot_sorted.loc[q3_ts, all_categories],
        "Worst": df_pivot_sorted.loc[worst_ts, all_categories]
    }).T  # Transpose so scenarios are rows

    # 3) Determine plot order: 'compute' first, then others sorted by average ascending
    avg_others = avg_values[other_categories]
    sorted_other_categories = avg_others.sort_values(ascending=True).index.tolist()
    category_order = [compute_category] + sorted_other_categories

    print("\n=== Six Scenarios Data (ms) ===")
    print(df_plot)
    print("\n=== Category Plot Order ===")
    print(category_order)


    # 4) Plotting
    fig, ax = plt.subplots(figsize=(5, 3.5)) # Adjusted size like mpip64
    bar_width = 0.8
    x_labels = df_plot.index.tolist() # ["Average", "Best", "Q1", "Q2", "Q3", "Worst"]
    x_positions = np.arange(len(x_labels))

    bottom_values = np.zeros(len(x_labels))
    used_labels = set()

    # Plot bars category by category in the determined order
    for category in category_order:
        if category not in df_plot.columns:
            print(f"Warning: Category '{category}' not found in plot data. Skipping.")
            continue
        heights = df_plot[category].values
        style = COLOR_HATCH_MAP.get(category, {"color":"#CCCCCC", "hatch":""}) # Default style
        color = style["color"]
        hatch = style["hatch"]
        label = category if category not in used_labels else None
        if label:
             used_labels.add(category)

        ax.bar(x_positions, heights, width=bar_width, bottom=bottom_values,
               color=color, hatch=hatch, edgecolor="black", label=label)
        bottom_values += heights

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontproperties=font_prop)
    y_labels = [f"{y:.1f}" for y in ax.get_yticks()]
    ax.set_yticklabels(y_labels, fontproperties=font_prop)
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.set_ylabel("Time (s)", fontproperties=font_prop)
    if APP == 'deepCAM':
        ax.set_title("Breakdown of " + "DeepCAM" + " runtime (" + NAME + ")", fontproperties=font_prop)
    else:
        ax.set_title("Breakdown of " + APP + " runtime (" + NAME + ")", fontproperties=font_prop)

    # 4. Add Braces to the Worst Bar (at x_positions[-1])
    worst_bar_x = x_positions[-1] # Changed from x_positions[0] to x_positions[-1]
    brace_x_offset = bar_width * 0.7 # Changed offset to be positive for right side
    text_x_offset = bar_width * 0.9  # Changed offset to be positive for right side
    cap_size = 0.1 # Use a fixed cap size in x-axis data coordinates

    # Calculate heights for braces on the Average bar
    compute_height = df_plot.loc["Worst", compute_category]
    total_height = df_plot.loc["Worst", all_categories].sum()
    non_compute_start_y = compute_height
    non_compute_end_y = total_height
    print(f"compute_height: {compute_height}, total_height: {total_height}")

    # --- Brace for Compute ---
    if compute_height > 1e-6: # Check if compute time is non-negligible
        brace_compute_x = worst_bar_x + brace_x_offset
        text_compute_x = worst_bar_x + text_x_offset
        # Vertical line
        ax.plot([brace_compute_x, brace_compute_x], [0, compute_height], color='black', lw=1)
        # Caps
        ax.plot([brace_compute_x - cap_size, brace_compute_x + cap_size], [0, 0], color='black', lw=1)
        ax.plot([brace_compute_x - cap_size, brace_compute_x + cap_size], [compute_height, compute_height], color='black', lw=1)
        # Text
        ax.text(text_compute_x, compute_height / 2, 'Compute', ha='left', va='center', fontproperties=font_prop)

    # --- Brace for NCCL+Mem ---
    if non_compute_end_y > non_compute_start_y + 1e-6: # Check if non-compute time exists
        brace_non_compute_x = worst_bar_x + brace_x_offset
        text_non_compute_x = worst_bar_x + text_x_offset
        # Vertical line
        ax.plot([brace_non_compute_x, brace_non_compute_x], [non_compute_start_y, non_compute_end_y], color='black', lw=1)
        # Caps
        ax.plot([brace_non_compute_x - cap_size, brace_non_compute_x + cap_size], [non_compute_start_y, non_compute_start_y], color='black', lw=1)
        ax.plot([brace_non_compute_x - cap_size, brace_non_compute_x + cap_size], [non_compute_end_y, non_compute_end_y], color='black', lw=1)
        # Text
        if system == 'perlmutter':
            ax.text(text_non_compute_x, (non_compute_start_y + non_compute_end_y) / 2, 'NCCL+Mem', ha='left', va='center', fontproperties=font_prop)
        else:
            ax.text(text_non_compute_x, (non_compute_start_y + non_compute_end_y) / 2, 'RCCL+Mem', ha='left', va='center', fontproperties=font_prop)

    # worst_total_time = sum(df_plot["Worst"].values())
    worst_total_time = df_plot.loc["Worst", all_categories].sum() # Ensure worst_total_time is
    if APP == 'nanoGPT' and NAME == 'Perlmutter':
        ax.set_ylim(bottom=0, top=14)
        ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
        ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14])
    elif APP == 'nanoGPT' and NAME == 'Frontier':
        ax.set_ylim(bottom=0, top=40)
        ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
        ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30, 35, 40])
    elif APP == 'deepCAM' and NAME == 'Perlmutter':
        ax.set_ylim(bottom=0, top=0.6)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    elif APP == 'deepCAM' and NAME == 'Frontier':
        ax.set_ylim(bottom=0, top=25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_yticklabels([0, 5, 10, 15, 20, 25])
    else:
        ax.set_ylim(bottom=0, top=worst_total_time * 1.1) # Set top limit to 1.2x Worst time

    # Create legend outside the plot area
    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend items based on plot order (bottom to top)
    handles_ordered = [handles[labels.index(cat)] for cat in reversed(category_order) if cat in labels]
    labels_ordered = [cat for cat in reversed(category_order) if cat in labels]
    if APP == 'deepCAM' and NAME == 'Perlmutter':
        ax.legend(handles[::-1], labels[::-1], ncol=1, prop=font_prop,
              columnspacing=0.2, handletextpad=0.2, frameon=False,
              labelspacing=0.3)
    elif APP == 'nanoGPT' and NAME == 'Perlmutter':
        ax.legend(handles[::-1], labels[::-1], ncol=1, prop=font_prop,
              columnspacing=0.2, handletextpad=0.2, frameon=False,
              labelspacing=0.3)
    else:
        ax.legend(handles[::-1], labels[::-1], ncol=2, prop=font_prop,
              columnspacing=0.2, handletextpad=0.2, frameon=False,
              labelspacing=0.3)
    
    plt.tight_layout()
    if APP == 'deepCAM' and NAME == 'Perlmutter':
        plt.subplots_adjust(left=0.12, bottom=0.09, right=0.81, top=0.93)
    else:
        plt.subplots_adjust(left=0.11, bottom=0.085, right=0.81, top=0.935)

    plt.savefig(APP + "_64nodes_six_breakdown_comp_first_braces_" + system + ".pdf") # Updated filename
    print(f"Saved Compute+NCCL+Mem stacked bar plot (Compute first, others asc avg time, with braces)")


##############################################################################
# 3. Plot distribution histograms for two specified NCCL routines (New Function)
##############################################################################
def plot_routine_distribution(df_pivot, routine1_name, range_bins1, routine2_name, range_bins2, num_bins=5):
    """
    Plots distribution histograms for two specified NCCL routines based on custom bin ranges,
    using data from the pivoted DataFrame.

    Args:
        df_pivot (pd.DataFrame): DataFrame with timestamps as index, categories as columns,
                                 and time values (in milliseconds).
        routine1_name (str): Name of the first NCCL routine (must be a column in df_pivot).
        range_bins1 (list): List of 5 floats defining the start points for the bins (in ms).
        routine2_name (str): Name of the second NCCL routine (must be a column in df_pivot).
        range_bins2 (list): List of 5 floats defining the start points for the bins (in ms).
        num_bins (int): The number of bins to create (should match len(range_bins)).
    """
    if routine1_name not in df_pivot.columns or routine2_name not in df_pivot.columns:
        print(f"Error: One or both routines ('{routine1_name}', '{routine2_name}') not found in DataFrame columns.")
        print(f"Available columns: {df_pivot.columns.tolist()}")
        return
    if len(range_bins1) != num_bins or len(range_bins2) != num_bins:
        print(f"Error: range_bins arrays must have length {num_bins}.")
        return

    # --- Data Extraction (Times are already in milliseconds) ---
    runtimes1 = df_pivot[routine1_name].tolist()
    runtimes2 = df_pivot[routine2_name].tolist()

    # Filter out zero/very small times for distribution calculation
    valid_runtimes1 = [t for t in runtimes1 if t > 1e-6] # Use small threshold for float comparison
    valid_runtimes2 = [t for t in runtimes2 if t > 1e-6]

    # --- Plotting Setup ---
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharey=True) # Share Y axis
    plot_details = [
        {"ax": axs[0], "name": routine1_name, "runtimes": valid_runtimes1, "bins": range_bins1, "color": "skyblue"},
        {"ax": axs[1], "name": routine2_name, "runtimes": valid_runtimes2, "bins": range_bins2, "color": "lightcoral"}
    ]

    # --- Process and Plot Each Routine ---
    for details in plot_details:
        ax = details["ax"]
        name = details["name"]
        runtimes = details["runtimes"]
        range_bins = details["bins"] # These are bin start points
        color = details["color"]

        if not runtimes:
            print(f"No positive {name} times found, cannot plot distribution.")
            ax.set_title(f"Distribution of NCCL {name} Time (No Data)")
            ax.set_xlabel("Time Range (ms)")
            ax.set_ylabel("Number of Runs")
            ax.set_xticks(range(num_bins))
            ax.set_xticklabels([f"Bin {i+1}" for i in range(num_bins)]) # Placeholder labels
            continue

        min_time = min(runtimes)
        max_time = max(runtimes)
        print(f"{name} Runtime range (ms): Min={min_time:.2f}, Max={max_time:.2f}")

        # Determine the upper bound for the last bin
        effective_max_time = max(max_time, range_bins[num_bins-1])
        # Add a small delta if max_time is exactly the last bin start to ensure it's included
        if abs(effective_max_time - range_bins[num_bins-1]) < 1e-9:
             effective_max_time += 0.1 # Add small delta (in ms)

        # --- Manual Bin Counting ---
        counts = [0] * num_bins
        bin_edges = range_bins + [effective_max_time] # Create edges: [b0, b1, b2, b3, b4, max]

        for time in runtimes:
            # Find the bin index
            # Bin 0: [b0, b1)
            # Bin 1: [b1, b2)
            # ...
            # Bin 4: [b4, max] (inclusive of max)
            if time < bin_edges[0]:
                 continue # Skip times below the first bin start

            assigned = False
            for i in range(num_bins - 1): # Check bins 0 to 3
                if time >= bin_edges[i] and time < bin_edges[i+1]:
                    counts[i] += 1
                    assigned = True
                    break
            # Check last bin (inclusive upper bound)
            if not assigned and time >= bin_edges[num_bins-1] and time <= bin_edges[num_bins]:
                 counts[num_bins-1] += 1


        # --- Generate Categorical Labels ---
        bin_labels = []
        for i in range(num_bins - 1):
            bin_labels.append(f"{range_bins[i]:.2f}-{range_bins[i+1]:.2f}")
        # Label for the last bin: [b4, max]
        bin_labels.append(f"{range_bins[num_bins-1]:.2f}-{effective_max_time:.2f}")

        # --- Plotting Bars ---
        x_pos = np.arange(num_bins)
        bars = ax.bar(x_pos, counts, align='center', color=color, edgecolor='black', label=name)

        ax.set_ylabel("Number of Runs", fontproperties=font_dis_prop)
        ax.set_xlabel("Time Range (ms)", fontproperties=font_dis_prop) # Unit is ms
        ax.legend(prop=font_dis_prop)
        # if system == 'perlmutter':
        #     ax.set_title(f"Distribution of NCCL {name} Time", fontproperties=font_prop)
        # else:
        #     ax.set_title(f"Distribution of RCCL {name} Time", fontproperties=font_prop)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=20)

        # Add count labels above bars
        max_count = max(counts) if counts else 1
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                # Adjust vertical offset based on max_count to prevent overlap
                ax.text(bar.get_x() + bar.get_width() / 2., height + max_count * 0.02,
                        f'{height}', ha='center', va='bottom', fontproperties=font_dis_prop)

    # --- Final adjustments ---
    if system == 'perlmutter':
        fig.suptitle(f"Distribution of Top 2 most variable routines", fontproperties=font_dis_prop)
    else:
        fig.suptitle(f"Distribution of Top 2 most variable routines", fontproperties=font_dis_prop)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout

    # Update filename
    filename_base = f"{APP}_routine_distribution_{routine1_name}_{routine2_name}_categorical_" + system
    plt.savefig(filename_base + ".pdf")
    print(f"Saved routine distribution plot ({routine1_name} & {routine2_name}, categorical bins)")


def main():
    df_pivot = pd.read_csv(os.path.join(os.getcwd(), system, "breakdown", (APP + ".csv")))
    # Plot 1: Stacked bars (using the filtered data)
    plot_six_stacked_bars(df_pivot)

    # if system == 'perlmutter':
    #     if APP == 'nanoGPT':
    #         # Bins are in milliseconds
    #         range_bins1 = [0.3, 0.4, 0.5, 0.9, 1.3]    # For Allreduce
    #         range_bins2 = [1.6, 1.8, 2.0, 2.2, 2.4]    # For AllGather (Note: User request was Allgather, mapping maps to AllGather)
    #         # Check if 'Allgather' exists, if not, maybe 'AllGather'? Let's assume 'AllGather' from mapping.
    #         routine1 = "AllReduce"
    #         routine2 = "AllGather" # Use the capitalized version from map_name_to_category
    #     elif APP == 'deepCAM':
    #         range_bins1 = [0.05, 0.1, 0.15, 0.2, 0.3]
    #         range_bins2 = [0.03, 0.04, 0.05, 0.06, 0.07]
    #         routine1 = "AllReduce"
    #         routine2 = "mem"
    # elif system == 'frontier':
    #     if APP == 'nanoGPT':
    #         # Bins are in milliseconds
    #         range_bins1 = [9.5, 9.9, 10.3, 10.7, 11.1]    # For Allreduce
    #         range_bins2 = [7.7, 7.8, 7.9, 8.0, 8.1]    # For AllGather (Note: User request was Allgather, mapping maps to AllGather)
    #         # Check if 'Allgather' exists, if not, maybe 'AllGather'? Let's assume 'AllGather' from mapping.
    #         routine1 = "AllReduce"
    #         routine2 = "AllGather" # Use the capitalized version from map_name_to_category
    #     elif APP == 'deepcam':
    #         range_bins1 = [0, 1.5, 2, 2.5, 4]
    #         range_bins2 = [0.02, 0.03, 0.04, 0.05, 0.06]
    #         routine1 = "AllReduce"
    #         routine2 = "mem"

    # plot_routine_distribution(df_pivot, routine1, range_bins1, routine2, range_bins2, num_bins=5)


if __name__ == "__main__":
    main()
