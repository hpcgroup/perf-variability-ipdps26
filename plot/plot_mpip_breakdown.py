#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as font_manager

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=14.2)
font_dis_prop = font_manager.FontProperties(fname=font_path, size=16)

# system = 'perlmutter'
# NAME = "Perlmutter"
system = 'frontier'
NAME = "Frontier"
APP = 'AMG2023'
# APP = 'MILC'
save_dir = os.getcwd()

colors = ['#D55E00', '#0072B2', '#009E73', '#800080', '#CC79A7', '#E69F00', '#56B4E9']

##############################################################################
# 3) Identify Best / Worst runs and calculate Average for all runs
##############################################################################
def compute_best_avg_worst(runs):
    if not runs:
        return (None, {}, {}, None, {})

    # Sort runs by total AppTime ascending
    # runs_sorted_by_apptime = sorted(runs, key=lambda x: x[1]['app_time']) # Already sorted in main
    runs_sorted_by_apptime = runs # Assume runs are pre-sorted by app_time

    best_run_name, best_data     = runs_sorted_by_apptime[0]
    worst_run_name, worst_data   = runs_sorted_by_apptime[-1]

    # Calculate average time: sum each call's time across all runs and divide by number of runs
    n_runs   = len(runs)
    all_mpi_calls = set()
    total_app_time_sum = 0.0
    total_mpi_time_sum = 0.0 # Sum of '*' line MPI times
    mpi_calls_sum_dict = {}

    for _, run_data in runs:
        all_mpi_calls.update(run_data['mpi_calls'].keys())
        total_app_time_sum += run_data['app_time']
        total_mpi_time_sum += run_data['total_mpi_time']
        for call, time_s in run_data['mpi_calls'].items():
            mpi_calls_sum_dict[call] = mpi_calls_sum_dict.get(call, 0.0) + time_s

    avg_mpi_calls = {}
    for c in all_mpi_calls:
        avg_mpi_calls[c] = mpi_calls_sum_dict.get(c, 0.0) / n_runs

    avg_data = {
        'mpi_calls': avg_mpi_calls,
        'app_time': total_app_time_sum / n_runs,
        'total_mpi_time': total_mpi_time_sum / n_runs
    }

    return (best_run_name, best_data, avg_data, worst_run_name, worst_data)

##############################################################################
# 4) Compress time_dict into "Top K + Others"
##############################################################################
def top_k_plus_others(time_dict, k=7):
    """
    Given a {call: time_s, ...}, sort by time_s in descending order, select the top k,
    and combine the rest into "Others".
    Returns a new dict.
    """
    if not time_dict:
        return {}

    items_sorted = sorted(time_dict.items(), key=lambda x: x[1], reverse=True)
    top = items_sorted[:k]
    rest = items_sorted[k:]

    new_dict = {}
    # Add top k calls
    for (name, val) in top:
        new_dict[name] = val

    # Combine the rest into "Others"
    if rest:
        others_sum = sum(x[1] for x in rest)
        new_dict["Others"] = others_sum
        
        others_functions = [x[0] for x in rest]
        print("Functions included in 'Others':", others_functions)

    return new_dict


##############################################################################
# 5) Plot six stacked bars: Avg / Best / Q1 / Q2 / Q3 / Worst (Compute + Top7 MPI + Others)
##############################################################################

hatches = ['x', 'xxx', '\\\\', '||','///', '+', 'o', '.', '*', '-', 'ooo', '+++', 'xx', '/', '\\', '***']
if APP == "MILC":
    COLOR_HATCH_MAP = {
    "Compute":   {"color": colors[0], "hatch": hatches[0]},     # Light Blue, no hatch
    "Others":    {"color": colors[1], "hatch": hatches[1]},
    "Allreduce":   {"color": colors[2], "hatch": hatches[4]},   # Light yellow, cross hatch
    "Test":      {"color": colors[3], "hatch": hatches[3]},   # Orange, diagonal lines
    }
else:
    COLOR_HATCH_MAP = {
        "Compute":   {"color": colors[0], "hatch": hatches[0]},     # Light Blue, no hatch 
        "Others":   {"color": colors[1], "hatch": hatches[1]},
        "Iprobe":   {"color": colors[3], "hatch": hatches[2]},
        "Allreduce": {"color": colors[2], "hatch": hatches[4]},
        "Irecv": {"color": colors[4], "hatch": hatches[3]},
        "Isend": {"color": colors[5], "hatch": hatches[5]},
        "Waitall": {"color": colors[6], "hatch": hatches[6]},
    }
def compute_quartiles(runs_sorted_by_apptime):
    """
    Input: runs = [(run_name, {'mpi_calls': {...}, 'app_time': T1, 'mpi_time': T2}), ...]
           Assumes runs are pre-sorted by 'app_time'.
    Returns: (q1_run_name, q1_data, q2_run_name, q2_data, q3_run_name, q3_data)
    """
    if not runs_sorted_by_apptime:
        return (None, {}, None, {}, None, {})

    # Calculate quartile positions
    n_runs = len(runs_sorted_by_apptime)
    q1_idx = n_runs // 4
    q2_idx = n_runs // 2
    q3_idx = (3 * n_runs) // 4

    # Ensure indices are valid
    q1_idx = min(q1_idx, n_runs - 1)
    q2_idx = min(q2_idx, n_runs - 1)
    q3_idx = min(q3_idx, n_runs - 1)


    # Get runs at quartile positions
    q1_run_name, q1_data = runs_sorted_by_apptime[q1_idx]
    q2_run_name, q2_data = runs_sorted_by_apptime[q2_idx]
    q3_run_name, q3_data = runs_sorted_by_apptime[q3_idx]

    return (q1_run_name, q1_data, q2_run_name, q2_data, q3_run_name, q3_data)

def plot_six_stacked_bars(avg_data, best_data, q1_data, q2_data, q3_data, worst_data, k=7):
    """
    Plot 6 stacked bars (Avg/Best/Q1/Q2/Q3/Worst), including Compute time
    and top-k MPI calls + Others. Stacks Compute at the bottom, then other
    categories based on average time ascending. Adds braces to the Average bar.
    Input dictionaries contain 'mpi_calls', 'app_time', 'total_mpi_time'.
    """
    datasets = {
        "Average": avg_data,
        "Best": best_data,
        "Q1": q1_data,
        "Q2": q2_data,
        "Q3": q3_data,
        "Worst": worst_data
    }

    plot_data = {}
    all_mpi_calls_in_topk = set()

    # 1. Calculate Compute time and apply top_k_plus_others to MPI calls for each dataset
    for name, data in datasets.items():
        compute_time = data['app_time'] - data['total_mpi_time']
        compute_time = max(0, compute_time)
        mpi_calls_topk = top_k_plus_others(data['mpi_calls'], k)
        all_mpi_calls_in_topk.update(mpi_calls_topk.keys())
        plot_data[name] = {'Compute': compute_time, **mpi_calls_topk}

    # 2. Determine category order: Compute first, then others sorted by average time ASCENDING
    avg_mpi_calls_topk = top_k_plus_others(avg_data['mpi_calls'], k)
    mpi_category_names = list(avg_mpi_calls_topk.keys())
    sorted_mpi_categories = sorted(mpi_category_names,
                                   key=lambda category: avg_mpi_calls_topk.get(category, 0.0),
                                   reverse=False)
    category_order = ["Compute"] + sorted_mpi_categories

    # 3. Plotting
    x_labels = ["Average", "Best", "Q1", "Q2", "Q3", "Worst"]
    x_positions = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bar_width = 0.8 # Define bar width for positioning annotations

    bottoms = np.zeros(len(x_labels))
    used_labels = set()

    # Store heights for the Average bar to draw braces later
    avg_bar_segment_heights = {}

    for category in category_order:
        heights = np.array([plot_data[name].get(category, 0.0) for name in x_labels])
        style = COLOR_HATCH_MAP.get(category, {"color":"#CCCCCC", "hatch":""})
        color = style["color"]
        hatch = style["hatch"]
        label = category if category not in used_labels else None
        if label:
             used_labels.add(category)

        bars = ax.bar(x_positions, heights, width=bar_width, bottom=bottoms,
                      color=color, hatch=hatch, edgecolor="black", label=label)

        # Store the height of this segment for the Average bar (index 0)
        if x_positions[0] == 0: # Check if it's the first bar (Average)
             avg_bar_segment_heights[category] = heights[0]

        bottoms += heights

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontproperties=font_prop)
    y_labels = [f"{y:.0f}" for y in ax.get_yticks()]
    ax.set_yticklabels(y_labels, fontproperties=font_prop)
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    
    ax.set_ylabel("Time (s)", fontproperties=font_prop)
    ax.set_title("Breakdown of " + APP + " runtime (" + NAME + ")", fontproperties=font_prop)

    # 4. Add Braces to the Worst Bar (at x_positions[-1])
    worst_bar_x = x_positions[-1] # Changed from x_positions[0] to x_positions[-1]
    brace_x_offset = bar_width * 0.7 # Changed offset to be positive for right side
    text_x_offset = bar_width * 0.9  # Changed offset to be positive for right side
    cap_size = 0.1 # Use a fixed cap size in x-axis data coordinates

    # Calculate heights for braces using "Worst" data
    compute_height = plot_data["Worst"].get("Compute", 0.0) # Changed from "Average"
    total_height = sum(plot_data["Worst"].values())      # Changed from "Average"
    mpi_start_y = compute_height
    mpi_end_y = total_height

    # --- Brace for Compute ---
    if compute_height > 0: # Only draw if compute time exists
        brace_compute_x = worst_bar_x + brace_x_offset # Changed from avg_bar_x
        text_compute_x = worst_bar_x + text_x_offset   # Changed from avg_bar_x
        # Vertical line
        ax.plot([brace_compute_x, brace_compute_x], [0, compute_height], color='black', lw=1)
        # Caps
        ax.plot([brace_compute_x - cap_size, brace_compute_x + cap_size], [0, 0], color='black', lw=1)
        ax.plot([brace_compute_x - cap_size, brace_compute_x + cap_size], [compute_height, compute_height], color='black', lw=1)
        # Text - Adjust alignment to left
        ax.text(text_compute_x, compute_height / 2, 'Compute', ha='left', va='center', fontproperties=font_prop) # Changed ha='right' to ha='left'

    # --- Brace for MPI ---
    if mpi_end_y > mpi_start_y: # Only draw if MPI time exists
        brace_mpi_x = worst_bar_x + brace_x_offset # Changed from avg_bar_x
        text_mpi_x = worst_bar_x + text_x_offset   # Changed from avg_bar_x
        # Vertical line
        ax.plot([brace_mpi_x, brace_mpi_x], [mpi_start_y, mpi_end_y], color='black', lw=1)
        # Caps
        ax.plot([brace_mpi_x - cap_size, brace_mpi_x + cap_size], [mpi_start_y, mpi_start_y], color='black', lw=1)
        ax.plot([brace_mpi_x - cap_size, brace_mpi_x + cap_size], [mpi_end_y, mpi_end_y], color='black', lw=1)
        # Text - Adjust alignment to left
        ax.text(text_mpi_x, (mpi_start_y + mpi_end_y) / 2, 'MPI', ha='left', va='center', fontproperties=font_prop) # Changed ha='right' to ha='left'

    # 5. Set Y-axis limit based on Worst run total time
    worst_total_time = sum(plot_data["Worst"].values())
    if system == 'frontier' and APP == 'AMG2023':
        ax.set_ylim(bottom=0, top=600)
        ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
        ax.set_yticklabels([0, 100, 200, 300, 400, 500, 600])
    elif system == 'frontier' and APP == 'MILC':
        ax.set_ylim(bottom=0, top=600)
        ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
        ax.set_yticklabels([0, 100, 200, 300, 400, 500, 600])
    elif system == 'perlmutter' and APP == 'AMG2023':
        ax.set_ylim(bottom=0, top=600)
        ax.set_yticks([0, 100, 200, 300, 400, 500, 600])
        ax.set_yticklabels([0, 100, 200, 300, 400, 500, 600])
    elif system == 'perlmutter' and APP == 'MILC':
        ax.set_ylim(bottom=0, top=300)
        ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
        ax.set_yticklabels([0, 50, 100, 150, 200, 250, 300])
    else:
        ax.set_ylim(bottom=0, top=worst_total_time * 1.4) # Set top limit to 1.2x Worst time

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    # Place legend below the plot, horizontally, in two columns, without a frame
    ax.legend(handles[::-1], labels[::-1], ncol=3, prop=font_prop,
              columnspacing=0.2, handletextpad=0.2, frameon=False,
              labelspacing=0.3, ) # Changed 'rowspacing' to 'labelspacing'
    # ax.tick_params(axis='x')
    # ax.tick_params(axis='y', labelsize=12)

    # Adjust layout to make space for legend and braces
    # First, use tight_layout to get a reasonable starting point
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust rect bottom value (second number) if legend overlaps x-axis label

    # Then, manually adjust subplot parameters to minimize margins
    # These values might need fine-tuning depending on label lengths, title, legend size etc.
    # Increase 'bottom' slightly to ensure legend doesn't get cut off.
    # Decrease 'right' slightly to ensure braces text doesn't get cut off.
    plt.subplots_adjust(left=0.13, bottom=0.09, right=0.87, top=0.93)

    # Update filename to reflect the new sorting
    # plt.savefig(os.path.join(save_dir, f"{APP}_mpi_compute_stacked_6.png")) # Added _braces
    plt.savefig(os.path.join(save_dir, f"{APP}_mpi_compute_stacked_6_" + system + ".pdf")) # Added _braces


##############################################################################
# 6) Plot distribution histograms for two specified MPI routines (New Function)
##############################################################################
def plot_routine_distribution(runs, routine1_name, range_bins1, routine2_name, range_bins2, num_bins=5):
    """
    Plots distribution histograms for two specified MPI routines based on custom bin ranges.

    Args:
        runs (list): List of tuples (run_name, run_data), where run_data contains 'mpi_calls'.
        routine1_name (str): Name of the first MPI routine.
        range_bins1 (list): List of 5 floats defining the start points for the first 4 bins
                             and the start of the last bin for routine 1.
        routine2_name (str): Name of the second MPI routine.
        range_bins2 (list): List of 5 floats defining the start points for the first 4 bins
                             and the start of the last bin for routine 2.
        num_bins (int): The number of bins to create (should match len(range_bins)).
    """
    if not runs:
        print("No runs data to plot distribution.")
        return
    if len(range_bins1) != num_bins or len(range_bins2) != num_bins:
        print(f"Error: range_bins arrays must have length {num_bins}.")
        return

    # --- Data Extraction ---
    runtimes1 = [run_data['mpi_calls'].get(routine1_name, 0.0) for _, run_data in runs]
    runtimes2 = [run_data['mpi_calls'].get(routine2_name, 0.0) for _, run_data in runs]

    # Filter out zero times for distribution calculation
    valid_runtimes1 = [t for t in runtimes1 if t > 1e-9] # Use small threshold for float comparison
    valid_runtimes2 = [t for t in runtimes2 if t > 1e-9]

    # --- Plotting Setup ---
    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharey=True) # Share Y axis for easier comparison
    plot_details = [
        {"ax": axs[0], "name": routine1_name, "runtimes": valid_runtimes1, "bins": range_bins1, "color": "skyblue"},
        {"ax": axs[1], "name": routine2_name, "runtimes": valid_runtimes2, "bins": range_bins2, "color": "lightcoral"}
    ]

    # --- Process and Plot Each Routine ---
    for details in plot_details:
        ax = details["ax"]
        name = details["name"]
        runtimes = details["runtimes"]
        range_bins = details["bins"]
        color = details["color"]

        min_time = min(runtimes)
        max_time = max(runtimes)
        print(f"{name} Runtime range: Min={min_time:.2f}s, Max={max_time:.2f}s")

        # Ensure max_time is at least the start of the last bin
        effective_max_time = max(max_time, range_bins[num_bins-1])
        if effective_max_time <= range_bins[num_bins-1]:
             effective_max_time += 0.1 # Add small delta if max is exactly the last bin start

        # --- Manual Bin Counting ---
        counts = [0] * num_bins
        for time in runtimes:
            if time < range_bins[0]:
                continue # Skip times below the first bin start
            elif time < range_bins[1]:
                counts[0] += 1
            elif time < range_bins[2]:
                counts[1] += 1
            elif time < range_bins[3]:
                counts[2] += 1
            elif time < range_bins[4]:
                counts[3] += 1
            # Last bin includes times >= range_bins[4] up to effective_max_time
            elif time <= effective_max_time:
                 counts[4] += 1

        # --- Generate Categorical Labels ---
        bin_labels = []
        for i in range(num_bins - 1):
            bin_labels.append(f"{range_bins[i]:.1f}-{range_bins[i+1]:.1f}")
        # Label for the last bin
        bin_labels.append(f"{range_bins[num_bins-1]:.1f}-{effective_max_time:.1f}")

        # --- Plotting Bars ---
        x_pos = np.arange(num_bins)
        # Add label=name to the bar plot for the legend
        bars = ax.bar(x_pos, counts, align='center', color=color, edgecolor='black', label=name)

        ax.set_ylabel("Number of Runs", fontproperties=font_dis_prop)
        ax.set_xlabel("Time Range (s)", fontproperties=font_dis_prop)
        ax.legend(prop=font_dis_prop)
        # ax.set_title(f"Distribution of MPI {name} Time", fontproperties=font_prop) # Title is handled by fig.suptitle
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels)
        if system == 'frontier' and (APP == 'AMG2023' or APP == "MILC"):
            ax.tick_params(axis='x', labelsize=15, rotation=15)
        else:
            ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=20)

        # Add count labels above bars
        max_count = max(counts) if counts else 1
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + max_count * 0.01,
                        f'{height}', ha='center', va='bottom', fontproperties=font_dis_prop)

    # --- Final adjustments ---
    fig.suptitle(f"Distribution of Top 2 most variable routines", fontproperties=font_dis_prop)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap

    # Update filename
    filename_base = f"{APP}_routine_distribution_{routine1_name}_{routine2_name}_categorical_" + system
    # plt.savefig(os.path.join(save_dir, filename_base + ".png"))
    plt.savefig(os.path.join(save_dir, filename_base + ".pdf"))
    print(f"Saved routine distribution plot ({routine1_name} & {routine2_name}, categorical bins) to {save_dir}")

def load_runs_from_csv(filepath):
    """
    Loads runs data from a CSV file created by save_runs_to_csv
    and reconstructs the original list of tuples format.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        list: List of tuples (run_name, run_data) or empty list if file not found/error.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
        return []
    print(df)
    runs = []
    for index, row in df.iterrows():
        run_name = row['run_name']
        app_time = row['app_time']
        total_mpi_time = row['total_mpi_time']
        mpi_calls = {}
        for col_name, value in row.items():
            if col_name.startswith('mpi_') and value > 0: # Only store non-zero values
                original_call_name = col_name[4:] # Remove 'mpi_' prefix
                mpi_calls[original_call_name] = value

        run_data = {
            'mpi_calls': mpi_calls,
            'app_time': app_time,
            'total_mpi_time': total_mpi_time
        }
        runs.append((run_name, run_data))

    print(f"Loaded {len(runs)} runs from {filepath}")
    return runs


##############################################################################
# 8) Main function: demonstration (renumbered from 7)
##############################################################################
def main():

    csv_path = os.path.join(save_dir, system, "breakdown", f"{APP}.csv")
    runs_sorted_by_apptime = load_runs_from_csv(csv_path)
    print(runs_sorted_by_apptime)

    # --- Continue with analysis using runs_sorted_by_apptime ---
    print(f"\nProcessing {len(runs_sorted_by_apptime)} runs...")

    # 8.1) Calculate best/avg/worst using the sorted list
    best_run_name, best_data, avg_data, worst_run_name, worst_data = compute_best_avg_worst(runs_sorted_by_apptime)

    # Calculate quartiles using the sorted list
    q1_run_name, q1_data, q2_run_name, q2_data, q3_run_name, q3_data = compute_quartiles(runs_sorted_by_apptime)

    # print("Best Run Data:", best_data) # Optional: print details
    # print("Worst Run Data:", worst_data) # Optional: print details

    # 8.2) Plot six stacked bars (Compute + Top k MPI)
    if APP == 'AMG2023':
        k = 5
    elif APP == 'MILC':
        k = 2
    plot_six_stacked_bars(
        avg_data, best_data, q1_data, q2_data, q3_data, worst_data, k=k
    )

    # 8.3) Plot routine distribution using the new function (Renumbered from 8.7)
    #      Pass the full sorted runs list
    # print("\nPlotting routine distribution...")

    # # Define bin edges (5 values define 4 bins + the last 'overflow' bin)
    # # Example: [21, 23, 25, 27, 29] means bins: [21-23), [23-25), [25-27), [27-29), [29-max]
    
    
    # if system == 'perlmutter':
    #     if APP == 'AMG2023':
    #         # Bins are in milliseconds
    #         range_bins1 = [21, 23, 25, 27, 29] # For Allreduce
    #         range_bins2 = [91, 99, 107, 115, 123] # For Waitall
    #         routine1 = "Allreduce"
    #         routine2 = "Waitall"
    #     elif APP == 'MILC':
    #         range_bins1 = [23, 27, 31, 35, 39] # For Allreduce
    #         range_bins2 = [80, 82, 84, 86, 88] # For Test
    #         routine1 = "Allreduce"
    #         routine2 = "Test"
    # elif system == 'frontier':
    #     if APP == 'AMG2023':
    #         # Bins are in milliseconds
    #         range_bins1 = [28, 30, 30.5, 31, 31.5] # For Allreduce
    #         range_bins2 = [88, 93, 95, 97, 100] # For Waitall
    #         routine1 = "Allreduce"
    #         routine2 = "Waitall"
    #     elif APP == 'MILC':
    #         range_bins1 = [17, 28, 36, 98, 115] # For Allreduce
    #         range_bins2 = [97, 104, 134, 139, 150] # For Test
    #         routine1 = "Allreduce"
    #         routine2 = "Test"

    # plot_routine_distribution(runs_sorted_by_apptime, routine1, range_bins1, routine2, range_bins2, num_bins=5)

if __name__ == "__main__":
    main()
