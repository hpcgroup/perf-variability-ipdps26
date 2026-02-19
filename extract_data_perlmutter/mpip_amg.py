#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
app_dir = "AMG2023/64nodes/"
# app_dir = "MILC/64nodes/"
trace_dir = "/pscratch/sd/c/cunyang/result/" + app_dir
save_dir = "/pscratch/sd/c/cunyang/result/ana/" + app_dir

##############################################################################
# 1) Parse mpiP file to get {MPI_Call: time_in_seconds_per_gpu, ...}
##############################################################################
def parse_mpip_time(file_path):
    """
    Parse the "Callsite Time statistics (all, milliseconds)" section from mpiP report,
    only count lines where rank == '*', accumulate (Count * Mean) for each MPI call,
    convert units from ms to seconds, and divide by 256 (e.g., for 256 GPUs).
    Returns { "Allreduce": 2.13, "Waitall": 1.09, ... } (units: s, already /256).
    """
    import re
    pattern_start   = re.compile(r"Callsite Time statistics \(all, milliseconds\):")
    pattern_dashes  = re.compile(r"^-{3,}$")

    in_section    = False
    dash_count    = 0
    time_dict_s   = {}

    with open(file_path, 'r') as f:
        for line in f:
            line_stripped = line.strip()

            if not in_section:
                if pattern_start.search(line_stripped):
                    in_section = True
                continue
            else:
                if pattern_dashes.search(line_stripped):
                    dash_count += 1
                    # The first dashed line is the header, the second dashed line marks the end
                    if dash_count == 2:
                        break
                    continue

                fields = line_stripped.split()
                if len(fields) < 9:
                    continue

                name  = fields[0]
                rank  = fields[2]
                c_str = fields[3]
                m_str = fields[5]

                # Only count lines where rank == '*'
                if rank != '*':
                    continue

                try:
                    count_val = float(c_str)
                    mean_val  = float(m_str)
                except ValueError:
                    continue

                total_time_s = (count_val * mean_val) / 1000.0  # ms -> s
                total_time_s /= 256.0  # Divide by 256 (e.g., 256 GPUs)
                time_dict_s[name] = time_dict_s.get(name, 0.0) + total_time_s

    return time_dict_s


##############################################################################
# 2) Traverse directories to collect all runs, returning [(run_name, {mpi_calls: {call: time_s, ...}, app_time: T1, mpi_time: T2}), ...]
##############################################################################
def collect_runs(trace_dir):
    job_dirs = sorted(
        [d for d in os.listdir(trace_dir)
         if os.path.isdir(os.path.join(trace_dir, d))
         and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-job\d+", d)],
        key=lambda x: x[:19]
    )
################################################################################
    job_dirs = [d for d in job_dirs if d[:10] >= "2025-02-08"]
################################################################################
    runs = []
    for d in job_dirs:
        subdir_path = os.path.join(trace_dir, d)
        mpi_files = [f for f in os.listdir(subdir_path) if re.match(r".*\.1\.mpiP$", f)]
        if not mpi_files:
            continue
        mpip_path = os.path.join(subdir_path, mpi_files[0])

        # Parse MPI call times
        mpi_call_dict = parse_mpip_time(mpip_path)

        # Parse App and MPI times from '*' line
        apptime_val, mpitime_val = parse_app_mpi_line(mpip_path)

        if apptime_val is None or mpitime_val is None:
            print(f"Warning: Could not parse App/MPI times from {mpip_path}. Skipping run {d}.")
            continue

        # Store data per run, dividing by 256 here
        run_data = {
            'mpi_calls': mpi_call_dict,
            'app_time': apptime_val / 256.0,
            'total_mpi_time': mpitime_val / 256.0 # Note: This mpi_time might differ slightly from sum(mpi_call_dict.values())
        }
        runs.append((d, run_data))

    return runs


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
COLOR_HATCH_MAP = {
    "Compute":   {"color": "#A6CEE3", "hatch": ""},     # Light Blue, no hatch (Example)
    "Waitall":   {"color": "#FFFF99", "hatch": "xx"},   # Light yellow, cross hatch
    "Test":      {"color": "#FFA500", "hatch": "//"},   # Orange, diagonal lines
    "Iprobe":    {"color": "#66C2A5", "hatch": ".."},   # Light cyan, dots
    "Allreduce": {"color": "#FFA07A", "hatch": "\\\\"}, # Light orange-red, backslashes
    "Testall":   {"color": "#EE82EE", "hatch": "**"},   # Pink-purple, stars
    "Isend":     {"color": "#B2DF8A", "hatch": "++"},   # Light green, plus signs
    "Irecv":     {"color": "#1F78B4", "hatch": "oo"},   # Blue, small circles
    "Others":    {"color": "#FFFFFF", "hatch": ""}      # White, no hatch
}

def parse_app_mpi_line(file_path):
    """
    Parse lines starting with '*' (e.g., line 146) from mpip report,
    format similar to:
        *   2.85e+04   1.89e+04    66.38
    representing: Task, AppTime, MPITime, MPI%
    This function returns (apptime, mpitime), or (None, None) if not found.
    """
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0] == '*':
                try:
                    app_time = float(fields[1])
                    mpi_time = float(fields[2])
                    return (app_time, mpi_time)
                except ValueError:
                    pass
    return (None, None)


def collect_app_mpi_data(trace_dir):
    """
    Similar to collect_runs(), traverse the same directories, find matching directories and files with mpip files.
    For each mpip report, call parse_app_mpi_line(),
    and put apptime/256 and mpitime/256 into two arrays respectively and return.
    """
    import re
    app_times = []
    mpi_times = []

    # Similar directory traversal as collect_runs()
    job_dirs = sorted(
        [d for d in os.listdir(trace_dir)
         if os.path.isdir(os.path.join(trace_dir, d))
         and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-job\d+", d)],
        key=lambda x: x[:19]
    )
    
    for d in job_dirs:
        subdir_path = os.path.join(trace_dir, d)
        # Similar file name matching as collect_runs(): milc.256.*.1.mpiP
        mpi_files = [f for f in os.listdir(subdir_path) if re.match(r".*\.1\.mpiP$", f)]
        if not mpi_files:
            continue
        mpip_path = os.path.join(subdir_path, mpi_files[0])

        # Parse '* 2.85e+04 1.89e+04 66.38' line
        apptime_val, mpitime_val = parse_app_mpi_line(mpip_path)
        if apptime_val is not None and mpitime_val is not None:
            # Divide by 256 (e.g., 256 GPUs)
            app_times.append(apptime_val / 256.0)
            mpi_times.append(mpitime_val / 256.0)

    return app_times, mpi_times

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
    fig, ax = plt.subplots(figsize=(8, 6))
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
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Time (s) [per GPU]")
    ax.set_title("AMG2023 64nodes Runtime Breakdown (Compute + MPI)")

    # 4. Add Braces to the Average Bar (at x_positions[0])
    avg_bar_x = x_positions[0]
    brace_x_offset = -bar_width * 0.7 # Position braces to the left of the bar
    text_x_offset = -bar_width * 0.9  # Position text further left
    cap_size = 0.05 # Size of the caps on the brace lines

    # Calculate heights for braces
    compute_height = plot_data["Average"].get("Compute", 0.0)
    total_height = sum(plot_data["Average"].values())
    mpi_start_y = compute_height
    mpi_end_y = total_height

    # --- Brace for Compute ---
    if compute_height > 0: # Only draw if compute time exists
        brace_compute_x = avg_bar_x + brace_x_offset
        text_compute_x = avg_bar_x + text_x_offset
        # Vertical line
        ax.plot([brace_compute_x, brace_compute_x], [0, compute_height], color='black', lw=1)
        # Caps
        ax.plot([brace_compute_x - cap_size, brace_compute_x + cap_size], [0, 0], color='black', lw=1)
        ax.plot([brace_compute_x - cap_size, brace_compute_x + cap_size], [compute_height, compute_height], color='black', lw=1)
        # Text
        ax.text(text_compute_x, compute_height / 2, 'Compute', ha='right', va='center', fontsize=9)

    # --- Brace for MPI ---
    if mpi_end_y > mpi_start_y: # Only draw if MPI time exists
        brace_mpi_x = avg_bar_x + brace_x_offset
        text_mpi_x = avg_bar_x + text_x_offset
        # Vertical line
        ax.plot([brace_mpi_x, brace_mpi_x], [mpi_start_y, mpi_end_y], color='black', lw=1)
        # Caps
        ax.plot([brace_mpi_x - cap_size, brace_mpi_x + cap_size], [mpi_start_y, mpi_start_y], color='black', lw=1)
        ax.plot([brace_mpi_x - cap_size, brace_mpi_x + cap_size], [mpi_end_y, mpi_end_y], color='black', lw=1)
        # Text
        ax.text(text_mpi_x, (mpi_start_y + mpi_end_y) / 2, 'MPI', ha='right', va='center', fontsize=9)


    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1)) # Adjusted anchor slightly

    # Adjust layout to make space for legend and braces
    plt.subplots_adjust(left=0.15, right=0.8) # Adjust left/right margins
    # plt.tight_layout(rect=[0.1, 0, 0.85, 1]) # Alternative adjustment

    # Update filename to reflect the new sorting
    plt.savefig(save_dir + "amg_mpi_compute_stacked_6_comp_first_asc_braces.png") # Added _braces
    plt.savefig(save_dir + "amg_mpi_compute_stacked_6_comp_first_asc_braces.pdf") # Added _braces
    print(f"Saved Compute+MPI stacked bar plot (Compute first, MPI asc avg time, with braces) to {save_dir}")


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
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharey=True) # Share Y axis for easier comparison
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

        if not runtimes:
            print(f"No positive {name} times found, cannot plot distribution.")
            ax.set_title(f"Distribution of MPI {name} Time (No Data)")
            ax.set_xlabel("Time Range (s)")
            ax.set_ylabel("Number of Runs")
            ax.set_xticks(range(num_bins))
            ax.set_xticklabels([f"Bin {i+1}" for i in range(num_bins)]) # Placeholder labels
            continue

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
        bars = ax.bar(x_pos, counts, align='center', color=color, edgecolor='black')

        ax.set_ylabel("Number of Runs")
        ax.set_xlabel("Time Range (s)")
        ax.set_title(f"Distribution of MPI {name} Time")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels)
        ax.tick_params(axis='x', rotation=45)

        # Add count labels above bars
        max_count = max(counts) if counts else 1
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + max_count * 0.01,
                        f'{height}', ha='center', va='bottom')

    # --- Final adjustments ---
    fig.suptitle(f"Distribution of MPI {routine1_name} and {routine2_name} Times (AMG2023)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Update filename
    filename_base = f"amg_routine_distribution_{routine1_name}_{routine2_name}_categorical"
    plt.savefig(save_dir + filename_base + ".png")
    plt.savefig(save_dir + filename_base + ".pdf")
    print(f"Saved routine distribution plot ({routine1_name} & {routine2_name}, categorical bins) to {save_dir}")


##############################################################################
# 7) Save and Load runs data to/from CSV (New Section)
##############################################################################

def save_runs_to_csv(runs, filepath):
    """
    Saves the runs data to a CSV file using pandas.
    Flattens the mpi_calls dictionary into columns prefixed with 'mpi_'.

    Args:
        runs (list): List of tuples (run_name, run_data).
        filepath (str): Path to save the CSV file.
    """
    if not runs:
        print("No runs data to save.")
        return

    all_mpi_calls = set()
    for _, run_data in runs:
        all_mpi_calls.update(run_data['mpi_calls'].keys())

    data_for_df = []
    for run_name, run_data in runs:
        row = {
            'run_name': run_name,
            'app_time': run_data['app_time'],
            'total_mpi_time': run_data['total_mpi_time']
        }
        # Add MPI call times, prefixing column names
        for call in all_mpi_calls:
            row[f"mpi_{call}"] = run_data['mpi_calls'].get(call, 0.0)
        data_for_df.append(row)

    df = pd.DataFrame(data_for_df)

    # Define column order: run_name, app_time, total_mpi_time, then sorted mpi calls
    mpi_cols_sorted = sorted([col for col in df.columns if col.startswith('mpi_')])
    column_order = ['run_name', 'app_time', 'total_mpi_time'] + mpi_cols_sorted
    df = df[column_order]

    df.to_csv(filepath, index=False, float_format='%.6e') # Use scientific notation for precision
    print(f"Saved runs data to {filepath}")


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

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # --- Option 1: Collect data from trace files ---
    print("Collecting data from trace directory...")
    runs = collect_runs(trace_dir)
    if not runs:
        print("No runs found in trace directory.")
        return

    # Sort runs by total AppTime ascending - IMPORTANT for best/worst/quartiles
    runs_sorted_by_apptime = sorted(runs, key=lambda x: x[1]['app_time'])
    # print(runs_sorted_by_apptime) # Optional: print loaded data

    # Save the collected and sorted data to CSV
    csv_path = os.path.join("AMG2023.csv")
    save_runs_to_csv(runs_sorted_by_apptime, csv_path)
    
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
    exit()
    # 8.2) Plot six stacked bars (Compute + Top k MPI)
    k = 7
    plot_six_stacked_bars(
        avg_data, best_data, q1_data, q2_data, q3_data, worst_data, k=k
    )

    # 8.3) Plot routine distribution using the new function (Renumbered from 8.7)
    #      Pass the full sorted runs list
    print("\nPlotting routine distribution...")

    # Define bin edges (5 values define 4 bins + the last 'overflow' bin)
    # Example: [21, 23, 25, 27, 29] means bins: [21-23), [23-25), [25-27), [27-29), [29-max]
    range_bins1 = [21, 23, 25, 27, 29] # For Allreduce
    range_bins2 = [91, 99, 107, 115, 123] # For Waitall
    # plot_routine_distribution(runs_sorted_by_apptime, "Allreduce", range_bins1, "Waitall", range_bins2, num_bins=5)

if __name__ == "__main__":
    main()

