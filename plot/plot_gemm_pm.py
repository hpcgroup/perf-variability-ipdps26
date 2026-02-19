import os
import re
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.font_manager as font_manager
import pandas as pd
import seaborn as sns

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=14)
colors = ['#D55E00', '#0072B2']
markers = ['o', 'o']
# 'orange': '#D55E00', 
# 'green': '#009E73',
# 'blue': '#0072B2', 
# 'purple': '#CC79A7', 
# 'black': '#000000', 
# 'red': '#E03A3D',
# 'yellow': '#F0E442',
# Data storage: each element is (node_id, gpu_id, time)
data = []
all_times = [] # Store all times to find the global minimum

data_csv = os.path.join(os.getcwd(), "perlmutter", "gemm", "gemm_data.csv")
with open(data_csv, 'r') as f:
    reader = csv.reader(f)
    tag = 0
    for row in reader:
        if tag == 0:
            tag = 1
            continue
        time_val = float(row[2])
        data.append((row[0], row[1], time_val))
        all_times.append(time_val) # Collect all times

# max_time = np.max(all_times)
# print(max_time)
# all_times = [t for t in all_times if t < max_time]

# --- First Plot: All points normalized by GLOBAL minimum time ---

# 1. Find the global minimum time across all measurements
global_min_time = np.min(all_times) if all_times else float('inf')

# 2. Create a sorted list of unique GPU identifiers for the x-axis
node_gpu_keys = sorted(list(set(f"{node}_GPU{gpu}" for node, gpu, t in data)))
node_to_index = {key: i for i, key in enumerate(node_gpu_keys)}

# Group data by node_gpu key for easier iteration
data_by_key = {}
for node, gpu, t in data:
    key = f"{node}_GPU{gpu}"
    data_by_key.setdefault(key, []).append((node, gpu, t))

# 3. Plotting (Refactored)
plt.figure(figsize=(5, 3))
blue_scatter = None
red_scatter = None

if global_min_time == float('inf') or global_min_time <= 0:
    print("Warning: Could not find a valid global minimum time. Skipping plot.")
else:
    for i, key in enumerate(node_gpu_keys):
        if key in data_by_key:
            plot_y = []
            color = None
            marker = None
            # Process all points for this specific GPU key
            for node, gpu, t in data_by_key[key]:
                normalized_time = t / global_min_time # Normalize by global minimum
                plot_y.append(normalized_time)

                # Determine color and marker (only need to do this once per key)
                if color is None:
                    match = re.search(r'nid0*(\d+)', node)
                    is_80gb = match and int(match.group(1)) >= 8000
                    color = colors[0] if is_80gb else colors[1]
                    marker = markers[0] if is_80gb else markers[1]

            # Plot points for this GPU key
            if plot_y and color is not None:
                x_vals = [i] * len(plot_y)
                scatter = plt.scatter(x_vals, plot_y, s=8, facecolors='none', edgecolors=color, marker=marker, linewidth=1)

                # Store scatter objects for legend (only the first instance of each type)
                if color == colors[0] and red_scatter is None: # 80GB - Orange, Marker 'o'
                    red_scatter = scatter
                elif color == colors[1] and blue_scatter is None: # 40GB - Blue, Marker 's'
                    blue_scatter = scatter

    # Add legend using hollow markers (similar to other plots)
    # if blue_scatter and red_scatter:
    #     legend_elements = [
    #         plt.Line2D([0], [0], marker=markers[1], color='w', label='A100 40GB', markerfacecolor='none', markeredgecolor=colors[1], markersize=8, linewidth=0), # Blue 's'
    #         plt.Line2D([0], [0], marker=markers[0], color='w', label='A100 80GB', markerfacecolor='none', markeredgecolor=colors[0], markersize=8, linewidth=0)  # Orange 'o'
    #     ]
    #     plt.legend(handles=legend_elements, prop=font_prop, loc='upper right', frameon=False)
    # elif blue_scatter: # Only 40GB found
    #      legend_elements = [plt.Line2D([0], [0], marker=markers[1], color='w', label='A100 40GB', markerfacecolor='none', markeredgecolor=colors[1], markersize=8, linewidth=0)]
    #      plt.legend(handles=legend_elements, prop=font_prop, frameon=False)
    # elif red_scatter: # Only 80GB found
    #      legend_elements = [plt.Line2D([0], [0], marker=markers[0], color='w', label='A100 80GB', markerfacecolor='none', markeredgecolor=colors[0], markersize=8, linewidth=0)]
    #      plt.legend(handles=legend_elements, prop=font_prop, frameon=False)


    plt.xlabel("Global GPU Index", fontproperties=font_prop)
    plt.ylabel("Relative GEMM Performance", fontproperties=font_prop) # Updated Y-label
    plt.title("Relative to best time for system (Perlmutter)", fontproperties=font_prop) # Updated Title
    # Set x-ticks to be invisible or minimal if desired, as they represent indices
    # plt.xticks([]) # Option 1: Remove x-ticks entirely
    # Option 2: Show fewer ticks if needed, e.g., every 50 GPUs
    plt.xticks(fontproperties=font_prop) # Show index every 50 GPUs

    ytick_positions = np.arange(1.0, 1.3, 0.05)
    ytick_labels = [f"{y:.2f}" for y in ytick_positions] # Use the calculated positions for labels
    plt.yticks(ytick_positions, ytick_labels, fontproperties=font_prop)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("GEMM_system_pm.png", bbox_inches='tight', dpi=300) # New filename
    # plt.savefig("GEMM_all_points_normalized_by_global_min_pm.pdf")
    # plt.show()

# --- Keep the rest of the script for the other plots ---

node_gpu_times = {}
for node, gpu, t in data:
    key = f"{node}_GPU{gpu}"  # Example: "nid001234_GPU0"
    node_gpu_times.setdefault(key, []).append(t)

plt.figure(figsize=(5, 3))  # Widen the figure to accommodate more labels
nodes_gpus = sorted(node_gpu_times.keys())  # Sort all node_gpu combinations

greater_than_25 = 0
total = 0
blue_scatter = None
red_scatter = None

for i, node_gpu in enumerate(nodes_gpus):
    times = node_gpu_times[node_gpu]
    min_time = min(times)
    # Calculate relative performance
    relative_times = [t / min_time for t in times]
    # x coordinate is index i
    greater_than_25 += sum(1 for t in relative_times if t > 1.025)
    total += len(relative_times)
    x_vals = [i] * len(relative_times)
    
    # Determine color based on node ID
    match = re.search(r'nid0*(\d+)', node_gpu)
    color = colors[0] if match and int(match.group(1)) > 8000 else colors[1]
    marker = markers[0] if match and int(match.group(1)) > 8000 else markers[1]
    # Use facecolors='none' and specify edgecolors for hollow markers
    scatter = plt.scatter(x_vals, relative_times, s=8, facecolors='none', edgecolors=color, marker=marker, linewidth=1)
    
    # Store scatter objects for legend
    if color == colors[0] and red_scatter is None:
        red_scatter = scatter
    elif color == colors[1] and blue_scatter is None:
        blue_scatter = scatter

# Add legend using hollow markers
if blue_scatter and red_scatter:
    legend_elements = [
        plt.Line2D([0], [0], marker=markers[1], color='w', label='A100 40GB', markerfacecolor='none', markeredgecolor=colors[1], markersize=8, linewidth=0),
        plt.Line2D([0], [0], marker=markers[0], color='w', label='A100 80GB', markerfacecolor='none', markeredgecolor=colors[0], markersize=8, linewidth=0)
    ]
    plt.legend(handles=legend_elements, prop=font_prop, loc='upper right', frameon=False)
elif blue_scatter:
    legend_elements = [plt.Line2D([0], [0], marker=markers[1], color='w', label='A100 40GB', markerfacecolor='none', markeredgecolor=colors[1], markersize=8, linewidth=0)]
    plt.legend(handles=legend_elements, prop=font_prop, frameon=False)
elif red_scatter:
    legend_elements = [plt.Line2D([0], [0], marker=markers[0], color='w', label='A100 80GB', markerfacecolor='none', markeredgecolor=colors[0], markersize=8, linewidth=0)]
    plt.legend(handles=legend_elements, prop=font_prop, frameon=False)

plt.xlabel("Global GPU Index", fontproperties=font_prop)
plt.ylabel("Relative GEMM Performance", fontproperties=font_prop)
plt.title("Relative to best time per GPU (Perlmutter)", fontproperties=font_prop)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines for better visibility
plt.xticks(fontproperties=font_prop)
ytick_positions = np.arange(1.0, 1.3, 0.05)
ytick_labels = [f"{y:.2f}" for y in np.arange(1.0, 1.3, 0.05)]
plt.yticks(ytick_positions, ytick_labels, fontproperties=font_prop)
plt.tight_layout()
plt.savefig("GEMM_per_gpu_pm.png", bbox_inches='tight', dpi=300)  # Ensure labels are fully displayed
# plt.savefig("GEMM_per_gpu_pm.pdf", bbox_inches='tight')  # Ensure labels are fully displayed


# ------------------------------------------------
# New Plot: Relative GEMM Performance per GPU (normalized by node minimum)
# ------------------------------------------------

# 1. Group all times by node ID
node_all_times = {}
for node, gpu, t in data:
    node_all_times.setdefault(node, []).append(t)

# 2. Calculate the minimum time for each node
node_min_times = {node: min(times) for node, times in node_all_times.items() if times}

# 3. Calculate relative times for each GPU measurement, normalized by its node's minimum
node_gpu_relative_node_min_times = {}
for node, gpu, t in data:
    key = f"{node}_GPU{gpu}"
    if node in node_min_times: # Ensure the node has data
        node_min = node_min_times[node]
        if node_min > 0: # Avoid division by zero if min time is 0
             relative_time = t / node_min
             node_gpu_relative_node_min_times.setdefault(key, []).append(relative_time)
        else:
             # Handle cases where node_min is 0 if necessary, e.g., assign a specific value or skip
             node_gpu_relative_node_min_times.setdefault(key, []).append(1.0) # Or np.nan, or skip

# 4. Plotting (adapted from GEMM_per_gpu_pm plot)
plt.figure(figsize=(5, 3))
nodes_gpus = sorted(node_gpu_relative_node_min_times.keys()) # Use keys from the new relative times dict

blue_scatter = None
red_scatter = None

for i, node_gpu in enumerate(nodes_gpus):
    # Retrieve the pre-calculated relative times normalized by node minimum
    relative_times = node_gpu_relative_node_min_times[node_gpu]
    x_vals = [i] * len(relative_times)

    # Determine color based on node ID (same logic as before)
    match = re.search(r'nid0*(\d+)', node_gpu)
    color = colors[0] if match and int(match.group(1)) >= 8000 else colors[1] # Adjusted condition >= 8000
    marker = markers[0] if match and int(match.group(1)) >= 8000 else markers[1]
    # Use facecolors='none' and specify edgecolors for hollow markers
    scatter = plt.scatter(x_vals, relative_times, s=8, facecolors='none', edgecolors=color, marker=marker, linewidth=1)

    # Store scatter objects for legend
    if color == colors[0] and red_scatter is None:
        red_scatter = scatter
    elif color == colors[1] and blue_scatter is None:
        blue_scatter = scatter

# Add legend using hollow markers
# if blue_scatter and red_scatter:
#     legend_elements = [
#         plt.Line2D([0], [0], marker=markers[1], color='w', label='A100 40GB', markerfacecolor='none', markeredgecolor=colors[1], markersize=8, linewidth=0),
#         plt.Line2D([0], [0], marker=markers[0], color='w', label='A100 80GB', markerfacecolor='none', markeredgecolor=colors[0], markersize=8, linewidth=0)
#     ]
#     plt.legend(handles=legend_elements, prop=font_prop, loc='upper right', frameon=False)
# elif blue_scatter:
#     legend_elements = [plt.Line2D([0], [0], marker=markers[1], color='w', label='A100 40GB', markerfacecolor='none', markeredgecolor=colors[1], markersize=8, linewidth=0)]
#     plt.legend(handles=legend_elements, prop=font_prop, frameon=False)
# elif red_scatter:
#     legend_elements = [plt.Line2D([0], [0], marker=markers[0], color='w', label='A100 80GB', markerfacecolor='none', markeredgecolor=colors[0], markersize=8, linewidth=0)]
#     plt.legend(handles=legend_elements, prop=font_prop, frameon=False)

plt.xlabel("Global GPU Index", fontproperties=font_prop)
plt.ylabel("Relative GEMM Performance", fontproperties=font_prop) # Updated Y-axis label
plt.title("Relative to best time per node (Perlmutter)", fontproperties=font_prop) # Updated Title
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontproperties=font_prop)
ytick_positions = np.arange(1.0, 1.3, 0.05)
ytick_labels = [f"{y:.2f}" for y in np.arange(1.0, 1.3, 0.05)]
plt.yticks(ytick_positions, ytick_labels, fontproperties=font_prop)
plt.tight_layout()
plt.savefig("GEMM_per_node_pm.png", bbox_inches='tight', dpi=300) # New filename
# plt.savefig("GEMM_per_gpu_rel_node_min_pm.pdf", bbox_inches='tight') # New filename

