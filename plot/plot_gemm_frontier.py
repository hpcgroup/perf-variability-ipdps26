import csv
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np

data_size = 32768

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=14)
colors = ['#D55E00', '#0072B2', '#009E73', '#000000', '#800080', '#CC79A7', '#E69F00', '#56B4E9']
markers = ['o']
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(os.path.join('frontier', 'gemm', "gemm_data.csv"))
data = df.values.tolist()

# 1. Create node times and node gpus times structures
node_gpus = set()
node_times = {}
node_gpu_times = {}
for node, gpu, t in data:
    key = f"{node}_GPU{gpu}"  # Example: "nid001234_GPU0"
    node_gpu_times.setdefault(key, []).append(t)
    node_times.setdefault(node, []).append(t)
    node_gpus.add(key)
node_gpus = sorted(list(node_gpus))
def parse_node(node_gpu): return node_gpu.split('_')[0]

# 2. Calculate min time (per gpu, per node, for system)
# 2a. per gpu
min_node_gpu_times = {}
for node_gpu in node_gpu_times:
    min_node_gpu_times[node_gpu] = min(node_gpu_times[node_gpu])
# 2b. per node
min_node_times = {}
for node in node_times:
    min_node_times[node] = min(node_times[node])
# 2c. for system
min_system_time = min(min_node_times.values())

# Initializaiton of structures for for loop
x_vals = []
relative_times_gpu = []
relative_times_node = []
relative_times_system = []
total_gpus = 0
total_datapoints = 0
greater_than_25_gpu = 0
greater_than_25_node = 0
greater_than_25_system = 0

for i, node_gpu in enumerate(node_gpus):
    print(f"Processing: {i}/{len(node_gpus)}")
    node = parse_node(node_gpu)
    times = np.array(node_gpu_times[node_gpu])

    # x coordinate is index i
    x_vals.extend([i] * len(times))
    # Calculate gemm relative perf (to best gpu time, to best node time, to best system time)
    curr_relative_times_gpu = times / min_node_gpu_times[node_gpu]
    relative_times_gpu.extend(curr_relative_times_gpu)
    curr_relative_times_node = times / min_node_times[node]
    relative_times_node.extend(curr_relative_times_node)
    curr_relative_times_system = times / min_system_time
    relative_times_system.extend(curr_relative_times_system)

    # Calculate some statistics
    total_gpus += 1
    total_datapoints += len(times)
    greater_than_25_gpu += sum(t > 1.025 for t in curr_relative_times_gpu)
    greater_than_25_node += sum(t > 1.025 for t in curr_relative_times_node)
    greater_than_25_system += sum(t > 1.025 for t in curr_relative_times_system)

# Track maximum y values
max_y_gpu = np.max(relative_times_gpu)
max_y_node = np.max(relative_times_node)
max_y_system = np.max(relative_times_system)

# settings for all plots
xtick_positions = list(range(0, total_gpus, 20_000))
xtick_labels = [i for i in xtick_positions]  # Corresponding labels
# Use np.linspace to ensure the endpoint (1.20) is included and generate 5 points
ytick_positions = np.linspace(1.0, 1.2, 5)
# Format labels to exactly two decimal places
ytick_labels = [f"{pos:.2f}" for pos in ytick_positions]
# ------------------------------------------------
# PLOT 1 - relative to GPU
# ------------------------------------------------
# scatter plot of relative times
plt.figure(figsize=(5, 3))
plt.scatter(x_vals, relative_times_gpu, s=8, color=colors[1], marker=markers[0], facecolors='none', edgecolors=colors[1], linewidth=1)
plt.xticks(xtick_positions, xtick_labels, fontproperties=font_prop)
plt.yticks(ytick_positions, ytick_labels, fontproperties=font_prop)
plt.xlabel("Global GPU Index", fontproperties=font_prop)
plt.ylabel("Relative GEMM Performance", fontproperties=font_prop)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.title("Relative to best time per GPU (Frontier)", fontproperties=font_prop)
plt.tight_layout()
plt.savefig(os.path.join("GEMM_per_gpu_frontier.png"), dpi=300)
# plt.savefig(os.path.join("GEMM_per_gpu_frontier.pdf"), format='pdf')


# ------------------------------------------------
# PLOT 2 - relative to node
# ------------------------------------------------
# scatter plot of relative times
plt.figure(figsize=(5, 3))
plt.scatter(x_vals, relative_times_node, s=8, color=colors[1], marker=markers[0], facecolors='none', edgecolors=colors[1], linewidth=1)
plt.xticks(xtick_positions, xtick_labels, fontproperties=font_prop)
plt.yticks(ytick_positions, ytick_labels, fontproperties=font_prop)
plt.xlabel("Global GPU Index", fontproperties=font_prop)
plt.ylabel("Relative GEMM Performance", fontproperties=font_prop)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.title("Relative to best time per node (Frontier)", fontproperties=font_prop)
plt.tight_layout()
plt.savefig(os.path.join("GEMM_per_node_frontier.png"), dpi=300)
# plt.savefig(os.path.join("GEMM_per_node_frontier.pdf"), format='pdf')


# ------------------------------------------------
# PLOT 3 - relative to system
# ------------------------------------------------
# scatter plot of relative times
plt.figure(figsize=(5, 3))
plt.scatter(x_vals, relative_times_system, s=8, color=colors[1], marker=markers[0], facecolors='none', edgecolors=colors[1], linewidth=1)
plt.xticks(xtick_positions, xtick_labels, fontproperties=font_prop)
plt.yticks(ytick_positions, ytick_labels, fontproperties=font_prop)
plt.xlabel("Global GPU Index", fontproperties=font_prop)
plt.ylabel("Relative GEMM Performance", fontproperties=font_prop)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.title("Relative to best time for system (Frontier)", fontproperties=font_prop)
plt.tight_layout()
plt.savefig(os.path.join("GEMM_system_frontier.png"), dpi=300)
# plt.savefig(os.path.join("GEMM_system_frontier.pdf"), format='pdf')

# Print all the max y values for debugging or analysis
print(f"Max relative GEMM time (GPU): {max_y_gpu}")
print(f"Max relative GEMM time (Node): {max_y_node}")
print(f"Max relative GEMM time (System): {max_y_system}")
print(f"Stats: GPUs={total_gpus}, Datapoints={total_datapoints}")
print(
    f"Points >2.5% variability - GPU: {greater_than_25_gpu}, Node: {greater_than_25_node}, System: {greater_than_25_system}")
print(f"Variability % - GPU: {100*greater_than_25_gpu/total_datapoints:.2f}%, "
        f"Node: {100*greater_than_25_node/total_datapoints:.2f}%, "
        f"System: {100*greater_than_25_system/total_datapoints:.2f}%")
