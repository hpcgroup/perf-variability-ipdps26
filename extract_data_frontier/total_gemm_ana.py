import csv
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# data_size = 4096
# data_size = 8192
# data_size = 16384
data_size = 32768
skip_plots = True

# Regex: match target lines in gemm.out and capture Node ID, GPU ID and average time
line_pattern = re.compile(
    r'SLURM Node ID:\s*(\S+),\s*GPU ID:\s*(\d+).*size' + f" {data_size} " + r'average:\s*([\d\.]+)\s*s'
)

# List of top-level directories to traverse
base_dir = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar")
out_dir = base_dir
app_folders = ["AMG2023_logs", "deepcam_logs", "MILC_logs", "nanoGPT_logs"]
# List of subfolders to enter
node_counts = [16, 64]

# Data storage: each element is (node_id, gpu_id, time)
data = []

# Traverse directories
for app in app_folders:
    for nnodes in node_counts:
        node_subfolder = f"{nnodes}nodes"
        folder_patterns = {
            "AMG2023_logs": re.compile(r"amg-\d{7}"),
            "MILC_logs": re.compile(f"milc_40.{nnodes}-"+r"\d{7}"),
            "deepcam_logs": re.compile(r"deepcam-\d{7}"),
            "nanoGPT_logs": re.compile(r"nanogpt-\d{7}"),
        }
        base_path = os.path.join(base_dir, app, node_subfolder)
        if not os.path.isdir(base_path):
            continue
        job_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path)]
        # include OLD nanogpt logs which didnt have torch profiler output
        if app == "nanoGPT_logs":
            job_dirs = job_dirs + [os.path.join(base_dir, app, f"{nnodes}nodes_notorchprof", d) for d in os.listdir(os.path.join(base_dir, app, f"{nnodes}nodes_notorchprof"))]
        # include OLD nanogpt/deepcam logs which didnt have RDZV env variables
        if app == "nanoGPT_logs" or app == "deepcam_logs":
            job_dirs = job_dirs + [os.path.join(base_dir, app, f"{nnodes}nodes_no_RDZV_env", d) for d in os.listdir(os.path.join(base_dir, app, f"{nnodes}nodes_no_RDZV_env"))]
        
        # Traverse folders within subdirectory
        for entry in job_dirs:
            # If folder name matches the pattern
            if folder_patterns[app].match(os.path.basename(entry)):
                gemm_file = os.path.join(entry, "output-gemm.log")
                if not os.path.isfile(gemm_file):
                    continue
                print(gemm_file)

                with open(gemm_file, 'r') as f:
                    for line in f:
                        # Process lines containing "size 32768" only
                        if f"size {data_size}" in line:
                            m = line_pattern.search(line)
                            if m:
                                node_id = m.group(1)
                                gpu_id = m.group(2)
                                time_val = float(m.group(3))
                                data.append((node_id, gpu_id, time_val))

# node_avg = {node: np.mean(times) for node, times in node_gpu_times.items()}
# Sorted list of nodes
# nodes = sorted(node_avg.keys())
# avg_times = [node_avg[node] for node in nodes]
# min_value = np.min(avg_times)
# avg_times = [t / min_value for t in avg_times]

# plt.figure(figsize=(8,6))
# x_pos = np.arange(len(nodes))
# plt.scatter(x_pos, avg_times, s=5, color='#07519c')
# # plt.xticks(x_pos, nodes, rotation=45)
# plt.xlabel("Remapped GPU Index")
# plt.ylabel("Average GEMM Time (s)")
# plt.title("Average GEMM Time per GPU (relative to best)")
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "GEMM_avg.png"))
# plt.savefig(os.path.join(out_dir, "GEMM_avg.pdf"), format='pdf')

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
parse_node = lambda node_gpu: node_gpu.split('_')[0]

# GENERATE gemm_data.csv
# Create a DataFrame from the data
df = pd.DataFrame(data, columns=["Node", "GPU", "Time"])
# Save the DataFrame to a CSV file
df.to_csv(os.path.join(base_dir, 'gemm_data.csv'), index=False)

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

# TAKES LONG TIME
# ???????????????????????????????????
# flag to skip_plots (takes very very long compared and can be skipped if we
# just want to find slowest gpus csv)
if not skip_plots:
    print("plotting...")

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    
    total_gpus = 0
    total_datapoints = 0
    greater_than_25_gpu = 0
    greater_than_25_node = 0
    greater_than_25_system = 0

    # Find maximum y value to set appropriate upper limit
    max_y_gpu = 1.0
    max_y_node = 1.0
    max_y_system = 1.0

    for i, node_gpu in enumerate(node_gpus):
        print(i)
        node = parse_node(node_gpu)
        times = np.array(node_gpu_times[node_gpu])
        
        # Calculate gemm relative perf (to best gpu time, to best node time, to best system time)
        relative_times_gpu = times / min_node_gpu_times[node_gpu]
        relative_times_node = times / min_node_times[node]
        relative_times_system = times / min_system_time

        # Calculate some statistics
        total_gpus += 1
        total_datapoints += len(times)
        greater_than_25_gpu += sum(t > 1.025 for t in relative_times_gpu)
        greater_than_25_node += sum(t > 1.025 for t in relative_times_node)
        greater_than_25_system += sum(t > 1.025 for t in relative_times_system)

        # Track maximum y values
        max_y_gpu = max(max_y_gpu, np.max(relative_times_gpu))
        max_y_node = max(max_y_node, np.max(relative_times_node))
        max_y_system = max(max_y_system, np.max(relative_times_system))

        # x coordinate is index i
        x_vals = [i] * len(times)
        # scatter plot of relative times
        ax1.scatter(x_vals, relative_times_gpu, s=5, color='#07519c')
        ax2.scatter(x_vals, relative_times_node, s=5, color='#07519c')
        ax3.scatter(x_vals, relative_times_system, s=5, color='#07519c')

        # if i == 100:
        #     break

    tick_positions = list(range(0, total_gpus, 10_000))
    tick_labels = [i for i in tick_positions]  # Corresponding labels

    for ax in (ax1, ax2, ax3):
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0.99, 1.2)
        ax.set_yticks(np.arange(1.0, 1.21, 0.025))
        ax.set_xlabel("Remapped GPU Index")
        ax.set_ylabel("Relative GEMM Time")
        # Add horizontal dashed lines at every ytick
        for ytick in ax.get_yticks():
            ax.axhline(y=ytick, color='gray', linestyle='--', linewidth=0.5)
        # ax.figure.tight_layout()
    
    print("saving...")
    ax1.set_title("GEMM Performance (relative to best time per GPU)")
    fig1.savefig(os.path.join(out_dir, "GEMM_gpu.png"))
    fig1.savefig(os.path.join(out_dir, "GEMM_gpu.pdf"), format='pdf')

    ax2.set_title("GEMM Performance (relative to best time per node)")
    fig2.savefig(os.path.join(out_dir, "GEMM_node.png"))
    fig2.savefig(os.path.join(out_dir, "GEMM_node.pdf"), format='pdf')
    
    ax3.set_title("GEMM Performance (relative to best time for system)")
    fig3.savefig(os.path.join(out_dir, "GEMM_system.png"))
    fig3.savefig(os.path.join(out_dir, "GEMM_system.pdf"), format='pdf')
    print("done saving...")
    print("done plotting...")

    # Print all the max y values for debugging or analysis
    print(f"Max relative GEMM time (GPU): {max_y_gpu}")
    print(f"Max relative GEMM time (Node): {max_y_node}")
    print(f"Max relative GEMM time (System): {max_y_system}")
    print(f"Stats: GPUs={total_gpus}, Datapoints={total_datapoints}")
    print(f"Points >2.5% variability - GPU: {greater_than_25_gpu}, Node: {greater_than_25_node}, System: {greater_than_25_system}")
    print(f"Variability % - GPU: {100*greater_than_25_gpu/total_datapoints:.2f}%, " 
            f"Node: {100*greater_than_25_node/total_datapoints:.2f}%, "
            f"System: {100*greater_than_25_system/total_datapoints:.2f}%")
# ???????????????????????????????????


# ================================================
# FIND SLOWEST GPUs
print("finding slowest gpus")

# ------------------------------------------------
# 1. Calculate the average execution time for each GPU (nodeID+gpuID)
# ------------------------------------------------
gpu_avg_data = []  # Store (node_gpu_str, avg_time)
for node_gpu, times in node_gpu_times.items():
    avg_time = np.mean(times)
    gpu_avg_data.append((node_gpu, avg_time))

# ------------------------------------------------
# 2. Sort by average execution time from smallest to largest
#    (smaller is faster, larger is slower)
# ------------------------------------------------
gpu_avg_data.sort(key=lambda x: x[1])  # Sort in ascending order by avg_time

# ------------------------------------------------
# 3. Assign labels based on sorted position
#    First 30% -> label=0
#    Middle 30%~70% -> label=1
#    Last 30% -> label=2
# ------------------------------------------------
n = len(gpu_avg_data)
labeled_data = []
for i, (node_gpu, avg_time) in enumerate(gpu_avg_data):
    percentile = i / n
    if percentile < 0.3:
        label = 0
    elif percentile < 0.7:
        label = 1
    else:
        label = 2
    labeled_data.append((node_gpu, avg_time, label))

# ------------------------------------------------
# 4. Output to CSV
#    One GPU per row:
#    [Average performance(time), Label, nodeID+gpuID]
# ------------------------------------------------
with open(os.path.join(base_dir, 'gpu_performance_30.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header (can be kept or removed as needed)
    writer.writerow(['AverageTime(s)', 'Label', 'Node_GPU'])
    for node_gpu, avg_time, label in labeled_data:
        writer.writerow([avg_time, label, node_gpu])

####################################################################################################
# ------------------------------------------------
# 3. Assign labels based on sorted position
#    First 10% -> label=0
#    Middle 10%~90% -> label=1
#    Last 10% -> label=2
# ------------------------------------------------
n = len(gpu_avg_data)
labeled_data = []
for i, (node_gpu, avg_time) in enumerate(gpu_avg_data):
    percentile = i / n
    if percentile < 0.1:
        label = 0
    elif percentile < 0.9:
        label = 1
    else:
        label = 2
    labeled_data.append((node_gpu, avg_time, label))

# ------------------------------------------------
# 4. Output to CSV
#    One GPU per row:
#    [Average performance(time), Label, nodeID+gpuID]
# ------------------------------------------------
with open(os.path.join(base_dir, 'gpu_performance_10.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header (can be kept or removed as needed)
    writer.writerow(['AverageTime(s)', 'Label', 'Node_GPU'])
    for node_gpu, avg_time, label in labeled_data:
        writer.writerow([avg_time, label, node_gpu])

####################################################################################################
# ------------------------------------------------
# 3. Assign labels based on sorted position
#    First 1% -> label=0
#    Middle 1%~99% -> label=1
#    Last 1% -> label=2
# ------------------------------------------------
n = len(gpu_avg_data)
labeled_data = []
for i, (node_gpu, avg_time) in enumerate(gpu_avg_data):
    percentile = i / n
    if percentile < 0.01:
        label = 0
    elif percentile < 0.99:
        label = 1
    else:
        label = 2
    labeled_data.append((node_gpu, avg_time, label))

# ------------------------------------------------
# 4. Output to CSV
#    One GPU per row:
#    [Average performance(time), Label, nodeID+gpuID]
# ------------------------------------------------
with open(os.path.join(base_dir, 'gpu_performance_1.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header (can be kept or removed as needed)
    writer.writerow(['AverageTime(s)', 'Label', 'Node_GPU'])
    for node_gpu, avg_time, label in labeled_data:
        writer.writerow([avg_time, label, node_gpu])

print("done finding slowest gpus")