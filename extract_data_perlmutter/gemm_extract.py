import os
import re
import matplotlib.pyplot as plt
import numpy as np
import csv

# Regex: match folders like "2024-12-26_12-22-14-job34298015"
folder_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-job\d+')

# Regex: match target lines in gemm.out and capture Node ID, GPU ID and average time
line_pattern = re.compile(
    r'SLURM Node ID:\s*(\S+),\s*GPU ID:\s*(\d+).*size 32768 average:\s*([\d\.]+)\s*s'
)

result_path = "/pscratch/sd/c/cunyang/result"
# List of top-level directories to traverse
base_folders = ["AMG2023", "deepCAM", "MILC", "nanoGPT"]
# base_folders = ["AMG2023"]
# List of subfolders to enter
node_subfolders = ["16nodes", "64nodes"]
# node_subfolders = ["16nodes"]

# Data storage: each element is (node_id, gpu_id, time)
data = []

# Traverse directories
for base in base_folders:
    for sub in node_subfolders:
        base_path = os.path.join(result_path, base, sub)
        if not os.path.isdir(base_path):
            continue
        print(f"Processing {base_path}...")
        # Traverse folders within subdirectory
        for entry in os.listdir(base_path):
            # If folder name matches the pattern
            if folder_pattern.fullmatch(entry):
                gemm_file = os.path.join(base_path, entry, "gemm.out")
                if os.path.isfile(gemm_file):
                    with open(gemm_file, 'r') as f:
                        for line in f:
                            # Process lines containing "size 32768" only
                            if "size 32768" in line:
                                m = line_pattern.search(line)
                                if m:
                                    node_id = m.group(1)
                                    gpu_id = m.group(2)
                                    time_val = float(m.group(3))
                                    data.append((node_id, gpu_id, time_val))

allnodes = set()
for node, gpu, t in data:
    allnodes.add(node)
print(f"Total nodes: {len(allnodes)}")
allnodes = sorted(allnodes)
csv_path = ("allnodes.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['Node ID'])
    for node in allnodes:
        writer.writerow([node])
print(f"CSV file created: {csv_path}")

node_gpu_times = {}
for node, gpu, t in data:
    key = f"{node}_GPU{gpu}"  # Example: "nid001234_GPU0"
    node_gpu_times.setdefault(key, []).append(t)
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
with open('gpu_performance_30.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header (can be kept or removed as needed)
    writer.writerow(['AverageTime(s)', 'Label', 'Node_GPU'])
    for node_gpu, avg_time, label in labeled_data:
        writer.writerow([avg_time, label, node_gpu])

####################################################################################################

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
with open('gpu_performance_10.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header (can be kept or removed as needed)
    writer.writerow(['AverageTime(s)', 'Label', 'Node_GPU'])
    for node_gpu, avg_time, label in labeled_data:
        writer.writerow([avg_time, label, node_gpu])
        
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
with open('gpu_performance_1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header (can be kept or removed as needed)
    writer.writerow(['AverageTime(s)', 'Label', 'Node_GPU'])
    for node_gpu, avg_time, label in labeled_data:
        writer.writerow([avg_time, label, node_gpu])
