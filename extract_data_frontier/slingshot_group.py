import os
import re
import glob
import matplotlib.pyplot as plt
import subprocess

import util
import count_runs
# Import necessary functions from neighborhood_extract.py
from neighborhood_extract import parse_node_id, expand_node_list, get_job_start_end_nodelist

NODES=64
app="deepcam"
base_path = f"/lustre/orion/csc547/scratch/keshprad/perfvar/{app}_logs/{NODES}nodes"

# Store (group_count, total_time_in_ms) data
results = []

# Traverse all subdirectories with pattern like "2024-12-30_16-14-51-job34357636"
for folder in os.listdir(base_path):
    full_folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(full_folder_path):
        continue  # Skip non-directory files
    if not util.verify_app_dir(folder, app, NODES):
        # skip directories which are not app runs
        continue

    # Get job ID from folder name
    job_id = util.parse_job_id(folder, app, NODES)
    if job_id is None:
        continue
        
    # Get node list using sacct
    _, _, node_list_raw = get_job_start_end_nodelist(job_id)
    if not node_list_raw:
        print(f"Warning: No nodes found for job {job_id}")
        continue
    
    # Expand node list
    node_expanded = []
    for node_str in node_list_raw:
        node_expanded.extend(expand_node_list(node_str))
    
    # Extract node IDs
    node_ids = set()
    for node_str in node_expanded:
        node_id = parse_node_id(node_str)
        if node_id is not None:
            node_ids.add(node_id)
            
    if not node_ids:
        print(f"Warning: Could not parse any node IDs for job {job_id}")
        continue

    # Calculate unique groups after deduplication
    groups = set(nid // 128 for nid in node_ids)
    group_count = len(groups)
    
    # Parse app time for all iterations
    total_time_ms = util.parse_app_time(full_folder_path, app, NODES)
    if total_time_ms is None:
        continue

    # Save results
    results.append((group_count, total_time_ms))

print("Total number of runs:", len(results))
# --- Scatter plot creation ---
# x-axis = group_count, y-axis = total_time_ms
# To convert to seconds, divide total_time_ms by 1000
x_vals = [r[0] for r in results]
# Filter out data points where time exceeds 300000 ms
# if NODES == 64:
#     results = [r for r in results if r[1] <= 300000]
# else:
#     results = [r for r in results if r[1] <= 200000]
x_vals = [r[0] for r in results]  # Recalculate x_vals with filtered data
y_vals = [r[1] for r in results]
print(x_vals)
print(y_vals)

plt.figure(figsize=(6, 4), dpi=120)
plt.scatter(x_vals, y_vals, alpha=0.7)
plt.xlabel("Number of distinct Dragonfly groups")
plt.ylabel("Total run time (s)")
plt.title(f"{app}: Dragonfly group count vs. total run time")
plt.grid(True)
# Set x-axis ticks with step size of 1
if x_vals:  # Check if x_vals is not empty
    min_x, max_x = min(x_vals), max(x_vals)
    plt.xticks(range(0, max_x + 1, 4))
plt.tight_layout()

# To save the image, use plt.savefig("group_vs_time.png")
plt.savefig(os.path.join(base_path, f"group_vs_time{NODES}.png"), dpi=300)
plt.savefig(os.path.join(base_path, f"group_vs_time{NODES}.pdf"))
