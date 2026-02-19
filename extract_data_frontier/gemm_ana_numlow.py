import csv
import os
import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import util

# CONFIG
base_dir = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar")
app = "nanoGPT"
nodes = 64
lowest_percentage = 1 # Valid: 1 / 10 / 30
gemm_data_size = 32768


app_base_dir = os.path.join(base_dir, f"{app}_logs", f'{nodes}nodes')
csv_file = os.path.join(base_dir, f"gpu_performance_{lowest_percentage}.csv")

# For counting the number of slow GPUs (label=2) on each node
slow_count_by_node = {}

# If you want nodes without slow GPUs to also show as 0 in the graph, collect all node IDs first
all_nodes = set()

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header: ['AverageTime(s)', 'Label', 'Node_GPU']
    for row in reader:
        if len(row) < 3:
            continue
        avg_time_str, label_str, node_gpu_str = row
        label = int(label_str)
        
        # Extract nodeID (nid001234) from node_gpu_str (e.g., "nid001234_GPU2")
        match = re.search(r'(frontier\d+)_GPU', node_gpu_str)
        if match:
            node_id = match.group(1)
            all_nodes.add(node_id)

            # If it's a slow label (label=2), increment the count for this node
            if label == 2:
                slow_count_by_node[node_id] = slow_count_by_node.get(node_id, 0) + 1

# Fill in 0 for nodes without slow GPUs to ensure they appear in the graph
for n in all_nodes:
    if n not in slow_count_by_node:
        slow_count_by_node[n] = 0

# Sort slow_count_by_node by node number for left-to-right display in the graph
# (Assuming nodeID like "nid001234" can be parsed to integer 1234 for sorting)
def parse_node_num(nid_str):
    m = re.search(r'frontier(\d+)', nid_str)
    return int(m.group(1)) if m else 999999  # If not found, use a large number
sorted_nodes = sorted(slow_count_by_node.keys(), key=parse_node_num)

# Prepare x, y for plotting
x_vals = range(len(sorted_nodes))
y_vals = [slow_count_by_node[n] for n in sorted_nodes]

# # Create scatter plot
# plt.figure(figsize=(12,6))
# plt.scatter(x_vals, y_vals, s=50, c='red')

# # Set X-axis ticks as node IDs and rotate to avoid overlap
# plt.xticks(x_vals, sorted_nodes, rotation=45, ha='right')

# plt.xlabel('Node ID', fontsize=14)
# plt.ylabel('Count of Slow GPUs (label=2)', fontsize=14)
# plt.title('Number of Slow GPUs per Node', fontsize=16)
# plt.tight_layout()
# plt.savefig(os.path.join(base_dir, "slow_gpus_per_node.png"), dpi=150)
# plt.savefig(os.path.join(base_dir, "slow_gpus_per_node.pdf"))
# # plt.show()


# ---------------------------------------------------------------------
# 1) Read global GPU performance labels (gpu_performance.csv)
#    Generate a dictionary: label_dict["nid001234_GPU2"] = 0/1/2
# ---------------------------------------------------------------------
gpu_label_csv = csv_file
label_dict = {}

with open(gpu_label_csv, 'r') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # ['AverageTime(s)', 'Label', 'Node_GPU']
    for row in reader:
        if len(row) < 3:
            continue
        # row: [avg_time_str, label_str, node_gpu_str]
        label = int(row[1])
        node_gpu = row[2]  # e.g., "nid001234_GPU3"
        label_dict[node_gpu] = label

# ---------------------------------------------------------------------
# 2) Traverse job folders under /pscratch/sd/c/cunyang/result/AMG2023/64nodes/
#    Match format like "2025-03-17_04-57-04-job36942712"
# ---------------------------------------------------------------------
# Regular expression for parsing gemm.out
gemm_line_pattern = re.compile(
    r'SLURM Node ID:\s*(\S+),\s*GPU ID:\s*(\d+).*size 32768 average:\s*([\d\.]+)\s*s'
)

results = []  # Store (num_slow, runtime) parsed from each job
folders = [os.path.join(app_base_dir, d) for d in os.listdir(app_base_dir) if util.verify_app_dir(d, app, nodes)]

# ===========================================================================
# DON'T INCLUDE OLD LOGS WITH DIFFERENT SETUP -> COULD HAVE DIFFERENT RUNTIME
# # include OLD nanogpt logs which didnt have torch profiler output
# if app == "nanoGPT_logs":
#     folders = folders + [os.path.join(app_base_dir, '..', f"{nodes}nodes_notorchprof", d) 
#                          for d in os.listdir(os.path.join(app_base_dir, '..', f"{nodes}nodes_notorchprof")) 
#                          if util.verify_app_dir(d, app, nodes)]
# # include OLD nanogpt/deepcam logs which didnt have RDZV env variables
# if app == "nanoGPT_logs" or app == "deepcam_logs":
#     folders = folders + [os.path.join(app_base_dir, '..', f"{nodes}nodes_no_RDZV_env", d) 
#                          for d in os.listdir(os.path.join(app_base_dir, '..', f"{nodes}nodes_no_RDZV_env")) 
#                          if util.verify_app_dir(d, app, nodes)]
# ===========================================================================

for job_folder in folders:
    if not os.path.isdir(job_folder):
        continue
    
    # Path to gemm.out
    gemm_file = os.path.join(job_folder, "output-gemm.log")
    if not os.path.isfile(gemm_file):
        # Skip if files are missing
        continue
    
    # -----------------------------------------------------------------
    # 2.1) Find allocated GPUs from gemm.out
    # -----------------------------------------------------------------
    used_gpus = set()
    with open(gemm_file, 'r') as gf:
        lines = gf.readlines()
        for line in lines:
            if f"size {gemm_data_size}" in line:
                m = gemm_line_pattern.search(line)
                if m:
                    node_id = m.group(1)  # e.g., "nid001234"
                    gpu_id  = m.group(2)  # e.g., "0"
                    # Compose "nid001234_GPU0"
                    node_gpu_id = f"{node_id}_GPU{gpu_id}"
                    used_gpus.add(node_gpu_id)

    # -----------------------------------------------------------------
    # 2.2) Count the number of "slow GPUs" (label=2) in this job
    # -----------------------------------------------------------------
    num_slow = 0
    for g in used_gpus:
        # If it's in the global dictionary and labeled as 2, count it
        if g in label_dict and label_dict[g] == 2:
            num_slow += 1

    # -----------------------------------------------------------------
    # 2.3) Read application runtime
    # -----------------------------------------------------------------
    runtime = util.parse_app_time(job_folder, app, nodes)
    
    if runtime is not None:
        # Store the (slow GPU count, runtime) pair
        results.append((num_slow, runtime))

# ---------------------------------------------------------------------
# 3) Analysis and visualization: (slow GPU count, application runtime)
# ---------------------------------------------------------------------

# Extract x, y
x_vals = [r[0] for r in results]  # Slow GPU count
y_vals = [r[1] for r in results]  # Application runtime

# Simple correlation analysis (Pearson coefficient)
corr, pval = 0, 1
try:
    from scipy.stats import pearsonr
    corr, pval = pearsonr(x_vals, y_vals)
except ImportError:
    pass

print(f"Collected data from {len(results)} jobs.")
print(f"Pearson correlation coefficient = {corr:.4f}, p-value = {pval:.4f}")

print("xvals")
print(x_vals)
print("yvals")
print(y_vals)

# Create scatter plot
plt.figure(figsize=(8,6))
plt.scatter(x_vals, y_vals, color='#07519c', alpha=0.7)
plt.xlabel("Count of Slow GPUs")
plt.ylabel(f"{app} Runtime (s)")
plt.title(f"Slow GPU Count vs. {app} Runtime\n(PearsonR={corr:.3f}, p={pval:.3f})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(app_base_dir, f"slow_gpu_vs_{app.lower()}_runtime_{lowest_percentage}_frontier.png"), dpi=150)
plt.savefig(os.path.join(app_base_dir, f"slow_gpu_vs_{app.lower()}_runtime_{lowest_percentage}_frontier.pdf"))
# plt.show()
