import os
import re
import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ---------------------------------------------------------------------
# A) Read global GPU performance labels (gpu_performance.csv)
#    Create a dictionary: label_dict["nid001234_GPU2"] = 0/1/2
# ---------------------------------------------------------------------
gpu_label_csv = "/pscratch/sd/c/cunyang/result/gpu_performance_1.csv" 
label_dict = {}

with open(gpu_label_csv, 'r') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # ['AverageTime(s)', 'Label', 'Node_GPU'] (example)
    for row in reader:
        if len(row) < 3:
            continue
        # row = [avg_time_str, label_str, node_gpu_str]
        label = int(row[1])
        node_gpu_str = row[2]  # e.g. "nid001234_GPU3"
        label_dict[node_gpu_str] = label

# ---------------------------------------------------------------------
# B) Basic Configuration
# ---------------------------------------------------------------------
NODES = 64
base_path = "/pscratch/sd/c/cunyang/result/nanoGPT/" + str(NODES) + "nodes"

# Regex to identify job folders, e.g. "2024-12-30_16-14-51-job34357636"
folder_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-job\d+')

# Regex to match lines in gemm.out containing "size 32768", extract NodeID + GPU ID
gemm_line_pattern = re.compile(
    r'SLURM Node ID:\s*(\S+),\s*GPU ID:\s*(\d+).*size 32768 average:\s*([\d\.]+)\s*s'
)

# Regex to match iteration time lines in nanoGPT.out
# Example: "iter 30: loss 8.7970, time 8207.68ms, mfu 26.41%, ..."
nanoGPT_time_regex = re.compile(r'time\s+([\d.]+)ms')

# ---------------------------------------------------------------------
# C) Traverse directories, parse gemm.out + nanoGPT.out
#    For each job, compute (number of slow GPUs, application total time in ms)
# ---------------------------------------------------------------------
results = []  # Store (num_slow, total_time_ms)

folders = os.listdir(base_path)
for folder_name in folders:
    if not folder_pattern.fullmatch(folder_name):
        continue
    
    job_folder = os.path.join(base_path, folder_name)
    if not os.path.isdir(job_folder):
        continue
    
    gemm_file = os.path.join(job_folder, "gemm.out")
    nano_file = os.path.join(job_folder, "nanoGPT.out")
    if not os.path.isfile(gemm_file) or not os.path.isfile(nano_file):
        continue
    
    # 1) Parse allocated GPUs from gemm.out
    used_gpus = set()
    with open(gemm_file, 'r') as gf:
        for line in gf:
            if "size 32768" in line:
                match_g = gemm_line_pattern.search(line)
                if match_g:
                    node_id = match_g.group(1)  # e.g. "nid001234"
                    gpu_id  = match_g.group(2)  # e.g. "0"
                    node_gpu_str = f"{node_id}_GPU{gpu_id}"
                    used_gpus.add(node_gpu_str)
    
    # 2) Count slow GPUs
    num_slow = 0
    for g in used_gpus:
        if g in label_dict and label_dict[g] == 2:
            num_slow += 1
    
    # 3) Extract total time from nanoGPT.out (sum all "time xxxx ms")
    total_time_ms = 0.0
    with open(nano_file, 'r') as nf:
        for line in nf:
            time_match = nanoGPT_time_regex.search(line)
            if time_match:
                time_str = time_match.group(1)
                try:
                    total_time_ms += float(time_str)
                except ValueError:
                    pass
    
    # Collect results for this job
    results.append((num_slow, total_time_ms))

# ---------------------------------------------------------------------
# D) Simple correlation analysis + plotting
# ---------------------------------------------------------------------
x_vals = [r[0] for r in results]  # slow GPU count
y_vals = [r[1] / 1000 for r in results]  # total time in s

# Pearson correlation coefficient
corr, pval = pearsonr(x_vals, y_vals)
print(f"Total records analyzed: {len(results)}")
print(f"Pearson correlation coefficient: {corr:.4f}, p-value: {pval:.4e}")

print(x_vals)
print(y_vals)
# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(x_vals, y_vals, alpha=0.7, color='#07519c')
plt.xlabel("Count of Slow GPUs")
plt.ylabel("nanoGPT Total Time (s)")
plt.title(f"Slow GPU Count vs. nanoGPT Runtime\n(PearsonR={corr:.3f}, p={pval:.3e})")
plt.grid(True)
plt.tight_layout()
plt.savefig("slow_gpu_vs_nanogpt_runtime_1.png", dpi=150)
plt.savefig("slow_gpu_vs_nanogpt_runtime_1.pdf")
plt.show()
