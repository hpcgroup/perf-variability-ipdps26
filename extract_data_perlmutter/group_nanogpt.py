import os
import re
import glob
import matplotlib.pyplot as plt
import datetime
NODES=64

base_path = "/pscratch/sd/c/cunyang/result/nanoGPT/" + str(NODES)+ "nodes"

# Regular expression for matching node ID lines in gemm.out
# Example: "MPI Rank: 230, SLURM Node ID: nid003332, GPU ID: 2, bfloat16: ..."
gemm_regex = re.compile(r"SLURM Node ID:\s*nid0*([0-9]+)")
DIRNAME_PATTERN = re.compile(
    r'^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})-job\d+'
)
# Regular expression for matching iteration time lines in nanoGPT.out
# Example: "iter 30: loss 8.7970, time 8207.68ms, mfu 26.41%, ..."
nanoGPT_time_regex = re.compile(r"time\s+([\d.]+)ms")
DATE_THRESHOLD1 = datetime.datetime(2025, 4, 10)
# Store (group_count, total_time_in_ms) data
results = []

def parse_dir_datetime(dirname: str):
    """
    根据目录名形如 2024-12-26_12-22-14-jobxxxxx，
    提取出 datetime 对象，用于和 DATE_THRESHOLD 比较。
    """
    m = DIRNAME_PATTERN.match(dirname)
    if not m:
        return None
    date_str = m.group('date')  # e.g. "2024-12-26"
    time_str = m.group('time')  # e.g. "12-22-14"
    try:
        # 替换一下 time_str 的 '-' 为 ':' 以便解析
        dt_str = date_str + " " + time_str.replace('-', ':')
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt
    except ValueError:
        return None

# Traverse all subdirectories with pattern like "2024-12-30_16-14-51-job34357636"
for folder in os.listdir(base_path):
    #print(folder)
    dt = parse_dir_datetime(folder)
    if dt and  dt >= DATE_THRESHOLD1:
        print(folder)
        continue
    full_folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(full_folder_path):
        continue  # Skip non-directory files

    gemm_file = os.path.join(full_folder_path, "gemm.out")
    nano_file = os.path.join(full_folder_path, "nanoGPT.out")

    if not (os.path.exists(gemm_file) and os.path.exists(nano_file)):
        # Skip if both files don't exist in this directory
        continue

    # 1) Parse gemm.out to get all node IDs, corresponding to groups
    node_ids = set()
    with open(gemm_file, 'r') as f:
        for line in f:
            match = gemm_regex.search(line)
            if match:
                node_id_str = match.group(1)  # e.g. '3332'
                try:
                    node_id = int(node_id_str)
                    node_ids.add(node_id)
                except ValueError:
                    pass

    # Calculate unique groups after deduplication
    groups = set(nid // 128 for nid in node_ids)
    group_count = len(groups)

    # 2) Parse nanoGPT.out to sum up time for all iterations
    total_time_ms = 0.0
    with open(nano_file, 'r') as f:
        for line in f:
            time_match = nanoGPT_time_regex.search(line)
            if time_match:
                time_str = time_match.group(1)  # e.g. '8207.68'
                try:
                    time_val = float(time_str)
                    total_time_ms += time_val
                except ValueError:
                    pass

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
# Convert times from milliseconds to seconds
y_vals = [time_ms / 1000.0 for time_ms in y_vals]  # Convert ms to seconds

print(x_vals)
print(y_vals)
exit()
# Update y-axis label for the plot
plt.figure(figsize=(6, 4), dpi=120)
plt.scatter(x_vals, y_vals, s=15, color='#07519c')
plt.xlabel("Number of distinct Dragonfly groups")
plt.ylabel("Total run time (s)")
plt.title("nanoGPT: Dragonfly group count vs. run time")
plt.grid(True)
# Set x-axis ticks with step size of 1
if x_vals:  # Check if x_vals is not empty
    min_x, max_x = min(x_vals), max(x_vals)
    plt.xticks(range(min_x, max_x + 1, 1))
plt.tight_layout()

# To save the image, use plt.savefig("group_vs_time.png")
plt.savefig("group_vs_time_nanogpt_pm" + str(NODES) + ".png", dpi=300)
plt.savefig("group_vs_time_nanogpt_pm" + str(NODES) + ".pdf")

