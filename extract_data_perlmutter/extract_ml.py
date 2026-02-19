#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import statistics

def parse_gemm(gemm_file):
    """
    Parse gemm.out file:
    - Only read lines with "size 32768"
    - Extract Node ID (e.g., nid002792 -> 2792), runtime (average: X s)
    - Return (group_count, gemm_min, gemm_mean, gemm_max) and the set of all node IDs (used for calculating groups)
    """
    pattern = re.compile(
        r"MPI Rank:\s*(\d+),\s*SLURM Node ID:\s*nid(\d+),\s*GPU ID:\s*(\d+),.*size\s*32768\s*average:\s*([0-9.]+)\s*s"
    )
    times = []
    node_ids = set()

    with open(gemm_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # rank = int(match.group(1))  # Can be stored if needed
                node_id_str = match.group(2)
                # gpu_id = int(match.group(3))  # Can be stored if needed
                avg_time = float(match.group(4))
                node_ids.add(int(node_id_str))
                times.append(avg_time)

    if len(times) == 0:
        # If no lines matched, return default 0 values or special handling
        return 0, 0.0, 0.0, 0.0

    gemm_min = min(times)
    gemm_mean = statistics.mean(times)
    gemm_max = max(times)

    # Calculate Dragonfly Group count
    # node_id // 128 => group_id
    group_ids = set([nid // 128 for nid in node_ids])
    group_count = len(group_ids)

    return group_count, gemm_min, gemm_mean, gemm_max


def parse_allreduce(allreduce_file, app):
    """
    Parse allreduce.out:
    - Line format: "<size> <time> seconds"
    - Choose size to read based on application:
      * AMG2023: 1024 -> allreduce_1K, 1048576 -> allreduce_1M
      * nanoGPT / deepCAM: 16777216 -> allreduce_16M, 2147483648 -> allreduce_2G
    - Only one record will be found, return dict
    """
    pattern = re.compile(r"(\d+)\s+([0-9.]+)\s+seconds")
    result = {}
    # Initialize to prevent empty returns
    if app == "AMG2023" or app == "MILC":
        result["allreduce_1K"] = 0.0
        result["allreduce_2K"] = 0.0
        result["allreduce_4K"] = 0.0
        result["allreduce_8K"] = 0.0
        result["allreduce_16K"] = 0.0
        result["allreduce_32K"] = 0.0
        result["allreduce_64K"] = 0.0
        result["allreduce_128K"] = 0.0
        result["allreduce_256K"] = 0.0
        result["allreduce_512K"] = 0.0
        result["allreduce_1M"] = 0.0
        sizes_map = {
            1024: "allreduce_1K",
            2048: "allreduce_2K",
            4096: "allreduce_4K",
            8192: "allreduce_8K",
            16384: "allreduce_16K",
            32768: "allreduce_32K",
            65536: "allreduce_64K",
            131072: "allreduce_128K",
            262144: "allreduce_256K",
            524288: "allreduce_512K",
            1048576: "allreduce_1M"
        }
    else:
        # nanoGPT, deepCAM
        result["allreduce_16M"] = 0.0
        result["allreduce_32M"] = 0.0
        result["allreduce_64M"] = 0.0
        result["allreduce_128M"] = 0.0
        result["allreduce_256M"] = 0.0
        result["allreduce_512M"] = 0.0
        result["allreduce_1G"] = 0.0
        result["allreduce_2G"] = 0.0
        sizes_map = {
            16777216: "allreduce_16M",
            33554432: "allreduce_32M",
            67108864: "allreduce_64M",
            134217728: "allreduce_128M",
            268435456: "allreduce_256M",
            536870912: "allreduce_512M",
            1073741824: "allreduce_1G",
            2147483648: "allreduce_2G"
        }

    with open(allreduce_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                size = int(match.group(1))
                time_val = float(match.group(2))
                if size in sizes_map:
                    result[sizes_map[size]] = time_val
    return result


def parse_counters(app_log_file):
    """
    Parse MPICH Slingshot CXI Counter:
    Return dictionary: {
      "<countername>_min": float,
      "<countername>_mean": float,
      "<countername>_max": float,
      ...
    }
    If the section or counter is not found, return empty dictionary
    """
    counters = {}

    # Flag to determine when to start parsing
    start_flag = False
    # Header typically looks like:
    # Counter                                Samples          Min         (/s)         Mean         (/s)          Max         (/s)
    header_pattern = re.compile(r"^Counter\s+Samples\s+Min\s+\(/s\)\s+Mean\s+\(/s\)\s+Max\s+\(/s\)")
    # Regex to match a counter line (using split is simpler here):
    # Name Samples Min (/s) Mean (/s) Max (/s)
    # Note that counter names might contain underscores, so using split is safer
    # line.split() => [name, samples, min, min(/s), mean, mean(/s), max, max(/s)]
    
    with open(app_log_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        # Start parsing when "MPICH Slingshot CXI Counter Summary:" is detected
        if "MPICH Slingshot CXI Counter Summary:" in line:
            start_flag = True
            i += 1
            continue

        if start_flag:
            # Skip header line
            if header_pattern.search(line):
                i += 1
                continue
            if not line.strip():
                i += 1
                continue

            parts = line.split()
            # Need at least 8 columns
            if len(parts) < 8:
                # Might indicate end of section (or format mismatch)
                i += 1
                continue

            counter_name = parts[0]
            # parts[1] => samples (not needed)
            c_min = float(parts[2])
            # parts[3] => min/s, not used
            c_mean = float(parts[4])
            # parts[5] => mean/s, not used
            c_max = float(parts[6])
            # parts[7] => max/s, not used

            counters[f"{counter_name}_min"] = c_min
            counters[f"{counter_name}_mean"] = c_mean
            counters[f"{counter_name}_max"] = c_max

        i += 1

    return counters


def parse_runtime_amg(amg_file):
    """
    Look for in amg.out:
    GMRES Solver:
    (next line)
    wall clock time = 329.502154 seconds
    """
    runtime = 0.0
    with open(amg_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "GMRES Solver:" in line:
            # Next line should have "wall clock time = X seconds"
            if i+1 < len(lines):
                next_line = lines[i+1]
                m = re.search(r"wall clock time\s*=\s*([0-9.]+)\s*seconds", next_line)
                if m:
                    runtime = float(m.group(1))
            break
    return runtime

def parse_runtime_milc(milc_file):
    """
    Parse the milc.out file in MILC:
    Search for a line like "Time = 1.563461e+02 seconds" and extract the time value (seconds).
    """
    pattern = re.compile(r"Time\s*=\s*([0-9.e\+]+)\s*seconds")
    with open(milc_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None

def parse_runtime_nanogpt(nanogpt_file):
    """
    Match all iter lines in nanoGPT.out:
    iter 30: loss 8.7970, time 8207.68ms, ...
    Sum up time (ms) -> convert to seconds
    """
    pattern = re.compile(r"iter\s+\d+:\s+loss\s+[\d.]+,\s+time\s+([0-9.]+)ms")
    total_time_s = 0.0
    with open(nanogpt_file, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ms_val = float(m.group(1))
                total_time_s += (ms_val / 1000.0)
    return total_time_s


def parse_runtime_deepcam(deepcam_file):
    """
    Match all step lines in deepcam.out:
    step 1831: time 74.13ms
    Sum up -> convert to seconds
    """
    pattern = re.compile(r"step\s+\d+:\s+time\s+([0-9.]+)ms")
    total_time_s = 0.0
    with open(deepcam_file, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ms_val = float(m.group(1))
                total_time_s += (ms_val / 1000.0)
    return total_time_s


def main():
    """
    Main function: Traverse three applications and their subdirectories, parse data and output CSV.
    """
    # Base directory (modify according to your actual situation)
    base_path = "/pscratch/sd/c/cunyang/result"

    # Applications and their output files
    apps_info = {
        "AMG2023": "amg.out",
        "MILC": "milc.out",
        "deepCAM": "deepcam.out",
        "nanoGPT": "nanoGPT.out"
    }

    # We first store parsing results in a list, each element is a dict.
    # Dictionary keys will be the CSV column names, for example:
    # {
    #   "group_count": ...,
    #   "gemm_min": ...,
    #   "gemm_mean": ...,
    #   "gemm_max": ...,
    #   "allreduce_1K": ...,
    #   "allreduce_1M": ...,
    #   ... counters ...
    #   "runtime": ...
    # }
    data_rows = []

    # To write all possible counter columns later, we collect a global set of counter names (without _min/_mean/_max)
    global_counters_set = set()
    
    folder_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-job(\d+)$")

    for app_name, app_log_name in apps_info.items():
        app_dir = os.path.join(base_path, app_name, "64nodes")
        if not os.path.isdir(app_dir):
            continue

        # Get subdirectories like 2024-12-30_16-14-51-job34357636
        for run_folder in os.listdir(app_dir):
            # Check if run_folder matches the expected pattern
            match = folder_pattern.match(run_folder)
            print(app_name, run_folder)
            if not match:
                continue
            
            # Extract date from folder name
            folder_date = match.group(1)  # This gets the YYYY-MM-DD part
            
            # Skip if date is before 2025-02-08
            if folder_date < "2025-02-08":
                continue
            if app_name == "MILC" and folder_date < "2025-04-01":
                continue
            elif folder_date >= "2025-04-10":
                continue
            
            sub_path = os.path.join(app_dir, run_folder)
            if not os.path.isdir(sub_path):
                continue  # Skip if not a directory
            
            run_time_str = ""
            job_id_str = ""
            run_time_str = match.group(1)  # "2024-12-30_16-14-51"
            job_id_str = match.group(2)    # "34357636"

            # gemm.out
            gemm_file = os.path.join(sub_path, "gemm.out")
            # allreduce.out
            allreduce_file = os.path.join(sub_path, "allreduce.out")
            # Application log
            app_out_file = os.path.join(sub_path, app_log_name)

            if (not os.path.isfile(gemm_file) or 
                not os.path.isfile(allreduce_file) or
                not os.path.isfile(app_out_file)):
                # Skip if required files are missing
                continue

            # 1) Parse gemm
            group_count, gemm_min, gemm_mean, gemm_max = parse_gemm(gemm_file)

            # 2) Parse allreduce
            allreduce_dict = parse_allreduce(allreduce_file, app_name)
            

            # 3) Parse counter
            counters_dict = parse_counters(app_out_file)

            # Extract counter base names and add to global_counters_set
            for key in counters_dict.keys():
                # Key format is like "atu_cache_evictions_min"
                # We want to get the base name, e.g., "atu_cache_evictions"
                # Removing _min / _mean / _max suffix
                if key.endswith("_min"):
                    base_name = key[:-4]  # Remove "_min"
                elif key.endswith("_mean"):
                    base_name = key[:-5]
                elif key.endswith("_max"):
                    base_name = key[:-4]
                else:
                    # Theoretically should not have other cases
                    base_name = key
                global_counters_set.add(base_name)

            # 4) Parse runtime
            if app_name == "AMG2023":
                runtime = parse_runtime_amg(app_out_file)
            elif app_name == "MILC":
                runtime = parse_runtime_milc(app_out_file)
            elif app_name == "nanoGPT":
                runtime = parse_runtime_nanogpt(app_out_file)
            else:
                # deepCAM
                runtime = parse_runtime_deepcam(app_out_file)

            # Prepare row record for this run
            row = {}
            row["app_name"] = app_name        # Application name
            row["run_time"] = run_time_str    # Time parsed from folder name
            row["job_id"] = job_id_str        # Job ID parsed from folder name
            
            row["group_count"] = group_count
            row["gemm_min"] = gemm_min
            row["gemm_mean"] = gemm_mean
            row["gemm_max"] = gemm_max

            if app_name == "AMG2023" or app_name == "MILC":
                # allreduce_1K / allreduce_1M
                row["allreduce_1K"] = allreduce_dict.get("allreduce_1K", 0.0)
                row["allreduce_2K"] = allreduce_dict.get("allreduce_2K", 0.0)
                row["allreduce_4K"] = allreduce_dict.get("allreduce_4K", 0.0)
                row["allreduce_8K"] = allreduce_dict.get("allreduce_8K", 0.0)
                row["allreduce_16K"] = allreduce_dict.get("allreduce_16K", 0.0)
                row["allreduce_32K"] = allreduce_dict.get("allreduce_32K", 0.0)
                row["allreduce_64K"] = allreduce_dict.get("allreduce_64K", 0.0)
                row["allreduce_128K"] = allreduce_dict.get("allreduce_128K", 0.0)
                row["allreduce_256K"] = allreduce_dict.get("allreduce_256K", 0.0)
                row["allreduce_512K"] = allreduce_dict.get("allreduce_512K", 0.0)
                row["allreduce_1M"] = allreduce_dict.get("allreduce_1M", 0.0)
            else:
                # nanoGPT / deepCAM
                row["allreduce_16M"] = allreduce_dict.get("allreduce_16M", 0.0)
                row["allreduce_32M"] = allreduce_dict.get("allreduce_32M", 0.0)
                row["allreduce_64M"] = allreduce_dict.get("allreduce_64M", 0.0)
                row["allreduce_128M"] = allreduce_dict.get("allreduce_128M", 0.0)
                row["allreduce_256M"] = allreduce_dict.get("allreduce_256M", 0.0)
                row["allreduce_512M"] = allreduce_dict.get("allreduce_512M", 0.0)
                row["allreduce_1G"] = allreduce_dict.get("allreduce_1G", 0.0)
                row["allreduce_2G"] = allreduce_dict.get("allreduce_2G", 0.0)

            # Add counters
            for c_key, c_val in counters_dict.items():
                # c_key format is like "atu_cache_evictions_min" etc.
                row[c_key] = c_val

            row["runtime"] = runtime

            data_rows.append(row)

    # ---- Output unified CSV ----
    # According to specified order:
    # First column: group_count
    # Next three columns: gemm_min, gemm_mean, gemm_max
    # Then depending on application type: allreduce_1K/allreduce_1M or allreduce_16M/allreduce_2G
    # Then all counters (each counter has _min, _mean, _max columns) - if a run doesn't have them, fill with 0
    # Last column: runtime
    #
    # For simplicity, first collect all possible column names:
    # 1) group_count, gemm_min, gemm_mean, gemm_max
    # 2) allreduce_1K, allreduce_1M, allreduce_16M, allreduce_2G (some applications won't use these, set to 0)
    # 3) counters: for each base_name in global_counters_set, add base_name_min, base_name_mean, base_name_max
    # 4) runtime

    # Create counter columns (sort to ensure consistent output order)
    sorted_counters = sorted(global_counters_set)

    # Assemble final CSV column order
    header = [
        "app_name",
        "run_time",
        "job_id",
        "group_count",
        "gemm_min",
        "gemm_mean",
        "gemm_max",
        # Include all, different applications will have empty values
        "allreduce_1K",
        "allreduce_2K",
        "allreduce_4K",
        "allreduce_8K",
        "allreduce_16K",
        "allreduce_32K",
        "allreduce_64K",
        "allreduce_128K",
        "allreduce_256K",
        "allreduce_512K",
        "allreduce_1M",
        "allreduce_16M",
        "allreduce_32M",
        "allreduce_64M",
        "allreduce_128M",
        "allreduce_256M",
        "allreduce_512M",
        "allreduce_1G",
        "allreduce_2G",
    ]
    # Add three columns for each counter
    for base_name in sorted_counters:
        header.append(f"{base_name}_min")
        header.append(f"{base_name}_mean")
        header.append(f"{base_name}_max")

    # Last column: runtime
    header.append("runtime")

    # Write CSV
    # You can change the filename as needed, e.g., to combined.csv
    out_csv = "combined_results_pm.csv"
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)  # Write header

        skipped_count = 0
        for row in data_rows:
            # Skip if runtime is 0
            if row.get("runtime", 0.0) == 0.0:
                skipped_count += 1
                continue
                
            row_out = []
            # Fill values in order of header
            for col in header:
                val = row.get(col, 0.0)  # If current row doesn't have this column, use 0.0
                row_out.append(val)
            writer.writerow(row_out)
    
    print(f"Parsing complete, {len(data_rows)} run records processed, {skipped_count} records skipped due to zero runtime, results written to {out_csv}.")


if __name__ == "__main__":
    main()

