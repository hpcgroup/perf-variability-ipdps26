#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import statistics

import util

def parse_gemm(gemm_file):
    """
    Parse gemm.out file:
    - Only read lines with "size 32768"
    - Extract Node ID (e.g., nid002792 -> 2792), runtime (average: X s)
    - Return (group_count, gemm_min, gemm_mean, gemm_max) and the set of all node IDs (used for calculating groups)
    """
    pattern = re.compile(
        r"MPI Rank:\s*(\d+),\s*SLURM Node ID:\s*frontier(\d+),\s*GPU ID:\s*(\d+),.*size\s*32768\s*average:\s*([0-9.]+)\s*s"
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
    if app == "AMG2023":
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
        # TODO: revisit should this be nan
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
        result["allreduce_16M"] = 0.0
        result["allreduce_32M"] = 0.0
        result["allreduce_64M"] = 0.0
        result["allreduce_128M"] = 0.0
        result["allreduce_256M"] = 0.0
        result["allreduce_512M"] = 0.0
        result["allreduce_1G"] = 0.0
        result["allreduce_2G"] = 0.0
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
            1048576: "allreduce_1M",
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


def parse_counters(app_log_file, app):
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
        if f"end {app}:" in line:
            i += 1
            continue

        if start_flag:
            # Skip header line
            if header_pattern.search(line):
                i += 1
                continue
            if not line.strip():
                continue

            parts = line.split()
            # Need at least 8 columns
            if len(parts) != 8:
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

def main():
    """
    Main function: Traverse three applications and their subdirectories, parse data and output CSV.
    """
    # Base directory (modify according to your actual situation)
    base_path = os.path.join("/lustre/orion/csc547/scratch/keshprad/perfvar")

    # Applications and their output files
    apps_info = {
        "AMG2023": "output-AMG2023.log",
        "deepcam": "output-deepcam.log",
        "nanoGPT": "output-nanoGPT.log",
        "MILC": "output-MILC.log",
    }
    timestamp_info = {
        "AMG2023": "output-AMG2023.log",
        "deepcam": "output-deepcam.log",
        "nanoGPT": "output-nanoGPT.log",
        "MILC": "output-gemm.log",
    }
    counter_files = {
        "AMG2023": "output-AMG2023.log",
        "deepcam": "output-deepcam-with_performance_counters.log",
        "nanoGPT": "output-nanoGPT-with_performance_counters.log",
        "MILC": "output-MILC.log",
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
    #   "allreduce_16M": ...,
    #   "allreduce_2G": ...,
    #   ... counters ...
    #   "runtime": ...
    # }
    data_rows = []

    # To write all possible counter columns later, we collect a global set of counter names (without _min/_mean/_max)
    global_counters_set = set()

    for app_name, app_log_name in apps_info.items():
        app_dir = os.path.join(base_path, f"{app_name}_logs", "64nodes")
        if not os.path.isdir(app_dir):
            continue

        # Get subdirectories like 2024-12-30_16-14-51-job34357636
        for run_folder in os.listdir(app_dir):
            print(run_folder)
            # Check if run_folder matches the expected pattern
            if not util.verify_app_dir(run_folder, app_name, 64):
                continue
            
            sub_path = os.path.join(app_dir, run_folder)
            if not os.path.isdir(sub_path):
                continue  # Skip if not a directory
            
            date = util.parse_timestamp(os.path.join(sub_path, timestamp_info[app_name]), app_name if app_name != 'MILC' else 'gemm')
            
            run_time_str = ""
            job_id_str = ""
            run_time_str = str(date)
            job_id_str = util.parse_job_id(run_folder, app_name, 64) 

            # gemm.out
            gemm_file = os.path.join(sub_path, "output-gemm.log")
            # allreduce.out
            allreduce_file = os.path.join(sub_path, "output-allreduce.log")
            # Counters file
            counters_file = os.path.join(sub_path, counter_files[app_name])

            if (not os.path.isfile(gemm_file) or 
                not os.path.isfile(allreduce_file) or 
                not os.path.isfile(counters_file)):
                # Skip if required files are missing
                continue

            # 1) Parse gemm
            group_count, gemm_min, gemm_mean, gemm_max = parse_gemm(gemm_file)

            # 2) Parse allreduce
            allreduce_dict = parse_allreduce(allreduce_file, app_name)

            # 3) Parse counter
            counters_dict = parse_counters(counters_file, app_name)
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
            runtime = util.parse_app_time(sub_path, app_name, 64, use_perf_counter_run=True)
            if not runtime:
                # skip if no runtime
                print("no runtime, skipping")
                continue

            # Prepare row record for this run
            row = {}
            row["app_name"] = app_name        # Application name
            row["run_time"] = run_time_str    # Time parsed from folder name
            row["job_id"] = job_id_str        # Job ID parsed from folder name
            
            row["group_count"] = group_count
            row["gemm_min"] = gemm_min
            row["gemm_mean"] = gemm_mean
            row["gemm_max"] = gemm_max

            if app_name == "AMG2023":
                # allreduce_1K / allreduce_1M
                row["allreduce_1K"] = allreduce_dict.get("allreduce_1K", 0.0)
                row["allreduce_1M"] = allreduce_dict.get("allreduce_1M", 0.0)
            else:
                # nanoGPT / deepCAM
                row["allreduce_1K"] = allreduce_dict.get("allreduce_1K", 0.0)
                row["allreduce_1M"] = allreduce_dict.get("allreduce_1M", 0.0)
                row["allreduce_16M"] = allreduce_dict.get("allreduce_16M", 0.0)
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
        "allreduce_1M",
        "allreduce_16M",
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
    out_csv = os.path.join(base_path, "model_data.csv")
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(header)  # Write header

        for row in data_rows:
            row_out = []
            # Fill values in order of header
            for col in header:
                val = row.get(col, 0.0)  # If current row doesn't have this column, use 0.0
                row_out.append(val)
            writer.writerow(row_out)

    print(f"Parsing complete, {len(data_rows)} run records processed, results written to {out_csv}.")


if __name__ == "__main__":
    main()
