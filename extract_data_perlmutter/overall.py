#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import statistics

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

def parse_runitme_milc(filepath):
    """
    Parse the milc.out file in MILC:
    Search for a line like "Time = 1.563461e+02 seconds" and extract the time value (seconds).
    """
    pattern = re.compile(r"Time\s*=\s*([0-9.e\+]+)\s*seconds")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
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
        "deepCAM": "deepcam.out",
        "nanoGPT": "nanoGPT.out",
        "MILC": "milc.out",
    }

    data_rows = []
    
    folder_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-job(\d+)$")

    for app_name, app_log_name in apps_info.items():
        app_dir = os.path.join(base_path, app_name, "64nodes")
        if not os.path.isdir(app_dir):
            continue

        # Get subdirectories like 2024-12-30_16-14-51-job34357636
        folders = os.listdir(app_dir)
        folders.sort()  # Sort folders to process in order
        for run_folder in folders:
            # Check if run_folder matches the expected pattern
            match = folder_pattern.match(run_folder)
            print(app_name, run_folder)
            if not match:
                continue
            
            # Extract date from folder name
            folder_date = match.group(1)  # This gets the YYYY-MM-DD part
            
            # Skip if date is before 2025-02-08
            if app_name == "AMG2023" and folder_date < "2025-02-08":
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

            # Application log
            app_out_file = os.path.join(sub_path, app_log_name)

            # 4) Parse runtime
            if app_name == "AMG2023":
                runtime = parse_runtime_amg(app_out_file)
            elif app_name == "nanoGPT":
                runtime = parse_runtime_nanogpt(app_out_file)
            elif app_name == "MILC":
                runtime = parse_runitme_milc(app_out_file)
            else:
                # deepCAM
                runtime = parse_runtime_deepcam(app_out_file)

            # Prepare row record for this run
            row = {}
            row["app_name"] = app_name        # Application name
            row["run_time"] = run_time_str    # Time parsed from folder name
            row["job_id"] = job_id_str        # Job ID parsed from folder name

            row["runtime"] = runtime

            data_rows.append(row)

    # Assemble final CSV column order
    header = [
        "app_name",
        "run_time",
        "job_id",
    ]

    # Last column: runtime
    header.append("runtime")

    # Write CSV
    # You can change the filename as needed, e.g., to combined.csv
    out_csv = "all_pm.csv"
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

