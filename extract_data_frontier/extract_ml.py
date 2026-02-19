#!/usr/bin/env python3

import os
import re
import json
from datetime import datetime
import pandas as pd
import argparse
from hta.trace_analysis import TraceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed

# skipped_jobs
skipped_jobs = {
}

pd.set_option('display.max_rows', 20)

def process_job(job_dir, app_name, base_dir, df_dir):
    print(f"Starting job: {job_dir}")
    if job_dir in skipped_jobs:
        print(f"Skipping job: {job_dir}")
        return

    full_job_dir = os.path.join(base_dir, job_dir)
    trace_dir = os.path.join(full_job_dir, "torchprof")

    # Skip if torch_profiler directory does not exist
    if not os.path.isdir(trace_dir):
        return

    dt_str = None
    job_file = os.path.join(full_job_dir, f'output-{app_name}.log')
    if os.path.isfile(job_file):
        with open(job_file, "r") as f:
            lines = f.read()
            match = re.search(f"start {app_name}: (.*)\n", lines)
            if match:
                dt_str = match.group(1)  # Extract the matched date string
                try:
                    # Parse the date string into a datetime object
                    date_time_obj = datetime.strptime(dt_str, "%a %d %b %Y %I:%M:%S %p %Z")
                    dt_str = str(date_time_obj)
                except ValueError as e:
                    try:
                        date_time_obj = datetime.strptime(dt_str, "%a %b %d %I:%M:%S %p %Z %Y")
                        dt_str = str(date_time_obj)
                    except ValueError as e:
                        dt_str = None
            else:
                dt_str = None
    csv_filename = dt_str.replace(' ', '_') + ".csv"
    csv_path = os.path.join(df_dir, csv_filename)
    if os.path.exists(csv_path):
        print(f"[INFO] {csv_filename} already exists. Skip.")
        return
    elif not dt_str:
        print(f"[INFO] job not successful. Skip.")
        return

    # 3. HTA analysis
    analyzer = TraceAnalysis(trace_dir=trace_dir)
    _, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown(visualize=False)

    if kernel_metrics_df.empty:
        print(f"[WARN] No kernel metrics found in {job_dir}, skipping.")
        return

    # 4. Group by name and aggregate sum (us), max (us), min (us)
    df_agg = (
        kernel_metrics_df
        .groupby('name')
        .agg({
            'sum (us)': 'sum',   # Sum of sum(us) for all ranks
            'max (us)': 'sum',   # Or use 'max' to see the maximum value
            'min (us)': 'sum',   # Or use 'min' to see the minimum value
        })
        .reset_index()
    )

    # Convert microseconds to seconds
    df_agg[['sum (us)', 'max (us)', 'min (us)']] = (
        df_agg[['sum (us)', 'max (us)', 'min (us)']] / 1e6
    )

    # Rename columns
    df_agg.rename(columns={
        'sum (us)': 'sum_time(s)',
        'max (us)': 'max_time(s)',
        'min (us)': 'min_time(s)'
    }, inplace=True)

    # 5. Save to CSV
    # Use the first 19 characters of the directory name (timestamp) as the filename
    df_agg.to_csv(csv_path, index=False)
    print(f"[INFO] Saved aggregated DataFrame to {csv_path}")

def main(app_name, base_dir):
    # Prepare directory to store DataFrame
    df_dir = os.path.join(base_dir, "df_rccl_breakdown")
    os.makedirs(df_dir, exist_ok=True)

    # 1. Find all directories matching the pattern deepcam-2870573
    job_dirs = sorted(
        [
            d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d)) 
            and re.match(app_name.lower()+r"-\d{7}", d)
        ],
        key=lambda x: x
    )
    print(job_dirs)

    for job_dir in job_dirs:
        process_job(job_dir, app_name, base_dir, df_dir)

    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_job, job_dir, app_name, base_dir, df_dir) for job_dir in job_dirs]
    #     for future in as_completed(futures):
    #         future.result()  # To raise any exceptions that occurred during processing

def parse_args():
    parser = argparse.ArgumentParser(description="Parse application name and base directory.")
    parser.add_argument("--app_name", type=str, required=True, help="Name of the application.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for the application.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.app_name, args.base_dir)
