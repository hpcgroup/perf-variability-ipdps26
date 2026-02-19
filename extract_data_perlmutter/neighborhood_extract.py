#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import subprocess
import pandas as pd
from datetime import datetime
from dateutil import parser  # pip install python-dateutil
from collections import defaultdict
import concurrent.futures

APPS = "nanoGPT"
# APPS = "AMG2023"
# APPS = "deepCAM"
# APPS = "MILC"
# 1. Set up some global regex patterns and paths
NODES = 64
base_path = f"/pscratch/sd/c/cunyang/result/{APPS}/{NODES}nodes"

folder_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-job(\d+)')
nanoGPT_time_regex = re.compile(r'time\s+([\d.]+)ms')


def parse_node_id(node_str):
    """Parse the numeric part in node name and return an integer. If no match, return None"""
    match = re.search(r'(\d+)$', node_str)
    if match:
        return int(match.group(1))
    return None

def get_dragonfly_group_id(node_str):
    """Determine group ID by parsing node number, then using node_id // 128"""
    node_id = parse_node_id(node_str)
    if node_id is None:
        return None
    return node_id // 128

def parse_sacct_time(time_str):
    """Convert time string from sacct output to datetime"""
    return parser.parse(time_str)

def run_sacct_command(cmd):
    """Run sacct command and return a list of result lines."""
    try:
        output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        lines = output.split("\n")
        return lines
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] sacct command execution failed: {cmd}")
        return []

def get_nanoGPT_total_time(nano_file):
    """Extract all iteration times from nanoGPT.out and accumulate them."""
    total_time_ms = 0.0
    with open(nano_file, 'r') as nf:
        for line in nf:
            m = nanoGPT_time_regex.search(line)
            if m:
                try:
                    total_time_ms += float(m.group(1))
                except ValueError:
                    pass
    return total_time_ms

def get_amg_total_time(amg_file):
    total_time_ms = 0.0
    with open(amg_file, "r") as f:
        for line in f:
            
            if "GMRES Solver:" in line:
                next_line = next(f)
                match = re.search(r"wall clock time = ([\d.]+)", next_line)
                if match:
                    total_time_ms = float(match.group(1))
                break
    return total_time_ms

def get_deepcam_total_time(deepcam_file):
    print(deepcam_file)
    pattern = re.compile(r"step\s+\d+:\s+time\s+([0-9.]+)ms")
    total_time_s = 0.0
    with open(deepcam_file, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ms_val = float(m.group(1))
                total_time_s += (ms_val / 1000.0)
    return total_time_s

def get_milc_total_time(milc_file):

    pattern = re.compile(r"Time\s*=\s*([0-9.e\+]+)\s*seconds")
    with open(milc_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None

def parse_sacct_job_record(line):
    """
    Parse sacct line, return a dict of (JobID,User,JobName,AllocNodes,NodeList,Start,End)
    """
    items = line.split('|')
    if len(items) < 7:
        return None
    return {
        'JobID': items[0],
        'User': items[1],
        'JobName': items[2],
        'AllocNodes': items[3],
        'NodeList': items[4],
        'Start': items[5],
        'End': items[6],
    }

def get_job_start_end_nodelist(job_id):
    """Get Start, End, NodeList of the actual running subjob (<job_id>.2)"""
    cmd = f"sacct -j {job_id}.2 --format=JobName,Start,End,NodeList --parsable2 --noheader"
    lines = run_sacct_command(cmd)
    if not lines:
        return None, None, []
    
    for line in lines:
        parts = line.split('|')
        if len(parts) < 4:
            continue
        job_name, start_str, end_str, node_list_str = parts
        start_dt = parse_sacct_time(start_str)
        end_dt = parse_sacct_time(end_str)
        node_list = []
        if '[' in node_list_str and ']' in node_list_str:
            node_list.append(node_list_str)
        else:
            node_list = node_list_str.split(',')
        return start_dt, end_dt, node_list
    
    return None, None, []

def expand_node_list(node_list_str):
    """Expand node strings formatted like 'nid[00032-00034,00036]'."""
    if '[' not in node_list_str or ']' not in node_list_str:
        return node_list_str.split(',')
    
    match = re.match(r'(\w+)\[(.*)\]', node_list_str)
    if not match:
        return [node_list_str]
    
    prefix = match.group(1)
    bracket_content = match.group(2)
    result = []
    segments = bracket_content.split(',')
    for seg in segments:
        if '-' in seg:
            start, end = seg.split('-')
            padding = len(start)
            for i in range(int(start), int(end) + 1):
                result.append(f"{prefix}{i:0{padding}d}")
        else:
            result.append(f"{prefix}{seg}")
    return result

def is_time_overlapped(start1, end1, start2, end2):
    """Check if there is an overlap between [start1, end1] and [start2, end2]."""
    return not (end1 <= start2 or end2 <= start1)

def compute_min_max_avg_group_occupancy(job_groups, concurrent_jobs):
    """Consistent with previous implementation, calculate the min/max/avg occupancy rate of groups where the job is located."""
    GROUP_SIZE = 128
    from collections import defaultdict
    occupancy_count = defaultdict(int)
    for job in concurrent_jobs:
        for nd in job['NodeList']:
            grp = get_dragonfly_group_id(nd)
            if grp in job_groups:
                occupancy_count[grp] += 1
    
    if not job_groups:
        return (0.0, 0.0, 0.0)
    
    ratios = []
    for grp in job_groups:
        occupied = occupancy_count.get(grp, 0)
        ratio = occupied / GROUP_SIZE
        ratios.append(ratio)
    if not ratios:
        return (0.0, 0.0, 0.0)
    return (min(ratios), max(ratios), sum(ratios)/len(ratios))

def get_total_concurrent_nodes_in_group(app_groups, concurrent_jobs):
    unique_nodes = set()
    for job in concurrent_jobs:
        for nd in job['NodeList']:
            grp_id = get_dragonfly_group_id(nd)
            if grp_id in app_groups:
                unique_nodes.add(nd)
    return len(unique_nodes)

def get_total_concurrent_nodes(concurrent_jobs):
    unique_nodes = set()
    for job in concurrent_jobs:
        unique_nodes.update(job.get('NodeList', []))
    return len(unique_nodes)

def process_folder(folder_name, APPS):
    """
    Logic for parallel processing of a single folder.
    If conditions are not met or content cannot be obtained, return None; otherwise, return a dict record.
    """
    if APPS == "AMG2023":
        if folder_name[:19] < "2025-02-09_20-32-38":
            return None
    elif APPS == "nanoGPT":
        if folder_name[:19] >= "2025-04-10_00-00-00":
            return None
    elif APPS == "deepCAM":
        if folder_name[:19] >= "2025-04-10_00-00-00":
            return None
    elif APPS == "MILC":
        if folder_name[:19] >= "2025-04-10_00-00-00":
            return None
    print(f"[INFO] Processing folder: {folder_name}")
    match = folder_pattern.match(folder_name)
    if not match:
        return None
    
    job_id = match.group(1)  # sacct job number
    job_folder = os.path.join(base_path, folder_name)
    if not os.path.isdir(job_folder):
        return None
    
    if APPS == "nanoGPT":
        app_file = os.path.join(job_folder, "nanoGPT.out")
    elif APPS == "AMG2023":
        app_file = os.path.join(job_folder, "amg.out")
    elif APPS == "deepCAM":
        app_file = os.path.join(job_folder, "deepcam.out")
    elif APPS == "MILC":
        app_file = os.path.join(job_folder, "milc.out")
    else:
        return None
    if not os.path.exists(app_file):
        return None
    
    # 1) Get the total runtime of this app
    if APPS == "nanoGPT":
        total_time_ms = get_nanoGPT_total_time(app_file)
    elif APPS == "AMG2023":
        total_time_ms = get_amg_total_time(app_file)
    elif APPS == "deepCAM":
        total_time_ms = get_deepcam_total_time(app_file)
    elif APPS == "MILC":
        total_time_ms = get_milc_total_time(app_file)
    if total_time_ms == 0:
        return None
    
    # 2) Get job start/end time & nodes
    start_dt, end_dt, app_nodes_raw = get_job_start_end_nodelist(job_id)
    if start_dt is None or end_dt is None:
        return None
    
    # Expand app's own node list
    app_nodes_expanded = []
    for node_str in app_nodes_raw:
        app_nodes_expanded.extend(expand_node_list(node_str))
    
    # Collect all groups where the app is located
    app_groups = set()
    for nd in app_nodes_expanded:
        g = get_dragonfly_group_id(nd)
        if g is not None:
            app_groups.add(g)

    # 3) Query concurrent jobs with sacct
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
    end_str   = end_dt.strftime("%Y-%m-%dT%H:%M:%S")
    sacct_cmd = (
        f"sacct --allusers -S {start_str} -E {end_str} "
        f"--format=JobID,User,JobName,AllocNodes,NodeList,Start,End -T --nnodes='2-1800' "
        f"--parsable2 --noheader"
    )
    # print(f"[INFO] Running command: {sacct_cmd}")
    lines = run_sacct_command(sacct_cmd)
    # print(lines)
    # Filter
    # lines = [ln for ln in lines if '.' not in ln.split('|')[0]]
    lines = [ln for ln in lines if not (ln.split('|')[0].endswith('.extern') or ln.split('|')[0].endswith('.batch'))]
    # print(f"[INFO] Filtered lines: {len(lines)}")
    # print(lines)
    filtered_lines = []
    for ln in lines:
        parts = ln.split('|')
        if len(parts) < 7:
            continue
        # Remove None/Unknown
        if any(x in parts[-3:] for x in ("None", "Unknown")):
            continue
        # Remove NodeList containing "login"
        if "login" in parts[4]:
            continue
        filtered_lines.append(ln)
    lines = filtered_lines
    
    concurrent_jobs = []
    
    # Additional storage: user -> list[ {jobID, alloc_nodes, group_set}, ... ]
    user_job_details = defaultdict(list)
    user_alloc_dict = defaultdict(int)
    for ln in lines:
        record = parse_sacct_job_record(ln)
        if not record:
            continue
        # Remove self job and subjobs
        if record['JobID'].startswith(job_id):
            continue
        if '.' in record['JobID']:
            continue
        
        # Time overlap
        other_start = parse_sacct_time(record['Start'])
        other_end = parse_sacct_time(record['End'])
        if not is_time_overlapped(start_dt, end_dt, other_start, other_end):
            continue
        
        # Node expansion
        node_list_expanded = expand_node_list(record['NodeList'])
        alloc_nodes = int(record['AllocNodes']) if record['AllocNodes'].isdigit() else 0
        
        # Collect groups for this job
        job_groups = set()
        for nd in node_list_expanded:
            grp = get_dragonfly_group_id(nd)
            if grp is not None:
                job_groups.add(grp)
        
        # Store concurrent job
        concurrent_jobs.append({
            'JobID': record['JobID'],
            'User': record['User'],
            'JobName': record['JobName'],
            'AllocNodes': alloc_nodes,
            'NodeList': node_list_expanded,
            'Groups': job_groups,
        })
        
        # Add to user_alloc_dict
        user_alloc_dict[record['User']] += alloc_nodes
        
        # Store grouping information for this job
        user_job_details[record['User']].append({
            'job_id': record['JobID'],
            'alloc_nodes': alloc_nodes,
            'groups': job_groups
        })
    
    # 4) Compute concurrency statistics
    # print(concurrent_jobs)
    # total_concurrent_nodes = sum(j['AllocNodes'] for j in concurrent_jobs)
    total_concurrent_nodes = get_total_concurrent_nodes(concurrent_jobs)
    total_concurrent_nodes_in_group = get_total_concurrent_nodes_in_group(app_groups, concurrent_jobs)
    # total_concurrent_nodes = get_total_concurrent_nodes_in_group(concurrent_jobs)
    # print(f"[INFO] Total concurrent nodes: {total_concurrent_nodes}", sum(j['AllocNodes'] for j in concurrent_jobs))
    # total_concurrent_nodes_in_group = 0
    # for job in concurrent_jobs:
    #     for nd in job['NodeList']:
    #         grp_id = get_dragonfly_group_id(nd)
    #         if grp_id in app_groups:
    #             total_concurrent_nodes_in_group += 1
    
    min_grp_ratio, max_grp_ratio, avg_grp_ratio = compute_min_max_avg_group_occupancy(app_groups, concurrent_jobs)
    num_concurrent_jobs = len(concurrent_jobs)

    # 5) Build user_alloc_dict_trimmed
    #    Remove concurrent jobs that only run in "a single group with no intersection with app_groups"
    user_alloc_dict_trimmed = defaultdict(int)

    
    for user, job_list in user_job_details.items():
        # Iterate through each job of this user
        for job_info in job_list:
            job_alloc = job_info['alloc_nodes']
            job_groups = job_info['groups']

            # Check if only in one group:
            if len(job_groups) == 1:
                # Get that sole group
                sole_group = next(iter(job_groups))
                # If sole_group not in app_groups, it's completely in another group
                # -> We exclude this job
                if sole_group not in app_groups:
                    # skip
                    continue
                else:
                    # Otherwise keep it
                    user_alloc_dict_trimmed[user] += job_alloc
            else:
                # If job_groups >= 2, we consider it affects the entire network, keep it
                user_alloc_dict_trimmed[user] += job_alloc
    
    # 6) Summarize results
    return {
        'JobID': job_id,
        'app_total_time_ms': total_time_ms,
        'app_start': start_dt,
        'app_end': end_dt,
        'num_concurrent_jobs': num_concurrent_jobs,
        'total_concurrent_nodes': total_concurrent_nodes,
        'concurrent_nodes_in_same_group': total_concurrent_nodes_in_group,
        'min_group_ratio': min_grp_ratio,
        'max_group_ratio': max_grp_ratio,
        'avg_group_ratio': avg_grp_ratio,
        'user_alloc_dict': dict(user_alloc_dict),
        'user_alloc_dict_trimmed': dict(user_alloc_dict_trimmed)
    }

def main():
    results = []
    folders = os.listdir(base_path)
    # print(folders)
    # folders = folders[0]
    # print(folders)
    
    # process_folder(folders, APPS)

    # Use multiprocessing pool (16 workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        # map will pass each folder in the folders list to process_folder
        futures = []
        for folder_name in folders:
            futures.append(executor.submit(process_folder, folder_name, APPS))
        
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res is not None:
                results.append(res)
    df = pd.DataFrame(results)
    csv_path = APPS + "_performance_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Analysis results saved to {csv_path}")
    
    # Simple correlation analysis
    numeric_cols = [
        'app_total_time_ms',
        'total_concurrent_nodes',
        'concurrent_nodes_in_same_group',
        'min_group_ratio', 
        'max_group_ratio', 
        'avg_group_ratio'
    ]
    if not df.empty:
        corr_matrix = df[numeric_cols].corr()
        print("Correlation matrix:\n", corr_matrix)
        corr_matrix.to_csv(APPS + "_correlation_matrix.csv")
    else:
        print("[INFO] DataFrame is empty.")


if __name__ == "__main__":
    main()

