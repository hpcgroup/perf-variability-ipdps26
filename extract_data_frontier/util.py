import os
import re
from datetime import datetime

def verify_app_dir(dirname, app, nodes):
    dir_patterns = {
        "AMG2023": re.compile(r"amg-\d{7}"),
        "MILC": re.compile(f"milc_40.{nodes}-"+r"\d{7}"),
        "deepcam": re.compile(r"deepcam-\d{7}"),
        "nanoGPT": re.compile(r"nanogpt-\d{7}"),
    }
    return bool(dir_patterns[app].match(dirname))

def parse_job_id(dirname, app, nodes):
    patterns = {
        "AMG2023": re.compile(r"amg-(\d{7})"),
        "MILC": re.compile(f"milc_40.{nodes}-"+r"(\d{7})"),
        "deepcam": re.compile(r"deepcam-(\d{7})"),
        "nanoGPT": re.compile(r"nanogpt-(\d{7})"),
    }
    match = patterns[app].match(dirname)
    if match:
        return match.group(1)
    return None

def parse_app_time(dirname, app, nodes, use_perf_counter_run=False):
    if app == 'AMG2023':
        t = parse_amg_out(os.path.join(dirname, 'output-AMG2023.log'))
    elif app == "MILC":
        t = parse_milc_out(os.path.join(dirname, 'output-MILC.log'))
    elif app == "nanoGPT":
        filepath = os.path.join(dirname, 'output-nanoGPT.log')
        if use_perf_counter_run:
            filepath = os.path.join(dirname, 'output-nanoGPT-with_performance_counters.log')
        t = parse_nanogpt_out(filepath, niters=30)
    elif app == "deepcam":
        filepath = os.path.join(dirname, 'output-deepcam.log')
        if use_perf_counter_run:
            filepath = os.path.join(dirname, 'output-deepcam-with_performance_counters.log')
        t = parse_deepcam_out(filepath, nsteps=470)
    return t


def parse_timestamp(file, app):
    """
    Parse a run directory name, e.g., "2024-12-26_12-22-14-job34298015",
    into a datetime object, assuming the format: YYYY-MM-DD_hh-mm-ss-jobxxxx.
    """
    pattern = re.compile(f"start {app}: " + r"(.*)\n")
    
    if not os.path.isfile(file):
        return None
    
    with open(file, 'r') as f:
        fcontent = f.read()
        match = pattern.search(fcontent)
        if match:
            date_time_str = match.group(1)  # Extract the matched date string
            try:
                # Parse the date string into a datetime object
                date_time_obj = datetime.strptime(date_time_str, "%a %d %b %Y %I:%M:%S %p %Z")
                return date_time_obj
            except ValueError as e:
                try:
                    date_time_obj = datetime.strptime(date_time_str, "%a %b %d %I:%M:%S %p %Z %Y")
                    return date_time_obj
                except ValueError as e:
                    return None
    return None

def parse_amg_out(filepath):
    """
    Parse the amg.out file in AMG2023:
    Find a line containing "GMRES Solver" and extract the wall clock time (in seconds)
    from the following line (e.g., "wall clock time = XXX seconds").
    """
    if os.path.exists(filepath):
        pattern = re.compile(r"wall clock time\s*=\s*([0-9\.]+)")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "GMRES Solver" in line:
                if i + 1 < len(lines):
                    match = pattern.search(lines[i + 1])
                    if match:
                        return float(match.group(1))
    return None

def parse_milc_out(filepath):
    """
    Parse the milc.out file in MILC:
    Search for a line like "Time = 1.563461e+02 seconds" and extract the time value (seconds).
    """
    if os.path.exists(filepath):
        pattern = re.compile(r"Time\s*=\s*([0-9.e\+]+)\s*seconds")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    return float(match.group(1))
    return None

def parse_deepcam_out(filepath, nsteps=470):
    """
    Parse the deepcam.out file in deepCAM:
    Sum up values from lines like "step XXXX: time YYYms" (milliseconds) and return total time in seconds.
    """
    total_time_ms = 0.0
    step_num = 1
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(f"step {step_num}:" + r"\s*time\s+([0-9\.]+)ms", line)
                if match:
                    total_time_ms += float(match.group(1))
                    step_num += 1
                
                # stop after reaching n steps
                if step_num == nsteps:
                    break

        if step_num == nsteps:
            return total_time_ms / 1000.0
    return None

def parse_nanogpt_out(filepath, niters=30):
    """
    Parse the nanoGPT.out file in nanoGPT:
    Sum up times from lines such as "iter xx: ... time ZZZZms ..." (milliseconds) and return total time in seconds.
    """
    total_time_ms = 0.0
    iter_num = 1
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(f"iter {iter_num}:.*" + r"time\s+([0-9\.]+)ms", line)
                if match:
                    total_time_ms += float(match.group(1))
                    iter_num += 1

                # stop after reaching n steps
                if iter_num == niters:
                    break

        if iter_num == niters:
            return total_time_ms / 1000.0
    return None