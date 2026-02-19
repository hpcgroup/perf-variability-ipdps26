import os
import re
import datetime
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

DATA_ROOT = Path("/pscratch/sd/c/cunyang/result/AMG2023/64nodes")

DATE_THRESHOLD = datetime.datetime(2025, 2, 8)

DIRNAME_PATTERN = re.compile(
    r'^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})-job\d+'
)

AMG_PATTERN = re.compile(
    r'GMRES Solver:'
)

COUNTER_START = "MPICH Slingshot CXI Counter Summary:"
COUNTER_END   = "CXI_COUNTER_DATA"

# counter_name, samples, min, min(/s), mean, mean(/s), max, max(/s)
COUNTER_PATTERN = re.compile(
    r'^(\S+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s*$'
)


def parse_dir_datetime(dirname: str):
    """
    # Based on the directory name format like 2024-12-26_12-22-14-jobxxxxx,
    # extract the datetime object for comparison with DATE_THRESHOLD.
    """
    m = DIRNAME_PATTERN.match(dirname)
    if not m:
        return None
    date_str = m.group('date')  # e.g. "2024-12-26"
    time_str = m.group('time')  # e.g. "12-22-14"
    try:
        # Replace '-' in time_str with ':' for parsing
        dt_str = date_str + " " + time_str.replace('-', ':')
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return dt
    except ValueError:
        return None


def parse_amg_out(amg_file: Path):
    """
    """
    time_list = []

    counters_section = False
    counters_data = {}  # Store parsed counters, key=counter_name, value=dict

    with amg_file.open('r') as f:
        for line in f:
            line_stripped = line.strip()
            # 1) Determine the counter parsing section
            if COUNTER_START in line_stripped:
                # Found the start of the counter summary
                counters_section = True
                continue
            # if COUNTER_END in line_stripped:
            #     # Counters end here
            #     counters_section = False
            #     continue
            
            
            
            if counters_section:
                
                # Match COUNTER_PATTERN for lines within the counter section
                cm = COUNTER_PATTERN.match(line_stripped)
                if cm:
                    c_name = cm.group(1)
                    samples = int(cm.group(2))
                    c_min = float(cm.group(3))
                    c_min_per_s = float(cm.group(4))
                    c_mean = float(cm.group(5))
                    c_mean_per_s = float(cm.group(6))
                    c_max = float(cm.group(7))
                    c_max_per_s = float(cm.group(8))
                    counters_data[c_name] = {
                        'samples': samples,
                        'min': c_min,
                        'min_per_s': c_min_per_s,
                        'mean': c_mean,
                        'mean_per_s': c_mean_per_s,
                        'max': c_max,
                        'max_per_s': c_max_per_s
                    }
                # else: Could be header lines or empty lines, skip
                continue

            

    # If 1048576 or 1024 appear multiple times in the file, decide whether to take the average or the last occurrence
    with amg_file.open('r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "GMRES Solver:" in line:
                # Next line
                if i+1 < len(lines):
                    next_line = lines[i+1].strip()
                    # Regex extract 329.502154
                    match = re.search(r"wall clock time\s*=\s*([\d\.]+)", next_line)
                    if match:
                        time_list.append(float(match.group(1)))
                break
    return time_list, counters_data

def main():
    records = []  # Used to aggregate all results

    # Iterate through all subdirectories under the root directory
    for subdir in DATA_ROOT.iterdir():
        if not subdir.is_dir():
            continue
        # Check directory name format & if it's after the date threshold
        dt = parse_dir_datetime(subdir.name)
        if dt is None:
            continue
        if dt < DATE_THRESHOLD:
            continue  # Skip directories before Feb 8th
        if dt == datetime.datetime(2025, 2, 9, 20, 32, 38):
            continue

        # Find the amg.out file in this directory
        amg_file = subdir / "amg.out"
        if not amg_file.exists():
            continue
        
        time_list, counters_data =  parse_amg_out(amg_file)
        
        # use the first time value if available, otherwise assign None
        row = {
            'run_dir': subdir.name,
            'datetime': dt,
            'time': time_list[0],
        }
        # Add all parsed counters from counters_data to the row
        # To avoid conflicts, column names can be prefixed like "counter_" + counter_name + "_mean", etc.
        for c_name, c_dict in counters_data.items():
            row[f"{c_name}_mean"] = c_dict['mean']
            # If you also want to add min, max, mean_per_s, etc., they can be included here
            row[f"{c_name}_min"] = c_dict['min']
            row[f"{c_name}_max"] = c_dict['max']
            # row[f"{c_name}_mean_per_s"] = c_dict['mean_per_s']
            # ...

        records.append(row)

    # # After collecting everything, convert to DataFrame
    df = pd.DataFrame(records)

    # Define base counter names needed for derived metrics
    non_posted_blocked = 'parbs_tarb_pi_non_posted_blocked_cnt'
    non_posted_pkts = 'parbs_tarb_pi_non_posted_pkts'
    posted_blocked = 'parbs_tarb_pi_posted_blocked_cnt'
    posted_pkts = 'parbs_tarb_pi_posted_pkts'

    # Calculate derived counters for mean, max, min if base counters exist
    for stat in ['mean', 'max', 'min']:
        non_posted_blocked_col = f"{non_posted_blocked}_{stat}"
        non_posted_pkts_col = f"{non_posted_pkts}_{stat}"
        posted_blocked_col = f"{posted_blocked}_{stat}"
        posted_pkts_col = f"{posted_pkts}_{stat}"

        # Check if necessary columns exist before calculation
        if non_posted_blocked_col in df.columns and non_posted_pkts_col in df.columns:
            # Calculate non-posted congestion rate, handle division by zero
            df[f'parbs_tarb_pi_non_posted_cong_rate_{stat}'] = df[non_posted_blocked_col].divide(df[non_posted_pkts_col]).fillna(0)
            # Replace potential infinity values with 0 as well (if numerator > 0 and denominator = 0)
            df[f'parbs_tarb_pi_non_posted_cong_rate_{stat}'].replace([float('inf'), -float('inf')], 0, inplace=True)


        if posted_blocked_col in df.columns and posted_pkts_col in df.columns:
            # Calculate posted congestion rate, handle division by zero
            df[f'parbs_tarb_pi_posted_cong_rate_{stat}'] = df[posted_blocked_col].divide(df[posted_pkts_col]).fillna(0)
            # Replace potential infinity values with 0 as well
            df[f'parbs_tarb_pi_posted_cong_rate_{stat}'].replace([float('inf'), -float('inf')], 0, inplace=True)


    # Handle potential missing values (fill with 0) AFTER calculating derived metrics
    # This ensures that if a base counter was missing (NaN), the derived counter also becomes NaN initially, then gets filled
    for col in list(df.columns):
        # Check if column is numeric before filling NaN
        if pd.api.types.is_numeric_dtype(df[col]):
            num_missing = df[col].isna().sum()
            if num_missing > 0:
                # print(f"Column {col} has {num_missing} missing values. Filling with 0.") # Optional debug info
                df[col].fillna(0, inplace=True)
        # Optional: Drop columns with too many missing values (consider carefully)
        # if num_missing > threshold:
        #     df.drop(columns=col, inplace=True)

    # Define function for formatted correlation output
    def print_formatted_correlations(df, cols, stat_type):
        print(f"\n=== {stat_type.capitalize()} Value Correlations (sorted by Spearman) ===")
        
        # Create a list to store all results
        results = []
        corr_df_list = []
        # Calculate correlations and store results
        for col in cols:
            # Remove the suffix for display name
            display_name = col.replace(f"_{stat_type}", "")
            
            pearson_r, pearson_p = pearsonr(df['time'], df[col])
            spearman_r, spearman_p = spearmanr(df['time'], df[col])
            results.append((display_name, pearson_r, pearson_p, spearman_r, spearman_p))
            
            # Add to correlation dataframe
            corr_df_list.append({
                'counter': display_name,
                'stat_type': stat_type,
                'pearson': pearson_r,
                'pearson_p': pearson_p,
                'spearman': spearman_r,
                'spearman_p': spearman_p
            })
        
        # Sort by Spearman correlation coefficient
        results.sort(key=lambda x: x[3], reverse=True)
        
        # Format header
        print(f"{'Counter':<40} {'Pearson':>10} {'P-value':>10} {'Spearman':>10} {'P-value':>10}")
        print('-' * 80)
        
        # Print results
        for name, p_r, p_p, s_r, s_p in results:
            print(f"{name:<40} {p_r:>10.6f} {p_p:>10.6f} {s_r:>10.6f} {s_p:>10.6f}")
        return pd.DataFrame(corr_df_list)

    # Process '_mean' columns
    # The derived columns ending in '_mean' will now be automatically included
    counter_cols = [c for c in df.columns if c.endswith('_mean')]
    if counter_cols: # Check if there are any mean columns before proceeding
        corr_df = print_formatted_correlations(df, counter_cols, "mean")
        corr_df.to_csv("AMG2023_mean.csv", index=False)
    else:
        print("No '_mean' columns found for correlation analysis.")

    # Process '_max' columns
    # The derived columns ending in '_max' will now be automatically included
    counter_cols = [c for c in df.columns if c.endswith('_max')]
    if counter_cols:
        corr_df = print_formatted_correlations(df, counter_cols, "max")
        corr_df.to_csv("AMG2023_max.csv", index=False)
    else:
        print("No '_max' columns found for correlation analysis.")

    # Process '_min' columns
    # The derived columns ending in '_min' will now be automatically included
    counter_cols = [c for c in df.columns if c.endswith('_min')]
    if counter_cols:
        corr_df = print_formatted_correlations(df, counter_cols, "min")
        corr_df.to_csv("AMG2023_min.csv", index=False)
    else:
        print("No '_min' columns found for correlation analysis.")

    # # 若要把结果输出到 CSV：
    # # df.to_csv("allreduce_counters_summary.csv", index=False) # Consider saving the full df
    # df.to_csv("AMG_counters_full_data_with_derived.csv", index=False) # Example: Save full data


if __name__ == "__main__":
    main()

