#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as font_manager
import seaborn as sns

font_path = './gillsans.ttf'
font_dis_prop = font_manager.FontProperties(fname=font_path, size=22)
font_tick_prop = font_manager.FontProperties(fname=font_path, size=22) # Smaller font for ticks if needed

system = 'perlmutter'
NAME = "Perlmutter"
# system = 'frontier'
# NAME = "Frontier"
save_dir = os.getcwd()

colors = ['#D55E00', '#0072B2', '#009E73', '#800080', '#CC79A7', '#E69F00', '#56B4E9']
# hatches = ['x', 'xxx', '\\\\', '||','///', '+', 'o', '.', '*', '-', 'ooo', '+++', 'xx', '/', '\\', '***']

def plot_routine_violin(runs, routine1_name, routine2_name, runs2, routine3_name, routine4_name):

    app1_name = "AMG2023"
    app2_name = "MILC"

    plot_data = []
    app1_labels = []
    app2_labels = []

    # --- Extract data for App 1 ---
    runtimes1 = [run_data['mpi_calls'].get(routine1_name, 0.0) for _, run_data in runs]
    runtimes2 = [run_data['mpi_calls'].get(routine2_name, 0.0) for _, run_data in runs]
    valid_runtimes1 = [t for t in runtimes1 if t > 1e-9]
    valid_runtimes2 = [t for t in runtimes2 if t > 1e-9]
    label1 = f"{app1_name}_{routine1_name}"
    label2 = f"{app1_name}_{routine2_name}"
    if valid_runtimes1:
        plot_data.extend([(t, routine1_name, app1_name, label1) for t in valid_runtimes1])
        app1_labels.append(label1)
    if valid_runtimes2:
        plot_data.extend([(t, routine2_name, app1_name, label2) for t in valid_runtimes2])
        app1_labels.append(label2)

    # --- Extract data for App 2 ---
    runtimes3 = [run_data['mpi_calls'].get(routine3_name, 0.0) for _, run_data in runs2]
    runtimes4 = [run_data['mpi_calls'].get(routine4_name, 0.0) for _, run_data in runs2]
    valid_runtimes3 = [t for t in runtimes3 if t > 1e-9]
    valid_runtimes4 = [t for t in runtimes4 if t > 1e-9]
    label3 = f"{app2_name}_{routine3_name}"
    label4 = f"{app2_name}_{routine4_name}"
    if valid_runtimes3:
        plot_data.extend([(t, routine3_name, app2_name, label3) for t in valid_runtimes3])
        app2_labels.append(label3)
    if valid_runtimes4:
        plot_data.extend([(t, routine4_name, app2_name, label4) for t in valid_runtimes4])
        app2_labels.append(label4)

    # --- Prepare data for Seaborn ---
    if not plot_data:
        print("Error: No data available to plot.")
        return
    df_plot = pd.DataFrame(plot_data, columns=['Time (s)', 'Routine', 'Application', 'PlotLabel'])

    plot_order = [label1, label2, label3, label4]
    existing_labels_in_order = [label for label in plot_order if label in df_plot['PlotLabel'].unique()]
    app1_labels_existing = [label for label in app1_labels if label in existing_labels_in_order]
    app2_labels_existing = [label for label in app2_labels if label in existing_labels_in_order]

    if not existing_labels_in_order:
        print("Error: No routines with data found to plot.")
        return

    # --- Plotting Setup ---
    fig, ax1 = plt.subplots(figsize=(7, 6)) # Slightly wider for two y-axes
    ax2 = ax1.twinx() # Create a twin Axes sharing the xaxis
    # Using a dictionary mapping label to color for consistency
    num_labels_to_plot = len(existing_labels_in_order)
    # palette_map = {label: colors[i % len(colors)] for i, label in enumerate(existing_labels_in_order)}
    palette_map = {}
    for i, label in enumerate(existing_labels_in_order):
        if label == "AMG2023_Allreduce":
            palette_map[label] = colors[2]
        elif label == "MILC_Allreduce":
            palette_map[label] = colors[2]
        elif label == "AMG2023_Waitall":
            palette_map[label] = colors[6]
        elif label == "MILC_Test":
            palette_map[label] = colors[3]
    violin_inner_style = "quartile" # Define inner style

    # --- Plot all violins on ax1 (Left Y-axis) ---
    sns.violinplot(x='PlotLabel', y='Time (s)', data=df_plot, order=existing_labels_in_order,
                   ax=ax1, palette=palette_map, inner=violin_inner_style, cut=0)

    # --- Plot all violins on ax2 (Right Y-axis) ---
    sns.violinplot(x='PlotLabel', y='Time (s)', data=df_plot, order=existing_labels_in_order,
                   ax=ax2, palette=palette_map, inner=violin_inner_style, cut=0)

    # --- Apply hatches and edge colors ---
    print(existing_labels_in_order)
    for i, label in enumerate(existing_labels_in_order):
        if label == "AMG2023_Allreduce":
            color = colors[2]
        elif label == "MILC_Allreduce":
            color = colors[2]
        elif label == "AMG2023_Waitall":
            color = colors[6]
        elif label == "MILC_Test":
            color = colors[3]
        if i < len(ax1.collections):
            ax1.collections[i].set_edgecolor(color) # Set edge color to match face color for hatch visibility
        if i < len(ax2.collections):
            ax2.collections[i].set_edgecolor(color) # Set edge color to match face color

    # --- Hide irrelevant parts ---
    num_violins = len(existing_labels_in_order)
    # Determine number of lines added by 'inner' style (3 for quartile: median, q1, q3)
    num_lines_per_violin = 3 if violin_inner_style == "quartile" else 0 # Adjust if inner changes

    # Hide App2 violins and lines on ax1
    for i, label in enumerate(existing_labels_in_order):
        if label in app2_labels_existing:
            if i < len(ax1.collections): # Hide violin body
                ax1.collections[i].set_visible(False)
            # Hide corresponding lines (quartiles)
            line_start_index = i * num_lines_per_violin
            for j in range(num_lines_per_violin):
                line_index = line_start_index + j
                if line_index < len(ax1.lines):
                    ax1.lines[line_index].set_visible(False)

    # Hide App1 violins and lines on ax2
    for i, label in enumerate(existing_labels_in_order):
        if label in app1_labels_existing:
            if i < len(ax2.collections): # Hide violin body
                ax2.collections[i].set_visible(False)
            # Hide corresponding lines (quartiles)
            line_start_index = i * num_lines_per_violin
            for j in range(num_lines_per_violin):
                line_index = line_start_index + j
                if line_index < len(ax2.lines):
                    ax2.lines[line_index].set_visible(False)


    # --- Customization ---
    ax1.set_xlabel("") # Remove default x-label, info is in ticks now
    ax1.set_ylabel("Time (s)", fontproperties=font_dis_prop) # Label left axis
    ax2.set_ylabel(None)
    # ax2.set_ylabel(f"{app2_name} Time (s)", fontproperties=font_dis_prop) # Label right axis
    ax1.set_title(f"Top 2 most variable routines ({NAME})", fontproperties=font_dis_prop, y=1.03)

    ax1.tick_params(axis='x', length=0) # Hide x-axis tick lines
    # Apply font_tick_prop to y-axis ticks for ax1
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_tick_prop)
    # Apply font_tick_prop to y-axis ticks for ax2
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_tick_prop)

    # Create the *first row* of x-axis labels (Routine names) on ax1
    tick_labels_map_routine = {
        label1: routine1_name, label2: routine2_name,
        label3: routine3_name, label4: routine4_name
    }
    primary_tick_labels = [tick_labels_map_routine.get(label, "N/A") for label in existing_labels_in_order]
    ax1.set_xticks(range(len(existing_labels_in_order)))
    ax1.set_xticklabels(primary_tick_labels, fontproperties=font_tick_prop)
    # y_labels = [f"{y:.0f}" for y in ax1.get_yticks()] # Commented out as direct setting might interfere
    # ax1.set_yticks(y_labels) # Commented out
    # ax1.set_yticklabels(y_labels, fontproperties=font_tick_prop) # Commented out - use loop instead

    # Add the *second row* of x-axis labels (App names) below the first row on ax1
    y_pos_app_label = -0.10
    if app1_labels_existing:
        indices1 = [existing_labels_in_order.index(label) for label in app1_labels_existing]
        x_pos_center1 = (min(indices1) + max(indices1)) / 2.0
        ax1.text(x_pos_center1, y_pos_app_label, f"{app1_name}",
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.get_xaxis_transform(), fontproperties=font_tick_prop)
    if app2_labels_existing:
        indices2 = [existing_labels_in_order.index(label) for label in app2_labels_existing]
        x_pos_center2 = (min(indices2) + max(indices2)) / 2.0
        ax1.text(x_pos_center2, y_pos_app_label, f"{app2_name}",
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.get_xaxis_transform(), fontproperties=font_tick_prop)

    # Add horizontal lines above the second row of x-axis labels (between the two rows)
    y_pos_line = -0.08 # Position the line slightly above the text label

    line_width = 1.5
    line_padding = 0.25 # How much the line extends beyond the center of the outer violins

    if app1_labels_existing:
        indices1 = [existing_labels_in_order.index(label) for label in app1_labels_existing]
        x_start1 = min(indices1) - line_padding
        x_end1 = max(indices1) + line_padding
        # Draw line using plot with axis transform for y coordinate
        ax1.plot([x_start1, x_end1], [y_pos_line, y_pos_line], linewidth=line_width, solid_capstyle='butt', color='black',
                 transform=ax1.get_xaxis_transform(), clip_on=False)

    if app2_labels_existing:
        indices2 = [existing_labels_in_order.index(label) for label in app2_labels_existing]
        x_start2 = min(indices2) - line_padding
        x_end2 = max(indices2) + line_padding
        # Draw line using plot with axis transform for y coordinate
        ax1.plot([x_start2, x_end2], [y_pos_line, y_pos_line], linewidth=line_width, solid_capstyle='butt', color='black',
                 transform=ax1.get_xaxis_transform(), clip_on=False)


    # Add vertical lines for visual separation between applications on ax1
    app_change_index = -1
    if app1_labels_existing and app2_labels_existing:
        try:
            last_app1_idx = max(existing_labels_in_order.index(label) for label in app1_labels_existing)
            first_app2_idx = min(existing_labels_in_order.index(label) for label in app2_labels_existing)
            if first_app2_idx == last_app1_idx + 1:
                app_change_index = last_app1_idx + 0.5
        except ValueError:
            pass # Should not happen if both lists are non-empty
    if app_change_index != -1:
        ax1.axvline(app_change_index, color='grey', linestyle='--', linewidth=1, ymin=0.05, ymax=0.95)

    # Set y-axis limits independently
    max_time_app1 = df_plot[df_plot['Application'] == app1_name]['Time (s)'].max() if app1_labels_existing else 0
    max_time_app2 = df_plot[df_plot['Application'] == app2_name]['Time (s)'].max() if app2_labels_existing else 0

    if NAME == 'Perlmutter':
        ax1.set_ylim(bottom=0, top=140)
        ax1.set_yticks([0, 20, 40, 60, 80, 100, 120, 140])
        ax1.set_yticklabels([0, 20, 40, 60, 80, 100, 120, 140])
        ax2.set_ylim(bottom=0, top=100)
        ax2.set_yticks([0, 20, 40, 60, 80, 100])
        ax2.set_yticklabels([0, 20, 40, 60, 80, 100])
    elif NAME == 'Frontier':
        ax1.set_ylim(bottom=0, top=120)
        ax1.set_yticks([0, 20, 40, 60, 80, 100, 120])
        ax1.set_yticklabels([0, 20, 40, 60, 80, 100, 120])
        ax2.set_ylim(bottom=0, top=250)
        ax2.set_yticks([0, 50, 100, 150, 200, 250])
        ax2.set_yticklabels([0, 50, 100, 150, 200, 250])
    else:
        ax1.set_ylim(bottom=0, top=max_time_app1 * 1.1) # Set top limit to 1.2x Worst time
        ax2.set_ylim(bottom=0, top=max_time_app2 * 1.1) # Set top limit to 1.2x Worst time

    # Ensure grid lines (if any) from ax2 don't obscure ax1 plot
    ax2.grid(False)

    fig.tight_layout() # Use fig.tight_layout()

    # --- Save Plot ---
    routines_str = "_".join(re.sub(r'[^a-zA-Z0-9_]', '', r) for r in primary_tick_labels if r != "N/A")
    filename_base = f"violin_mpi"
    save_path = os.path.join(save_dir, filename_base + f"_{system}.pdf")
    plt.savefig(save_path)
    print(f"Saved dual-axis violin plot to {save_path}")



def load_runs_from_csv(filepath):
    """
    Loads runs data from a CSV file created by save_runs_to_csv
    and reconstructs the original list of tuples format.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        list: List of tuples (run_name, run_data) or empty list if file not found/error.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
        return []
    # print(df)
    # print(df.columns)
    # print(df['mpi_Allreduce'].describe())
    # print(df['mpi_Test'].describe())
    runs = []
    for index, row in df.iterrows():
        run_name = row['run_name']
        app_time = row['app_time']
        total_mpi_time = row['total_mpi_time']
        mpi_calls = {}
        for col_name, value in row.items():
            # Check if value is numeric and non-null before comparison
            if col_name.startswith('mpi_') and pd.notna(value) and isinstance(value, (int, float)) and value > 0:
                original_call_name = col_name[4:] # Remove 'mpi_' prefix
                mpi_calls[original_call_name] = value

        run_data = {
            'mpi_calls': mpi_calls,
            'app_time': app_time,
            'total_mpi_time': total_mpi_time
        }
        runs.append((run_name, run_data))

    print(f"Loaded {len(runs)} runs from {filepath}")
    return runs



##############################################################################
# 8) Main function: demonstration (renumbered from 7)
##############################################################################
def main():

    csv_path_amg = os.path.join(save_dir, system, "breakdown", f"AMG2023.csv")
    amg2023_runs = load_runs_from_csv(csv_path_amg)
    csv_path_milc = os.path.join(save_dir, system, "breakdown", f"MILC.csv")
    milc_runs = load_runs_from_csv(csv_path_milc)

    if not amg2023_runs and not milc_runs:
        print("Error: No data loaded for either application. Exiting.")
        return
    elif not amg2023_runs:
        print("Warning: No data loaded for AMG2023.")
        # Optionally create dummy data structure if needed downstream, or handle plots gracefully
    elif not milc_runs:
        print("Warning: No data loaded for MILC.")
        # Optionally create dummy data structure

    # Example 2: Same routine name ("Allreduce") for both apps
    plot_routine_violin(amg2023_runs, "Allreduce", "Waitall", milc_runs, "Allreduce", "Test")


if __name__ == "__main__":
    main()
