#!/usr/bin/env python3

import os
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import seaborn as sns # Import seaborn
from matplotlib.ticker import MaxNLocator

#######################
font_path = './gillsans.ttf'
font_dis_prop = font_manager.FontProperties(fname=font_path, size=22)
font_tick_prop = font_manager.FontProperties(fname=font_path, size=22) # Smaller font for ticks

system = 'perlmutter'
NAME = 'Perlmutter'
# system = 'frontier'
# NAME = 'Frontier'

save_dir = os.getcwd() # Define save_dir
colors = ['#D55E00', '#0072B2', '#009E73', '#800080', '#CC79A7', '#E69F00', '#56B4E9']
# --- Violin Plot Function (adapted for dual axis) ---
def plot_routine_violin(df1, routine1_name, routine2_name, df2, routine3_name, routine4_name, app1_name="nanoGPT", app2_name="deepCAM"):
    """
    Creates a dual-axis violin plot comparing NCCL/RCCL routine times (ms)
    for two applications. App1 uses left Y-axis, App2 uses right Y-axis.

    Args:
        df1 (pd.DataFrame): DataFrame for the first application.
        routine1_name (str): First routine for app1.
        routine2_name (str): Second routine for app1.
        df2 (pd.DataFrame): DataFrame for the second application.
        routine3_name (str): First routine for app2.
        routine4_name (str): Second routine for app2.
        app1_name (str): Name of the first application.
        app2_name (str): Name of the second application.
    """
    plot_data = []
    app1_labels = []
    app2_labels = []

    # --- Extract data for App 1 ---
    runtimes1 = df1[routine1_name].dropna().tolist() if routine1_name in df1.columns else []
    runtimes2 = df1[routine2_name].dropna().tolist() if routine2_name in df1.columns else []
    # Convert to seconds (divide by 1000) and filter
    valid_runtimes1 = [t for t in runtimes1 if t > 1e-6] # Convert ms to s
    valid_runtimes2 = [t for t in runtimes2 if t > 1e-6] # Convert ms to s
    label1 = f"{app1_name}_{routine1_name}"
    label2 = f"{app1_name}_{routine2_name}"
    if valid_runtimes1:
        # Add data already in seconds
        plot_data.extend([(t, routine1_name, app1_name, label1) for t in valid_runtimes1])
        app1_labels.append(label1)
    else: print(f"Warning: No valid data for {app1_name} - {routine1_name}")
    if valid_runtimes2:
        # Add data already in seconds
        plot_data.extend([(t, routine2_name, app1_name, label2) for t in valid_runtimes2])
        app1_labels.append(label2)
    else: print(f"Warning: No valid data for {app1_name} - {routine2_name}")

    # --- Extract data for App 2 ---
    runtimes3 = df2[routine3_name].dropna().tolist() if routine3_name in df2.columns else []
    routine4_actual_name = routine4_name
    if routine4_name not in df2.columns and routine4_name.capitalize() in df2.columns:
        routine4_actual_name = routine4_name.capitalize()
    elif routine4_name not in df2.columns:
         print(f"Warning: Column '{routine4_name}' not found in {app2_name} DataFrame.")
         routine4_actual_name = None
    runtimes4 = df2[routine4_actual_name].dropna().tolist() if routine4_actual_name and routine4_actual_name in df2.columns else []

    # Convert to seconds (divide by 1000) and filter
    valid_runtimes3 = [t for t in runtimes3 if t > 1e-6] # Convert ms to s
    valid_runtimes4 = [t for t in runtimes4 if t > 1e-6] # Convert ms to s
    label3 = f"{app2_name}_{routine3_name}"
    label4 = f"{app2_name}_{routine4_actual_name}" if routine4_actual_name else f"{app2_name}_{routine4_name}_NotFound"

    if valid_runtimes3:
        # Add data already in seconds
        plot_data.extend([(t, routine3_name, app2_name, label3) for t in valid_runtimes3])
        app2_labels.append(label3)
    else: print(f"Warning: No valid data for {app2_name} - {routine3_name}")
    if valid_runtimes4 and routine4_actual_name:
        # Add data already in seconds
        plot_data.extend([(t, routine4_actual_name, app2_name, label4) for t in valid_runtimes4])
        app2_labels.append(label4)
    elif routine4_actual_name: print(f"Warning: No valid data for {app2_name} - {routine4_actual_name}")

    # --- Prepare data for Seaborn ---
    if not plot_data:
        print("Error: No data available to plot.")
        return
    # Update DataFrame column name to 'Time (s)'
    df_plot = pd.DataFrame(plot_data, columns=['Time (s)', 'Routine', 'Application', 'PlotLabel'])

    print(df_plot)

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

    # Define a color palette map
    num_labels_to_plot = len(existing_labels_in_order)
    # palette_map = {label: colors[i % len(colors)] for i, label in enumerate(existing_labels_in_order)}
    palette_map = {}
    for i, label in enumerate(existing_labels_in_order):
        if label == "nanoGPT_AllReduce":
            palette_map[label] = colors[2]
        elif label == "nanoGPT_AllGather":
            palette_map[label] = colors[6]
        elif label == "deepCAM_AllReduce":
            palette_map[label] = colors[2]
        elif label == "deepCAM_mem":
            palette_map[label] = colors[4]
            
    violin_inner_style = "quartile"

    # --- Plot all violins on ax1 (Left Y-axis) ---
    # Use 'Time (s)' for y-axis
    sns.violinplot(x='PlotLabel', y='Time (s)', data=df_plot, order=existing_labels_in_order,
                   ax=ax1, palette=palette_map, inner=violin_inner_style, cut=0)

    # --- Plot all violins on ax2 (Right Y-axis) ---
    # Use 'Time (s)' for y-axis
    sns.violinplot(x='PlotLabel', y='Time (s)', data=df_plot, order=existing_labels_in_order,
                   ax=ax2, palette=palette_map, inner=violin_inner_style, cut=0)

    # --- Apply hatches and edge colors ---
    for i, label in enumerate(existing_labels_in_order):
        # color = palette_map[label]
        if label == "nanoGPT_AllReduce":
            color = colors[2]
        elif label == "nanoGPT_AllGather":
            color = colors[6]
        elif label == "deepCAM_AllReduce":
            color = colors[2]
        elif label == "deepCAM_mem":
            color = colors[4]
        if i < len(ax1.collections):
            ax1.collections[i].set_edgecolor(color) # Set edge color to match face color for hatch visibility
        if i < len(ax2.collections):
            ax2.collections[i].set_edgecolor(color) # Set edge color to match face color

    # --- Hide irrelevant parts 
    
    # --- Hide irrelevant parts ---
    num_violins = len(existing_labels_in_order)
    num_lines_per_violin = 3 if violin_inner_style == "quartile" else 0

    # Hide App2 violins/lines on ax1
    for i, label in enumerate(existing_labels_in_order):
        if label in app2_labels_existing:
            if i < len(ax1.collections): ax1.collections[i].set_visible(False)
            line_start_index = i * num_lines_per_violin
            for j in range(num_lines_per_violin):
                line_index = line_start_index + j
                if line_index < len(ax1.lines): ax1.lines[line_index].set_visible(False)

    # Hide App1 violins/lines on ax2
    for i, label in enumerate(existing_labels_in_order):
        if label in app1_labels_existing:
            if i < len(ax2.collections): ax2.collections[i].set_visible(False)
            line_start_index = i * num_lines_per_violin
            for j in range(num_lines_per_violin):
                line_index = line_start_index + j
                if line_index < len(ax2.lines): ax2.lines[line_index].set_visible(False)

    # --- Customization ---
    ax1.set_xlabel("")
    # Update Y-axis labels to 'Time (s)'
    ax1.set_ylabel("Time (s)", fontproperties=font_dis_prop)
    ax2.set_ylabel(None)
    # ax2.set_ylabel(f"{app2_name} Time (ms)", fontproperties=font_dis_prop)
    plot_title = f"Top 2 most variable routines ({NAME})"
    ax1.set_title(plot_title, fontproperties=font_dis_prop, y=1.03)

    # Remove labelcolor from tick_params
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(axis='x', length=0)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.tick_params(axis='x', length=0)
    # Create the *first row* of x-axis labels (Routine names) on ax1
    tick_labels_map_routine = {}
    if routine1_name in df1.columns: tick_labels_map_routine[label1] = routine1_name
    if routine2_name in df1.columns: tick_labels_map_routine[label2] = routine2_name
    if routine3_name in df2.columns: tick_labels_map_routine[label3] = routine3_name
    if routine4_actual_name and routine4_actual_name in df2.columns: tick_labels_map_routine[label4] = routine4_actual_name

    primary_tick_labels = [tick_labels_map_routine.get(label, "N/A") for label in existing_labels_in_order]
    ax1.set_xticks(range(len(existing_labels_in_order)))
    ax1.set_xticklabels(primary_tick_labels, fontproperties=font_tick_prop)

    # Add the *second row* of x-axis labels (App names) below the first row on ax1
    y_pos_app_label = -0.10
    if app1_labels_existing:
        indices1 = [existing_labels_in_order.index(label) for label in app1_labels_existing]
        x_pos_center1 = (min(indices1) + max(indices1)) / 2.0
        ax1.text(x_pos_center1, y_pos_app_label, "DeepCAM",
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.get_xaxis_transform(), fontproperties=font_tick_prop)
    if app2_labels_existing:
        indices2 = [existing_labels_in_order.index(label) for label in app2_labels_existing]
        x_pos_center2 = (min(indices2) + max(indices2)) / 2.0
        ax1.text(x_pos_center2, y_pos_app_label, "nanoGPT",
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
        except ValueError: pass
    if app_change_index != -1:
        ax1.axvline(app_change_index, color='grey', linestyle='--', linewidth=1, ymin=0.05, ymax=0.95)

    # Set y-axis limits independently
    # Use 'Time (s)' column for max calculation
    max_time_app1 = df_plot[df_plot['Application'] == app1_name]['Time (s)'].max() if app1_labels_existing else 0
    max_time_app2 = df_plot[df_plot['Application'] == app2_name]['Time (s)'].max() if app2_labels_existing else 0

    # Determine initial top limits with some padding
    top_lim_app1 = (max_time_app1 * 1.1) if max_time_app1 > 0 else 0.1
    top_lim_app2 = (max_time_app2 * 1.1) if max_time_app2 > 0 else 0.1

    # --- Calculate Ticks and Proportional Padding for ax1 ---
    locator1 = MaxNLocator(nbins=5, prune='lower') # Adjust nbins as needed
    ticks1 = locator1.tick_values(0, top_lim_app1) # Calculate ticks for range [0, top]
    # Ensure 0.0 is included if the range covers it and it wasn't added automatically
    if 0 not in ticks1 and top_lim_app1 > 0:
         if not ticks1.size or ticks1[0] > 1e-9: # Avoid adding 0 if ticks start very close to it
             ticks1 = np.insert(ticks1, 0, 0.0)
    ticks1 = ticks1[ticks1 >= 0] # Ensure no negative ticks remain if prune='lower' included one

    # Calculate interval for padding
    if len(ticks1) >= 2:
        interval1 = ticks1[1] - ticks1[0]
    elif len(ticks1) == 1 and ticks1[0] > 0:
        interval1 = ticks1[0] # Use the single tick value as interval guess
    elif top_lim_app1 > 0:
        interval1 = top_lim_app1 / 5 # Estimate interval based on range and nbins
    else:
        interval1 = 0.02 # Default interval if no data/ticks

    bottom_padding_factor = 0.2 # e.g., 20% of the tick interval
    bottom_pad1 = -bottom_padding_factor * interval1 if interval1 > 0 else -0.005 # Fallback fixed padding

    # --- Calculate Ticks and Proportional Padding for ax2 ---
    locator2 = MaxNLocator(nbins=5, prune='lower') # Adjust nbins as needed
    ticks2 = locator2.tick_values(0, top_lim_app2) # Calculate ticks for range [0, top]
    # Ensure 0.0 is included if the range covers it and it wasn't added automatically
    if 0 not in ticks2 and top_lim_app2 > 0:
        if not ticks2.size or ticks2[0] > 1e-9:
            ticks2 = np.insert(ticks2, 0, 0.0)
    ticks2 = ticks2[ticks2 >= 0] # Ensure no negative ticks

    # Calculate interval for padding
    if len(ticks2) >= 2:
        interval2 = ticks2[1] - ticks2[0]
    elif len(ticks2) == 1 and ticks2[0] > 0:
        interval2 = ticks2[0]
    elif top_lim_app2 > 0:
        interval2 = top_lim_app2 / 5
    else:
        interval2 = 0.02

    bottom_pad2 = -bottom_padding_factor * interval2 if interval2 > 0 else -0.005 # Fallback fixed padding

    if NAME == 'Perlmutter':
        ax2.set_ylim(bottom=-0.18, top=7.5)
        ax2.set_yticks([0, 1.5, 3, 4.5, 6, 7.5])
        ax2.set_yticklabels([0, 1.5, 3, 4.5, 6, 7.5])
        ax1.set_ylim(bottom=-0.01, top=0.4)
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax1.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])
    elif NAME == 'Frontier':
        ax2.set_ylim(bottom=-0.36, top=12)
        ax2.set_yticks([0, 2, 4, 6, 8, 10, 12])
        ax2.set_yticklabels([0, 2, 4, 6, 8, 10, 12])
        ax1.set_ylim(bottom=-0.72, top=25)
        ax1.set_yticks([0, 5, 10, 15, 20, 25])
        ax1.set_yticklabels([0, 5, 10, 15, 20, 25])
    else:
        ax1.set_ylim(bottom=0, top=max_time_app1 * 1.1) # Set top limit to 1.2x Worst time
        ax2.set_ylim(bottom=0, top=max_time_app2 * 1.1) # Set top limit to 1.2x Worst time

    # Ensure grid lines (if any) from ax2 don't obscure ax1 plot
    ax1.grid(False)
    ax2.grid(False)

    fig.tight_layout() # Use fig.tight_layout()

    # --- Save Plot ---
    routines_str = "_".join(re.sub(r'[^a-zA-Z0-9_]', '', r) for r in primary_tick_labels if r != "N/A")
    filename_base = "violin_nccl"
    save_path = os.path.join(save_dir, filename_base + f"_{system}.pdf")
    plt.savefig(save_path)
    print(f"Saved dual-axis violin plot to {save_path}")


def main():
    # Load data for both applications
    csv_path_nanogpt = os.path.join(save_dir, system, "breakdown", "nanoGPT.csv")

    csv_path_deepcam = os.path.join(save_dir, system, "breakdown", "deepCAM.csv") # Assuming filename deepCAM.csv

    df_nanogpt_pivot = pd.DataFrame() # Initialize as empty
    df_deepcam_pivot = pd.DataFrame() # Initialize as empty

    try:
        df_nanogpt_pivot = pd.read_csv(csv_path_nanogpt)
        print(f"Loaded nanoGPT data from {csv_path_nanogpt}")
    except FileNotFoundError:
        print(f"Error: nanoGPT CSV file not found at {csv_path_nanogpt}")
    except Exception as e:
        print(f"Error reading nanoGPT CSV file {csv_path_nanogpt}: {e}")

    try:
        df_deepcam_pivot = pd.read_csv(csv_path_deepcam)
        print(f"Loaded deepCAM data from {csv_path_deepcam}")
    except FileNotFoundError:
        print(f"Error: deepCAM CSV file not found at {csv_path_deepcam}")
    except Exception as e:
        print(f"Error reading deepCAM CSV file {csv_path_deepcam}: {e}")


    if df_nanogpt_pivot.empty and df_deepcam_pivot.empty:
        print("Error: No data loaded for either application. Exiting.")
        return
    # Warnings are printed inside the plot function if data is missing for one app
    # print(df_nanogpt_pivot)
    print(df_deepcam_pivot)
    # Call the dual-axis violin plot function
    plot_routine_violin(df_deepcam_pivot, "AllReduce", "mem", df_nanogpt_pivot, "AllReduce", "AllGather", app1_name="deepCAM", app2_name="nanoGPT")


if __name__ == "__main__":
    main()
