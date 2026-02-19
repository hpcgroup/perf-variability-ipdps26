import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=20)

# AMG, MILC, deepCAM, nanoGPT
colors = ['#D55E00', '#0072B2', '#009E73', '#000000', '#800080', '#CC79A7', '#E69F00', '#56B4E9']
markers = ['o', 's', 'd', '^']

# Perlmutter
pm_amg_mape = [1.97,  2.35,  1.26,  0.13]
pm_amg_da   = [1.00,  0.50,  1.00,  1.00]

pm_nano_mape = [4.07,  4.32,  3.63,  0.82]
pm_nano_da   = [0.50,  1.00,  0.50,  1.00]

pm_dc_mape   = [2.96,  1.11,  1.24,  0.73]
pm_dc_da     = [1.00,  1.00,  1.00,  1.00]

pm_milc_mape = [7.72,  5.76,  5.18,  3.17]
pm_milc_da   = [0.00,  0.67,  0.33,  1.00]

# Frontier
ft_amg_mape = [1.25,  3.29,  1.46,  0.62]
ft_amg_da   = [0.00,  0.00,  1.00,  1.00]

ft_nano_mape = [1.21,  9.63,  1.85,  0.68]
ft_nano_da   = [1.00,  1.00,  1.00,  1.00]

ft_dc_mape   = [23.21, 20.49, 24.67,  6.45]
ft_dc_da     = [0.22,  0.44,  0.65,   0.89]

ft_milc_mape = [12.66,  14.07,  11.26,  5.29]
ft_milc_da   = [0.40,  0.60,  1.00,  1.00]

# 0:AMG, 1:nanoGPT, 2:deepCAM
pm_mape_data = [pm_amg_mape, pm_nano_mape, pm_dc_mape, pm_milc_mape]
pm_da_data   = [pm_amg_da,   pm_nano_da,   pm_dc_da,   pm_milc_da]

ft_mape_data = [ft_amg_mape, ft_nano_mape, ft_dc_mape, ft_milc_mape]
ft_da_data   = [ft_amg_da,   ft_nano_da,   ft_dc_da,   ft_milc_da]

method_labels = [
    "placement",
    "placement+gemm",
    "placement+gemm+allreduce",
    "placement+gemm+allreduce+NIC"
]

model_names = ["AMG", "nanoGPT", "DeepCAM", "MILC"]

# hatch
config_colors = ['#D55E00', '#0072B2', '#009E73', '#800080']
hatches = ['x', '\\\\', 'xxx', '||']

perlmutter_positions = np.array([0, 1, 2, 3])
frontier_positions   = np.array([4, 5, 6, 7])

width = 0.15
offsets = np.linspace(-1.5*width, 1.5*width, 4)

fig, (ax_mape, ax_da) = plt.subplots(1, 2, figsize=(16, 7), sharex=True)

# Perlmutter MAPE
for i in range(4):
    base = perlmutter_positions[i]
    for j in range(4):
        ax_mape.bar(base + offsets[j],
                    pm_mape_data[i][j],
                    width=width,
                    color=config_colors[j],
                    edgecolor='black',
                    hatch=hatches[j],
                    alpha=0.9,
                    label=method_labels[j] if (i==0) else None)
# Frontier MAPE
for i in range(4):
    base = frontier_positions[i]
    for j in range(4):
        ax_mape.bar(base + offsets[j],
                    ft_mape_data[i][j],
                    width=width,
                    color=config_colors[j],
                    edgecolor='black',
                    hatch=hatches[j],
                    alpha=0.9)

# Perlmutter DA
for i in range(4):
    base = perlmutter_positions[i]
    for j in range(4):
        ax_da.bar(base + offsets[j],
                  pm_da_data[i][j],
                  width=width,
                  color=config_colors[j],
                  edgecolor='black',
                  hatch=hatches[j],
                  alpha=0.9,
                  label=method_labels[j] if (i==0) else None)
# Frontier DA
for i in range(4):
    base = frontier_positions[i]
    for j in range(4):
        ax_da.bar(base + offsets[j],
                  ft_da_data[i][j],
                  width=width,
                  color=config_colors[j],
                  edgecolor='black',
                  hatch=hatches[j],
                  alpha=0.9)


xticks_positions = np.concatenate([perlmutter_positions, frontier_positions])
# Change x-tick labels to only show app names
xtick_labels = model_names + model_names
ax_mape.set_xticks(xticks_positions)
ax_mape.set_xticklabels(xtick_labels, rotation=25, fontproperties=font_prop)
ax_da.set_xticks(xticks_positions)
ax_da.set_xticklabels(xtick_labels, rotation=25, fontproperties=font_prop)

# Use secondary_xaxis for system grouping labels and vertical separators
def add_system_labels_with_secondary_xaxis(ax):
    # First secondary x-axis for system labels
    sec = ax.secondary_xaxis(location=0)
    # Adjust y position slightly for system labels to make space for the line
    sec.set_xticks([1.5, 5.5], labels=['\n\n\nPerlmutter', '\n\n\nFrontier'], fontproperties=font_prop, fontsize=25)
    sec.tick_params('x', length=0)

    # Remove the second secondary x-axis for vertical separator lines
    # secax_lines = ax.secondary_xaxis(location=0)
    # secax_lines.set_xticks([3.5], labels=[])  # Middle separator only
    # secax_lines.tick_params(axis='x', length=85, width=1.5)  # Vertical lines

    # Add horizontal lines below system labels, similar to plot_mpi_violin.py
    y_pos_line = -0.19 # Adjust this value to position the line correctly below system labels
    line_width = 1.5
    line_padding = 0.4 # How much the line extends beyond the center of the outer bars in each group

    # Line for Perlmutter (indices 0 to 3)
    x_start1 = -line_padding
    x_end1 = 3 + line_padding
    ax.plot([x_start1, x_end1], [y_pos_line, y_pos_line], linewidth=line_width, solid_capstyle='butt', color='black',
             transform=ax.get_xaxis_transform(), clip_on=False)

    # Line for Frontier (indices 4 to 7)
    x_start2 = 4 - line_padding
    x_end2 = 7 + line_padding
    ax.plot([x_start2, x_end2], [y_pos_line, y_pos_line], linewidth=line_width, solid_capstyle='butt', color='black',
             transform=ax.get_xaxis_transform(), clip_on=False)


    # Set x-axis limits to ensure separators are visible
    ax.set_xlim(-0.6, 7.6)

# Add system labels to both subplots
add_system_labels_with_secondary_xaxis(ax_mape)
add_system_labels_with_secondary_xaxis(ax_da)

# Add padding to titles to increase space
title_pad = 20 # Adjust this value as needed
ax_mape.set_title("MAPE (%) (lower is better)", fontproperties=font_prop, fontsize=30, pad=title_pad)
ax_da.set_title("Direction Accuracy (higher is better)", fontproperties=font_prop, fontsize=30, pad=title_pad)
ax_mape.set_ylabel("MAPE (%)", fontproperties=font_prop, fontsize=30)
ax_da.set_ylabel("Direction Accuracy", fontproperties=font_prop, fontsize=30)

# Adjust bottom margin to accommodate braces and labels
# Increase bottom margin slightly more to accommodate the new horizontal lines and adjusted labels
plt.subplots_adjust(bottom=0.3)

ax_mape.set_ylim(0, 30)
ax_da.set_ylim(0, 1.2)
ax_mape.grid(axis='y', alpha=0.3)
ax_da.grid(axis='y', alpha=0.3)

# Set font for y-axis tick labels
for label in ax_mape.get_yticklabels():
    label.set_fontproperties(font_prop)
for label in ax_da.get_yticklabels():
    label.set_fontproperties(font_prop)

handles, labels = ax_mape.get_legend_handles_labels()
ax_mape.legend(handles, labels, loc='upper left', prop=font_prop, fontsize=17)

# Call tight_layout before savefig to ensure adjustments are considered
plt.tight_layout()
plt.savefig("machine_pm_frontier_mape_da.pdf")
# plt.show()
