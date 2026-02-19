import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=20)
# Define one color per application
app_colors = {
    'AMG': '#D55E00',      # Orange
    'nanoGPT': '#000000',  # Black (swapped with MILC)
    'DeepCAM': '#009E73',  # Green
    'MILC': '#0072B2'      # Blue (swapped with nanoGPT)
}

# Define one marker per application (with 'x' reserved for predictions)
app_markers = {
    'AMG': 'o',
    'nanoGPT': '^',        # Triangle (swapped with MILC)
    'DeepCAM': 'd', 
    'MILC': 's'            # Square (swapped with nanoGPT)
}

linestyle_tuple = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),
     
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
     
     ]

linestyle_dict = {k : v for k,v in linestyle_tuple}

def get_linestyles():
    return [
            linestyle_dict['solid'], 
            linestyle_dict['dotted'], 
            linestyle_dict['dashed'],
            linestyle_dict['dashdotted'],
            linestyle_dict['dashdotdotted'],
            linestyle_dict['densely dashdotted'],
            linestyle_dict['loosely dashdotted'],
            linestyle_dict['densely dashdotdotted'],
            
    ]

def get_linestyles_dict():
    return linestyle_dict

MECHINE_NAME = "Perlmutter"
model_name = "full"
test_size = 10  
test_size_milc = 7

y_amg_test2 = [331.958984, 371.952952, 332.429003, 331.997612, 330.681412, 335.518995, 330.327343, 330.744501, 331.702567, 331.550358]
y_pred2 = [331.958984, 371.952952, 332.429003, 331.997612, 330.681412, 335.518995, 330.327343, 330.744501, 331.702567, 331.550358]
y_nanogpt_test2 = [275.94912, 274.79561, 316.62572, 277.9513, 278.08595, 278.99186, 277.50953, 343.51278, 279.023, 276.41794]
y_pred3 = [276.1189, 275.6523, 305.5281, 279.22684, 278.5395, 280.52023, 275.90427, 346.11136, 276.94733, 279.35324]
y_deepcam_test2 = [150.23162, 150.35823, 149.42864, 150.22772, 165.39279, 150.81214, 148.52473, 149.18053, 149.69671, 150.4368]
y_pred4 = [150.07307, 154.99947, 148.96419, 151.1872, 164.47012, 149.64818, 148.80872, 150.68932, 149.50015, 149.67888]
y_milc_test2 = [216.171, 219.3841, 255.5231, 237.6713, 219.8716, 217.4444, 214.9355]
y_pred5 = [220.83304, 223.49634, 232.62338, 223.85934, 223.63135, 218.94667, 212.8922]

plt.figure(figsize=(8, 7))

# Get min and max for axis limits
all_actual = y_amg_test2 + y_nanogpt_test2 + y_deepcam_test2 + y_milc_test2
all_predicted = y_pred2 + y_pred3 + y_pred4 + y_pred5
min_val = min(min(all_actual), min(all_predicted))
max_val = max(max(all_actual), max(all_predicted))

# Add diagonal line for perfect predictions
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)

# Plot scatter points for each application with hollow markers
plt.scatter(y_amg_test2, y_pred2, marker=app_markers['AMG'], edgecolor=app_colors['AMG'], facecolor='none', label="AMG", s=80, linewidth=2)
plt.scatter(y_nanogpt_test2, y_pred3, marker=app_markers['nanoGPT'], edgecolor=app_colors['nanoGPT'], facecolor='none', label="nanoGPT", s=80, linewidth=2)
plt.scatter(y_deepcam_test2, y_pred4, marker=app_markers['DeepCAM'], edgecolor=app_colors['DeepCAM'], facecolor='none', label="DeepCAM", s=80, linewidth=2)
plt.scatter(y_milc_test2, y_pred5, marker=app_markers['MILC'], edgecolor=app_colors['MILC'], facecolor='none', label="MILC", s=80, linewidth=2)

# Set equal axis limits with a small buffer
plt.xlim(100, 400)
plt.ylim(100, 400)

plt.xlabel("Actual Runtime (s)", fontproperties=font_prop, fontsize=30)
plt.ylabel("Predicted Runtime (s)", fontproperties=font_prop, fontsize=30)
plt.title("Actual vs Predicted Runtime (Perlmutter)", fontproperties=font_prop, fontsize=30)
plt.grid(True, axis='y', linestyle='--')  # Changed to horizontal dashed grid lines only
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.tight_layout()
plt.savefig("%s_%s_prediction_scatter.pdf" % (MECHINE_NAME, model_name), bbox_inches='tight', dpi=300)

MECHINE_NAME = "frontier"

y_amg_test2 = [378.293064, 372.893222, 385.868694, 382.152044, 375.523224, 388.394872,
 385.436363, 382.367393, 387.740909, 383.284512]
y_pred2 = [381.7034, 378.5662, 384.02243, 384.6357, 379.91208, 386.31302, 383.68204,
 381.83597, 387.99545, 382.27777]
y_nanogpt_test2 = [775.49447, 773.89862, 767.4066, 781.55213, 771.26949, 775.29754, 771.63054,
 784.68297, 764.06958, 786.59664]
y_pred3 = [773.15814, 774.13556, 775.41583, 772.558, 767.7256, 774.48975, 774.96655,
 774.60455, 769.28613, 776.2653]
y_deepcam_test2 = [686.3399, 1443.17087, 1541.04328, 608.14027, 647.85368, 581.2617,
  627.10096, 586.08948, 1182.29571, 705.57754]
y_pred4 = [693.7587, 1608.1637, 1495.5405, 595.0299, 645.315, 586.002,
  616.278, 584.6304, 706.8277, 680.72174]
y_milc_test2 = [380.3204, 374.8269, 313.1841, 450.7646, 316.3376, 293.4118, 378.5142]
y_pred5 = [384.9352, 347.16928, 315.6485, 348.50052, 317.61703, 306.17862, 377.8151]

plt.figure(figsize=(8, 7))

# Get min and max for axis limits
all_actual = y_amg_test2 + y_nanogpt_test2 + y_deepcam_test2 + y_milc_test2
all_predicted = y_pred2 + y_pred3 + y_pred4 + y_pred5
min_val = min(min(all_actual), min(all_predicted))
max_val = max(max(all_actual), max(all_predicted))

# Add diagonal line for perfect predictions
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2)

# Plot scatter points for each application with hollow markers
plt.scatter(y_amg_test2, y_pred2, marker=app_markers['AMG'], edgecolor=app_colors['AMG'], facecolor='none', label="AMG", s=80, linewidth=2)
plt.scatter(y_nanogpt_test2, y_pred3, marker=app_markers['nanoGPT'], edgecolor=app_colors['nanoGPT'], facecolor='none', label="nanoGPT", s=80, linewidth=2)
plt.scatter(y_deepcam_test2, y_pred4, marker=app_markers['DeepCAM'], edgecolor=app_colors['DeepCAM'], facecolor='none', label="DeepCAM", s=80, linewidth=2)
plt.scatter(y_milc_test2, y_pred5, marker=app_markers['MILC'], edgecolor=app_colors['MILC'], facecolor='none', label="MILC", s=80, linewidth=2)

# Set equal axis limits with a small buffer
plt.xlim(200, 1800)
plt.ylim(200, 1800)

plt.xlabel("Actual Runtime (s)", fontproperties=font_prop, fontsize=30)
plt.ylabel("Predicted Runtime (s)", fontproperties=font_prop, fontsize=30)
plt.title("Actual vs Predicted Runtime (Frontier)", fontproperties=font_prop, fontsize=30)
plt.grid(True, axis='y', linestyle='--')  # Changed to horizontal dashed grid lines only
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
# plt.legend(prop=font_prop)
plt.tight_layout()
plt.savefig("%s_%s_prediction_scatter.pdf" % (MECHINE_NAME, model_name), bbox_inches='tight', dpi=300)