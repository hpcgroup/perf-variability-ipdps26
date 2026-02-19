import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.font_manager as font_manager

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=18)
# AMG, MILC, deepCAM, nanoGPT
colors = ['#D55E00', '#0072B2', '#009E73', '#000000', '#800080', '#CC79A7', '#E69F00', '#56B4E9']

markers = ['o', 's', 'd', '^']
apps_dict = {
    'AMG2023': 0,
    'MILC': 1,
    'deepCAM': 2,
    'nanoGPT': 3
}

system = 'frontier'
# system = 'perlmutter'
# APP = 'AMG2023'
# APP = 'nanoGPT'
# APP = 'MILC'
APP = 'deepCAM'

APP_DISPLAY_NAME = APP if APP != 'deepCAM' else 'DeepCAM'
# Importing the dataset
performance_analysis_dir = os.path.join(os.getcwd(), system, 'neighborhood')

dataset = pd.read_csv(os.path.join(performance_analysis_dir, APP + '_performance_analysis.csv'))
# print(dataset['app_start'])
dataset = dataset[dataset['app_start'] > '2024-02-08']

user_dict = {}
# Extract user allocation dictionaries from dataset
user_alloc_dicts = dataset['user_alloc_dict_trimmed']

if isinstance(user_alloc_dicts.iloc[0], str):
    user_alloc_dicts = user_alloc_dicts.apply(eval)

apptime = dataset["app_total_time_ms"]
relative_apptime = apptime / apptime.min()
total_concurrent_nodes = dataset["total_concurrent_nodes"]
concurrent_nodes_in_same_group = dataset["concurrent_nodes_in_same_group"]

plt.figure(figsize=(6,4))
if system == 'frontier' and (APP == 'nanoGPT' or APP == 'deepCAM'):
    plt.scatter(total_concurrent_nodes, relative_apptime, facecolors='none', edgecolors=colors[apps_dict[APP]], marker=markers[apps_dict[APP]], s=80, linewidth=2)
elif system == 'perlmutter' and APP == 'nanoGPT':
    plt.scatter(total_concurrent_nodes, relative_apptime, facecolors='none', edgecolors=colors[apps_dict[APP]], marker=markers[apps_dict[APP]], s=80, linewidth=2)
else:
    plt.scatter(total_concurrent_nodes, relative_apptime, facecolors='none', edgecolors=colors[apps_dict[APP]], marker=markers[apps_dict[APP]], s=80, linewidth=2)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.title(f'Total concurrent nodes vs. {APP_DISPLAY_NAME} perf. \n({system.capitalize()})', fontproperties=font_prop, fontsize=19.5)
if system == "perlmutter":
    if APP == 'AMG2023':
        plt.ylim(0.98, 1.35)
if system == 'frontier':
    if APP == 'deepCAM':
        plt.ylim(0.8, 4)
        tick_values = [1.0, 2.0, 3.0, 4.0]
        tick_labels = [f"{val:.1f}" for val in tick_values]
        plt.yticks(ticks=tick_values, labels=tick_labels, fontproperties=font_prop, fontsize=18)
plt.xlabel('Number of total allocated concurrent nodes', fontproperties=font_prop, fontsize=20)
plt.ylabel('Relative Performance', fontproperties=font_prop, fontsize=20)
plt.xticks(fontproperties=font_prop, fontsize=18)
plt.yticks(fontproperties=font_prop, fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), APP + '_total_concurrent_nodes_vs_apptime_' + system + '.pdf'))
plt.close()


corr = pd.read_csv(os.path.join(performance_analysis_dir, APP + "user_time_correlation.csv"))

if system == 'perlmutter':
    if APP == 'AMG2023':
        corr = corr[corr["correlation"] > 0.30]
        filtered = corr[corr["max"] > 8]
    elif APP == 'nanoGPT':
        corr = corr[corr["correlation"] > 0.30]
        filtered = corr[corr["max"] > 16]
    elif APP == 'deepCAM':
        corr = corr[corr["correlation"] > 0.4]
        filtered = corr[corr["max"] > 32]
    elif APP == 'MILC':
        corr = corr[corr["correlation"] > 0.3]
        filtered = corr[corr["max"] > 8]
elif system == 'frontier':
    if APP == 'AMG2023':
        corr = corr[corr["correlation"] > 0.3]
        filtered = corr[corr["max"] > 8]
    elif APP == 'nanoGPT':
        corr = corr[corr["correlation"] > 0.35]
        filtered = corr[corr["max"] > 8]
    elif APP == 'deepCAM':
        corr = corr[corr["correlation"] > 0.35]
        filtered = corr[corr["max"] > 16]

top_user = []
for index, row in filtered.iterrows():
    top_user.append(row["username"][5:])

i = 0
concurrent = [0] * len(user_alloc_dicts)

for job_dict in user_alloc_dicts:
    for user, value in job_dict.items():
        if user in top_user:
            concurrent[i] += value
    i += 1
print(apptime)
apprelativetime = apptime / apptime.min()
# print(apprelativetime)
plt.figure(figsize=(6,4))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.scatter(concurrent, apprelativetime, facecolors='none', edgecolors=colors[apps_dict[APP]], marker=markers[apps_dict[APP]], s=80, linewidth=2)
if system == 'perlmutter':
    if APP == 'AMG2023':
        plt.ylim(0.98, 1.35)
        plt.yticks([1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35], fontproperties=font_prop, fontsize=18)
        plt.ylabel('Relative Performance', fontproperties=font_prop, fontsize=20)
    if APP == 'nanoGPT':
        plt.ylim(0.98, 1.40)
        plt.yticks([1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4], fontproperties=font_prop, fontsize=18)
if system == 'frontier':
    if APP == 'AMG2023':
        plt.ylim(0.995, 1.10)
        plt.yticks([1, 1.02, 1.04, 1.06, 1.08, 1.1], fontproperties=font_prop, fontsize=18)
    if APP == 'deepCAM':
        plt.ylim(0.8, 4)
        # Explicitly set labels as strings with decimal points
        tick_values = [1.0, 2.0, 3.0, 4.0]
        tick_labels = [f"{val:.1f}" for val in tick_values]
        plt.yticks(ticks=tick_values, labels=tick_labels, fontproperties=font_prop, fontsize=18)
plt.title(f"Num of concurrent nodes for top users vs. \n{APP_DISPLAY_NAME} perf. ({system.capitalize()})",
          fontproperties=font_prop, fontsize=21)
plt.xlabel('Number of concurrent nodes of top users', fontproperties=font_prop, fontsize=20)
plt.ylabel('Relative Performance', fontproperties=font_prop, fontsize=20)
plt.xticks(fontproperties=font_prop, fontsize=18)
# plt.yticks(fontproperties=font_prop, fontsize=18)
plt.tight_layout()
plt.savefig(APP + '_concurrent_allocated_nodes_vs_app_time_' + system + '.pdf')
plt.close()