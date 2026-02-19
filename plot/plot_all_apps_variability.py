#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import dates as mdates
import matplotlib.pyplot as plt
import os
from brokenaxes import brokenaxes
import matplotlib.dates as mdates

# mechine = 'perlmutter'
# NAME = "Perlmutter"
mechine = 'frontier'
NAME = "Frontier"

colors = ['#D55E00', '#0072B2', '#009E73', '#000000', '#800080', '#CC79A7', '#E69F00', '#56B4E9']
markers = ['o', 's', 'd', '^']

df = pd.read_csv(os.path.join(os.getcwd(), mechine, "overall", 'all_' + mechine + '.csv'))
df = df.sort_values(by=['app_name', 'run_time'])

# Add date filtering for Frontier
if mechine == 'frontier':
    # Convert run_time to datetime for comparison
    df['run_time_dt'] = pd.to_datetime(df['run_time'])
    cutoff_date = pd.to_datetime('2025-01-12')
    
    # Filter data: keep all AMG2023 and MILC data, but only keep data after cutoff for nanoGPT and deepCAM
    df = pd.concat([
        df[df['app_name'].isin(['AMG2023', 'MILC', 'deepCAM'])],
        df[(df['app_name'].isin(['nanoGPT'])) & (df['run_time_dt'] >= cutoff_date)]
    ])
    
    # df = df[df['app_name'].isin(['AMG2023', 'MILC', 'nanoGPT', 'deepCAM'])]
    
    # Drop the temporary datetime column
    df = df.drop('run_time_dt', axis=1)

# print(df)
amg = df[df['app_name'] == 'AMG2023']
milc = df[df['app_name'] == 'MILC']
deepcam = df[df['app_name'] == 'deepCAM']
nanogpt = df[df['app_name'] == 'nanoGPT']
# print(deepcam)

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=20)
amg_x = np.arange(len(amg['runtime']))
amg.loc[:, 'relative'] = amg['runtime'] / amg['runtime'].min()
milc_x = np.arange(len(milc['runtime']))
milc['relative'] = milc['runtime'] / milc['runtime'].min()
deepcam_x = np.arange(len(deepcam['runtime']))
deepcam['relative'] = deepcam['runtime'] / deepcam['runtime'].min()
nanogpt_x = np.arange(len(nanogpt['runtime']))
nanogpt['relative'] = nanogpt['runtime'] / nanogpt['runtime'].min()

# plt.figure(figsize=(20, 4))
# # plt.title('Relative Performance', fontproperties=font_prop)

# plt.xlabel('Run', fontproperties=font_prop)
# plt.ylabel(NAME + ' Relative Runtime', fontproperties=font_prop)
# ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.set_ylim(0.9, 3.5)
# ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
# plt.grid(axis='y', linestyle=':', linewidth=0.7)
# plt.scatter(amg_x, amg['relative'], label='AMG2023', marker=markers[0], facecolors='none', edgecolors=colors[0])
# plt.scatter(milc_x, milc['relative'], label='MILC', marker=markers[1], facecolors='none', edgecolors=colors[1])
# plt.scatter(deepcam_x, deepcam['relative'], label='deepCAM', marker=markers[2], facecolors='none', edgecolors=colors[2])
# plt.scatter(nanogpt_x, nanogpt['relative'], label='nanoGPT', marker=markers[3], facecolors='none', edgecolors=colors[3])
# plt.legend(prop=font_prop)
# plt.savefig('relative_' + mechine + '.pdf')


amg_datetime = [date for date in amg['run_time']]
milc_datetime = [date for date in milc['run_time']]
deepcam_datetime = [date for date in deepcam['run_time']]
nanogpt_datetime = [date for date in nanogpt['run_time']]

# Convert string timestamps to datetime objects
if mechine == 'perlmutter':
    amg_datetime = [dt.datetime.strptime(date, '%Y-%m-%d_%H-%M-%S') for date in amg_datetime]
    milc_datetime = [dt.datetime.strptime(date, '%Y-%m-%d_%H-%M-%S') for date in milc_datetime]
    deepcam_datetime = [dt.datetime.strptime(date, '%Y-%m-%d_%H-%M-%S') for date in deepcam_datetime]
    nanogpt_datetime = [dt.datetime.strptime(date, '%Y-%m-%d_%H-%M-%S') for date in nanogpt_datetime]
else:
    amg_datetime = [dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in amg_datetime]
    milc_datetime = [dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in milc_datetime]
    deepcam_datetime = [dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in deepcam_datetime]
    nanogpt_datetime = [dt.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in nanogpt_datetime]

# Create a figure to see distribution by date/time

# if mechine == 'frontier':
#     plt.figure(figsize=(20, 5))
#     # plt.xlabel('Date/Time', fontproperties=font_prop)
#     plt.ylabel(NAME + ' Relative Runtime', fontproperties=font_prop)
#     ax = plt.gca()
#     ax.spines['top'].set_color('none')
#     ax.spines['right'].set_color('none')

#     # Format x-axis to show dates nicely
#     date_format = mdates.DateFormatter('%Y-%m-%d %H:%M')
#     ax.xaxis.set_major_formatter(date_format)
#     plt.xticks(rotation=45, fontproperties=font_prop)

#     # Plot the data with timestamps on x-axis
#     plt.scatter(amg_datetime, amg['relative'], label='AMG2023', marker='o', facecolors='none', edgecolors='steelblue')
#     plt.scatter(milc_datetime, milc['relative'], label='MILC', marker='s', facecolors='none', edgecolors='orange')
#     plt.scatter(deepcam_datetime, deepcam['relative'], label='deepCAM', marker='D', facecolors='none', edgecolors='black')
#     plt.scatter(nanogpt_datetime, nanogpt['relative'], label='nanoGPT', marker='^', facecolors='none', edgecolors='teal')

#     plt.legend(prop=font_prop)
#     plt.tight_layout()
#     plt.savefig('relative_by_datetime_' + mechine + '.pdf')


# Define the desired date format
date_fmt = mdates.DateFormatter('%b %d')

if mechine == 'perlmutter':
    pre_break = dt.datetime(2025, 1, 16)
    post_break = dt.datetime(2025, 2, 8)

    min_date = dt.datetime(2024, 12, 21)
    max_date = dt.datetime(2025, 4, 16)

    fig = plt.figure(figsize=(20, 3.3))
    bax = brokenaxes(xlims=((min_date, pre_break), (post_break, max_date)), d=0.005, wspace=0.03, fig=fig)
    # bax.set_xlabel('Date/Time', fontproperties=font_prop)
    bax.set_ylabel("Relative Performance", fontproperties=font_prop, labelpad=40)
    bax.scatter(amg_datetime, amg['relative'], label='AMG2023', marker=markers[0], facecolors='none', edgecolors=colors[0], alpha=1, s=80, linewidth=2)
    bax.scatter(milc_datetime, milc['relative'], label='MILC', marker=markers[1], facecolors='none', edgecolors=colors[1], alpha=1, s=80, linewidth=2)
    bax.scatter(deepcam_datetime, deepcam['relative'], label='deepCAM', marker=markers[2], facecolors='none', edgecolors=colors[2], alpha=1, s=80, linewidth=2)
    bax.scatter(nanogpt_datetime, nanogpt['relative'], label='nanoGPT', marker=markers[3], facecolors='none', edgecolors=colors[3], alpha=1, s=80, linewidth=2)
    # bax.legend(prop=font_prop, loc='upper left', ncol=4)
    bax.grid(axis='y', linestyle=':', linewidth=0.7)
    bax.set_title("Relative performance of different applications (" + NAME + ")", fontproperties=font_prop)
    for ax in bax.axs:
        ax.set_yticks([1, 1.1, 1.2, 1.3, 1.4])
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        # Apply the date formatter to the x-axis
        ax.xaxis.set_major_formatter(date_fmt)
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)
    # fig.autofmt_xdate() # Commented out as we handle formatting and rotation manually

    plt.savefig('relative_by_datetime_' + mechine + '.pdf')


if mechine == 'frontier':
    
    pre_break = dt.datetime(2025, 2, 11)
    post_break = dt.datetime(2025, 2, 16)

    min_date = dt.datetime(2024, 12, 21)
    max_date = dt.datetime(2025, 4, 16)

    fig = plt.figure(figsize=(20, 3.3))
    
    bax = brokenaxes(xlims=((min_date, pre_break), (post_break, max_date)), d=0.005, wspace=0.03, fig=fig)
    # bax.xticks(rotation=45, fontproperties=font_prop) # Commented out as we handle formatting manually
    # bax.set_xlabel('Date/Time', fontproperties=font_prop)
    bax.set_ylabel("Relative Performance", fontproperties=font_prop, labelpad=40)
    bax.scatter(amg_datetime, amg['relative'], label='AMG2023', marker=markers[0], facecolors='none', edgecolors=colors[0], alpha=1, s=80, linewidth=2)
    bax.scatter(milc_datetime, milc['relative'], label='MILC', marker=markers[1], facecolors='none', edgecolors=colors[1], alpha=1, s=80, linewidth=2)
    bax.scatter(deepcam_datetime, deepcam['relative'], label='DeepCAM', marker=markers[2], facecolors='none', edgecolors=colors[2], alpha=1, s=80, linewidth=2)
    bax.scatter(nanogpt_datetime, nanogpt['relative'], label='nanoGPT', marker=markers[3], facecolors='none', edgecolors=colors[3], alpha=1, s=80, linewidth=2)
    bax.legend(prop=font_prop, loc='upper left', ncol=2)
    bax.grid(axis='y', linestyle=':', linewidth=0.7)
    bax.set_title("Relative performance of different applications (" + NAME + ")", fontproperties=font_prop)
    for ax in bax.axs:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        # Set Y tick positions
        ax.set_yticks([1.0, 2.0, 3.0, 4.0])
        # Set Y tick labels to integers
        ax.set_yticklabels(['1.0', '2.0', '3.0', '4.0'])
        # Apply the date formatter to the x-axis
        ax.xaxis.set_major_formatter(date_fmt)
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)

    # fig.autofmt_xdate() # Commented out as we handle formatting and rotation manually
    # plt.tight_layout(rect=[0.05, 0.95, 1, 1])
    plt.savefig('relative_by_datetime_' + mechine + '.pdf')
