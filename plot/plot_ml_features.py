import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

font_path = './gillsans.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=9)

colors = ['#D55E00', '#0072B2', '#009E73', '#800080', '#2A3240', '#CC79A7', '#E69F00', '#56B4E9', '#E03A3D', '#E4F4F7']
hatches = ['x', 'xxx', '\\\\', '||','///', '+', 'o', '.', '*', '-', 'ooo', '+++', 'xx', '/', '\\', '***']

p_features = [
    "hni_rx_paused_0_mean",
    "allreduce_2G",
    "allreduce_256M",
    "hni_tx_paused_0_min",
    "lpe_net_match_overflow_0_mean",
    # "hni_rx_paused_0_min",
    # "lpe_net_match_overflow_0_mean",
    # "hni_rx_paused_0_max",
    # "parbs_tarb_pi_non_posted_pkts_mean",
    # "atu_cache_evictions_mean"
]
p_importances = [
    0.73116994,
    0.121059924,
    0.041390736,
    0.025131278,
    0.020690637,
    # 0.011193991,
    # 0.006799197,
    # 0.0051169298,
    # 0.0038949489,
    # 0.002756486
]

# Frontier 
f_features = [
    "lpe_net_match_request_0_mean",
    "atu_cache_hit_derivative1_page_size_0_mean",
    "parbs_tarb_pi_non_posted_blocked_cnt_mean",
    "parbs_tarb_pi_posted_pkts_mean",
    "lpe_net_match_request_0_min",
    # "hni_rx_paused_1_mean",
    # "parbs_tarb_pi_non_posted_pkts_mean",
    # "atu_cache_hit_base_page_size_0_min",
    # "parbs_tarb_pi_posted_blocked_cnt_mean",
    # "parbs_tarb_pi_posted_pkts_max"
]
f_importances = [
    0.6819358,
    0.19697559,
    0.06646452,
    0.017458675,
    0.0049250787,
    # 0.0038739939,
    # 0.0031638565,
    # 0.0027003405,
    # 0.0026707216,
    # 0.0020770787
]

# Create a figure with two subplots sharing the y-axis
fig, ax = plt.subplots(1, 2, figsize=(5, 3), sharey=True)

# Plot Perlmutter data in left subplot
x_p = np.arange(len(p_features))
ax[0].bar(x_p, p_importances, 
        color=colors[:5], edgecolor='black', alpha=0.9, hatch=hatches[:5])
# ax[0].set_title("Perlmutter", fontproperties=font_prop, fontsize=10)
ax[0].set_xticks(x_p)
ax[0].set_xticklabels(p_features, rotation=25, ha='right', fontproperties=font_prop)
ax[0].grid(axis='y', linestyle='--', alpha=0.5)
ax[0].set_ylabel("Importance", fontproperties=font_prop, fontsize=10)

# Plot Frontier data in right subplot
x_f = np.arange(len(f_features))
ax[1].bar(x_f, f_importances, 
        color=colors[:5], edgecolor='black', alpha=0.9, hatch=hatches[:5])
# ax[1].set_title("Frontier", fontproperties=font_prop, fontsize=10)
ax[1].set_xticks(x_f)
ax[1].set_xticklabels(f_features, rotation=25, ha='right', fontproperties=font_prop)
ax[1].grid(axis='y', linestyle='--', alpha=0.5)

# Set common title for both subplots
fig.suptitle("Top 5 feature importances: Perlmutter (left) vs. Frontier (right)", fontproperties=font_prop, fontsize=12)

# Adjust layout
fig.tight_layout()
plt.subplots_adjust(top=0.9, wspace=0.2)  # Make room for the suptitle
plt.savefig("machine_pm_frontier_feature_importance.pdf", bbox_inches='tight', dpi=300)
