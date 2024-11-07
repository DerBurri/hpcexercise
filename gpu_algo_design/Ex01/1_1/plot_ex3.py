#!/usr/bin/env python3

import re
import pandas as pd
import matplotlib.pyplot as plt

# Regex to extract relevant data
pattern = r"bandwidthTest-(H2D-Pinned|D2D|D2H-Pinned)-(cudaMemcpy|kernel3-[\w-]+), Bandwidth = ([\d.]+) GB/s, Time = ([\d.]+) s, Size = (\d+) bytes"

with open("ex3_results.txt", "r") as f:
    data = f.read()

# Extract data
matches = re.findall(pattern, data)
performance_data = [(transfer_type, operation, float(bw), float(time), int(size)) for transfer_type, operation, bw, time, size in matches]

# Create DataFrame
df = pd.DataFrame(performance_data, columns=['Transfer Type', 'Operation', 'Bandwidth (GB/s)', 'Time (s)', 'Size (bytes)'])

df_host_trans = df[df['Transfer Type'].isin(['H2D-Pinned', 'D2H-Pinned'])]
df_dev_trans = df[df['Transfer Type'] == 'D2D']

# Display the processed dataframe (shown here for verification)
fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True, layout="tight")

marker_dict = {
    'H2D-Pinned': 'o',  # Circle
    'D2D'       : 'x',  # Cross
    'D2H-Pinned': '^',  # Triangle
}

# Group by Transfer Type and Operation, and plot each group
for (transfer_type, operation), group in df_host_trans.groupby(['Transfer Type', 'Operation']):
    marker = marker_dict[transfer_type]
    axes[0].scatter(group['Size (bytes)'], group['Bandwidth (GB/s)'], alpha=0.7, label=f'{transfer_type}-{operation}', marker=marker)
for (transfer_type, operation), group in df_dev_trans.groupby(['Transfer Type', 'Operation']):
    marker = marker_dict[transfer_type]
    axes[1].scatter(group['Size (bytes)'], group['Bandwidth (GB/s)'], alpha=0.7, label=f'{transfer_type}-{operation}', marker=marker)

axes[0].set_title('Throughput of host transfers', fontsize=18)
axes[0].set_ylabel('Bandwidth [GBps]', fontsize=13)
axes[0].tick_params(axis='x', labelsize=13)
axes[0].tick_params(axis='y', labelsize=13)
axes[0].legend(fontsize=14)
axes[0].grid(True)

axes[1].set_title('Throughput of device-only transfers', fontsize=18)
axes[1].set_xlabel('Size [B]', fontsize=13)
axes[1].set_ylabel('Bandwidth [GBps]', fontsize=13)
axes[1].set_xscale('log')
axes[1].tick_params(axis='x', labelsize=13)
axes[1].tick_params(axis='y', labelsize=13)
axes[1].legend(fontsize=14)
axes[1].grid(True)

plt.tight_layout(h_pad=3.0)

plt.savefig('thrp_ex3.pdf', format='pdf')
