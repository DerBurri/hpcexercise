#!/usr/bin/python3

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Regex to extract relevant data
pattern = r"bandwidthTest-(H2D-Pinned|D2D|D2H-Pinned)-(cudaMemcpy|kernel1|kernel2|kernel3)-bytesperinst([\d]+), Bandwidth = ([\d.]+) GB/s, Time = ([\d.]+) s, Size = (\d+) bytes, NumDevsUsed = 1"

with open("ex3_results.txt", "r") as f:
    data = f.read()

# Extract data
matches = re.findall(pattern, data)
performance_data = [(transfer_type, operation, bytes_per_inst, float(bw), float(time), int(size)) for transfer_type, operation, bytes_per_inst, bw, time, size in matches]

# Create DataFrame
df = pd.DataFrame(performance_data, columns=['Transfer Type', 'Operation', 'Bytes per Inst', 'Bandwidth (GB/s)', 'Time (s)', 'Size (bytes)'])

# Split data into two categories
df_h2d_d2h = df[df['Transfer Type'].isin(['H2D-Pinned', 'D2H-Pinned'])]
df_d2d = df[df['Transfer Type'] == 'D2D']

# Create a dictionary to map 'bytes_per_inst' to colors
color_dict = {
    2: 'green',
    1: 'orange',
    4: 'blue',
    8: 'red',
    16: 'purple'
}
marker_dict = {
    'H2D-Pinned': 'o',  # Circle
    'D2D': 's',         # Square
    'D2H-Pinned': '^',  # Triangle
}

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, layout="tight")


# Plot H2D and D2H data in the first axis
for bytes_per_inst, group in df_h2d_d2h.groupby('Bytes per Inst'):
    color = color_dict[int(bytes_per_inst)]  # Ensure bytes_per_inst is treated as an integer
    for (transfer_type, operation), subgroup in group.groupby(['Transfer Type', 'Operation']):
        marker = marker_dict[transfer_type] 
        ax1.scatter(subgroup['Size (bytes)'], subgroup['Bandwidth (GB/s)'], alpha=0.5, color=color, marker=marker ,
                    label=f'{transfer_type}-{operation}-BytesPerInst={bytes_per_inst}')

# Plot H2D and D2H data in the first axis
# for bytes_per_inst, group in df_h2d_d2h.groupby('Bytes per Inst'):
#     color = color_dict[int(bytes_per_inst)]
#     for (transfer_type, operation), group in df_h2d_d2h.groupby(['Transfer Type', 'Operation']):
#         print(color)
#         ax1.scatter(group['Size (bytes)'], group['Bandwidth (GB/s)'], alpha=0.5, color=color,label=f'{transfer_type}-{operation}-BytesPerInst={bytes_per_inst}')

ax1.set_title('Memory Copy Throughput (H2D and D2H)', fontsize=18)
ax1.set_ylabel('Bandwidth [GBps]', fontsize=13)
ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelsize=13)
ax1.legend(title='Transfer-Operation', fontsize=14)
ax1.grid(True)

# Plot D2D data in the second axis
for bytes_per_inst, group in df_d2d.groupby('Bytes per Inst'):
    color = color_dict[int(bytes_per_inst)]  # Ensure bytes_per_inst is treated as an integer
    for (transfer_type, operation), subgroup in group.groupby(['Transfer Type', 'Operation']):
        ax2.scatter(subgroup['Size (bytes)'], subgroup['Bandwidth (GB/s)'], alpha=0.5, color=color,
                    label=f'{transfer_type}-{operation}-BytesPerInst={bytes_per_inst}')

ax2.set_title('Memory Copy Throughput (D2D)', fontsize=18)
ax2.set_xlabel('Size [B]', fontsize=13)
ax2.set_ylabel('Bandwidth [GBps]', fontsize=13)
ax2.set_xscale('log')
ax2.tick_params(axis='x', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
ax2.legend(title='Transfer-Operation', fontsize=14)
ax2.grid(True)

# Adjust spacing between subplots
plt.tight_layout(h_pad=3.0)

# Save the figure
plt.savefig('thrp_ex3.pdf', format='pdf')
plt.show()