import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the training results
result_mac = pd.read_csv('model_info.csv')

# Convert the number_parameters column to numeric (in case they are read as strings)
result_mac['number_parameters'] = pd.to_numeric(result_mac['number_parameters'])

# Plotting the results
plt.figure(figsize=(12, 8))

# Set up bar positions
bar_width = 0.35
index = np.arange(len(result_mac['model']))

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 8))

# Bars for MACs
bar1 = ax1.bar(index, result_mac['mac'], bar_width, color='b', alpha=0.6, label='MACs (G)')
ax1.set_xlabel('Model')
ax1.set_ylabel('MACs (G)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create another y-axis to show the number of parameters
ax2 = ax1.twinx()
bar2 = ax2.bar(index + bar_width, result_mac['number_parameters'] / 1e6, bar_width, color='r', alpha=0.6, label='Parameters (M)')
ax2.set_ylabel('Parameters (M)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Set the x-ticks to the middle of the groups
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels(result_mac['model'])

# Title and legend
plt.title('Model Comparison: MACs and Number of Parameters')
fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Save the plot
plt.savefig("exercise5_2_mac.pdf")
plt.show()
