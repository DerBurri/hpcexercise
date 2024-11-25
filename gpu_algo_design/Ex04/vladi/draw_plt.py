#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import re

# File path for the text file containing the data
file_path = 'histogram-65542.out'

# Read data from file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Modify the regex pattern based on the data format in the file
pattern = re.compile(r'binNum = (\d+), Throughput = ([\d.]+) MB/s.*Workgroup = (\d+)')

# Create a list to hold the parsed data
parsed_data = []

# Process each line to extract relevant details
for line in lines:
    match = pattern.search(line)
    if match:
        bin_num = int(match.group(1))
        throughput = float(match.group(2)) / 1000
        workgroup = int(match.group(3))
        parsed_data.append({"binNum": bin_num, "Throughput": throughput, "Workgroup": workgroup})

# Check if any data was parsed
if not parsed_data:
    print("No data matched. Please check the regular expression and the input file.")
else:
    # Create a DataFrame from the parsed data
    df = pd.DataFrame(parsed_data)

    # Begin plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data for each Workgroup
    for workgroup, group_data in df.groupby('Workgroup'):
        ax.plot(group_data['binNum'], group_data['Throughput'], label=f'Workgroup {workgroup}', marker='o')

    # Set plot labels and title
    ax.set_xlabel('Bin Number')
    ax.set_ylabel('Throughput [GBps]')
    ax.set_title('Throughput related to the amount of bins')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True)

    fig.savefig("plot_4_1.pdf", format="pdf", bbox_inches="tight")
    # Display the plot
    plt.show()
