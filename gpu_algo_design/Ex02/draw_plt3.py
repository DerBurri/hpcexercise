#!/usr/bin/python3.11

import matplotlib.pyplot as plt
import re
plt.rc("ytick.major", size=5, width=1)
plt.rcParams["text.usetex"] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = 15
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 22

# Define a function to calculate FLOP/point based on radius
def flop_per_point(radius):
    return (6 * radius) + (radius + 1)

# Read data from the file
with open('results2_4.txt', 'r') as file:
    lines = file.readlines()

# Initialize lists to store radii and throughput data
radii = []
throughput = []
cachings = []

# Parse each line from the file
for line in lines:
    parts = line.split(", ")
    radius = int(parts[0].split("radius")[1])
    thrpt = float(parts[1].split(" = ")[1].split(" ")[0])
    match = re.search(r'caching (input|output)', line)
    if match:
        caching = match.group(1)
    else:
        caching = 'none'


    radii.append(radius)
    throughput.append(thrpt)
    cachings.append(caching)
    print(radius, thrpt, caching)

# Initialize subplots for Throughput and FLOP/s
fig, axs = plt.subplots(2, 1, figsize=(12, 12), layout="tight")
# Filter radii and throughput based on caching type
radii_input = [r for r, c in zip(radii, cachings) if c == 'input']
throughput_input = [t for t, c in zip(throughput, cachings) if c == 'input']

radii_output = [r for r, c in zip(radii, cachings) if c == 'output']
throughput_output = [t for t, c in zip(throughput, cachings) if c == 'output']

# Throughput Plot
axs[0].plot(radii_input, throughput_input, marker='o', label='Throughput (Input Caching) (MPoints/s)')
axs[0].plot(radii_output, throughput_output, marker='x', label='Throughput (Output Caching) (MPoints/s)')
axs[0].set_title('Point throughput')
axs[0].set_xlabel('Radius')
axs[0].set_ylabel('Throughput (MPoints/s)')
axs[0].grid(True)
axs[0].legend()

# FLOP/sec calculation
flops_s_input = [t * flop_per_point(r) * 1e6 for t, r, c in zip(throughput, radii, cachings) if c == 'input']

tflops_s_input = [f / 1e12 for f in flops_s_input]

flops_s_output = [t * flop_per_point(r) * 1e6 for t, r, c in zip(throughput, radii, cachings) if c == 'output']

tflops_s_output = [f / 1e12 for f in flops_s_output]

# FLOP/sec Plot
axs[1].plot(radii_input, tflops_s_input, marker='o', color='r', label='FLOPS (Input Caching)')
axs[1].plot(radii_output, tflops_s_output, marker='x', color='b', label='FLOPS (Output Caching)')
axs[1].set_title('Floating-point operations throughput')
axs[1].set_xlabel('Radius')
axs[1].set_ylabel('TFLOP/s')
axs[1].grid(True)
axs[1].legend()

# Adjust layout and display the plots
plt.savefig('thrp_ex4.pdf', format='pdf', bbox_inches='tight')
