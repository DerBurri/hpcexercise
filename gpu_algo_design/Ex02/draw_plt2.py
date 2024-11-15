#!/usr/bin/python3.11

import matplotlib.pyplot as plt
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
with open('fdtd3d-62222.out', 'r') as file:
    lines = file.readlines()

# Initialize lists to store radii and throughput data
radii = []
throughput = []

# Parse each line from the file
for line in lines:
    parts = line.split(", ")
    radius = int(parts[0].split("radius")[1])
    thrpt = float(parts[1].split(" = ")[1].split(" ")[0])
    radii.append(radius)
    throughput.append(thrpt)

# Initialize subplots for Throughput and FLOP/s
fig, axs = plt.subplots(2, 1, figsize=(12, 12), layout="tight")

# Throughput Plot
axs[0].plot(radii, throughput, marker='o', label='Throughput (MPoints/s)')
axs[0].set_title('Point throughput')
axs[0].set_xlabel('Radius')
axs[0].set_ylabel('Throughput (MPoints/s)')
axs[0].grid(True)

# FLOP/sec calculation
flops_s = [t * flop_per_point(r) * 1e6 for t, r in zip(throughput, radii)]

tflops_s = [f / 1e12 for f in flops_s]

# FLOP/sec Plot
axs[1].plot(radii, tflops_s, marker='s', color='r', label='FLOPS')
axs[1].set_title('Floating-point operations throughput')
axs[1].set_xlabel('Radius')
axs[1].set_ylabel('TFLOP/s')
axs[1].grid(True)

# Adjust layout and display the plots
plt.savefig('thrp_ex3.pdf', format='pdf', bbox_inches='tight')
