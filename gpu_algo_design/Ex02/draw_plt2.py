#!/usr/bin/python3.11

import matplotlib.pyplot as plt

# Define a function to calculate FLOP/point based on radius
def flop_per_point(radius):
    return (6 * radius) + (radius + 1)

# Read data from the file
with open('test_fdtd3d_62075.log', 'r') as file:
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
fig, axs = plt.subplots(2, 1, figsize=(8, 8), layout="tight")

# Throughput Plot
axs[0].plot(radii, throughput, marker='o', label='Throughput (MPoints/s)')
axs[0].set_title('Point throughput')
axs[0].set_xlabel('Radius')
axs[0].set_ylabel('Throughput (MPoints/s)')
axs[0].grid(True)

# FLOP/sec calculation
flops_s = [t * flop_per_point(r) * 1e6 for t, r in zip(throughput, radii)]

# FLOP/sec Plot
axs[1].plot(radii, flops_s, marker='s', color='r', label='FLOPS')
axs[1].set_title('Floating-point operations throughput')
axs[1].set_xlabel('Radius')
axs[1].set_ylabel('FLOP/s')
axs[1].grid(True)

# Adjust layout and display the plots
plt.savefig('thrp_ex3.pdf', format='pdf')
