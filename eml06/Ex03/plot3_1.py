import matplotlib.pyplot as plt
import pandas as pd

df_gpu = pd.read_csv("gpu_data.csv")
df_cpu = pd.read_csv("cpu_data.csv")

time_gpu = df_gpu["time"]
time_cpu = df_cpu["time"]

accuracy_gpu = df_gpu["accuracy"]
accuracy_cpu = df_cpu["accuracy"]

plt.plot(time_cpu, accuracy_cpu, label='CPU')
plt.plot(time_gpu, accuracy_gpu, label='GPU')
plt.xlabel('Training Time[s]')
plt.ylabel('Test Accuracy[%]')
plt.title('Test Accuracy Over Training Time')
plt.grid()
plt.legend()
plt.savefig("Training accuracy over Time.pdf")