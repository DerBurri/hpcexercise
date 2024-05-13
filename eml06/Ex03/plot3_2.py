import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("CNN_MLP_Data.csv")

time_cnn = data["time_cnn"]
time_mlp = data["time_mlp"]

accuracy_cnn = data["accuracy_cnn"]
accuracy_mlp = data["accuracy_mlp"]

plt.plot(time_cnn, accuracy_cnn, label='CNN')
plt.plot(time_mlp, accuracy_mlp, label='MLP')
plt.xlabel('Training Time[s]')
plt.ylabel('Test Accuracy[%]')
plt.title('Test Accuracy Over Training Time')
plt.grid()
plt.legend()
plt.savefig("CNN MLP comparison.pdf")