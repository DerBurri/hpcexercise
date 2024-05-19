import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results_l2.csv')
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(data['Epoch'], data['Test Loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(data['Epoch'], data['Test Accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("Ex4_2.pdf")
