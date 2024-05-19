import pandas as pd
import matplotlib.pyplot as plt

# Load the training results
results = pd.read_csv('exercise4_1_1.csv')

# Plotting the results
plt.figure(figsize=(12, 8))

# Train and Test Loss
plt.subplot(2, 1, 1)
plt.plot(results['epoch'], results['train_loss'], label='Train Loss')
plt.plot(results['epoch'], results['test_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()

# Test Accuracy
plt.subplot(2, 1, 2)
plt.plot(results['epoch'], results['test_accuracy'], label='Test Accuracy', color='g')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("exercise4_1_1.pdf")