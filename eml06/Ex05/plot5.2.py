import pandas as pd
import matplotlib.pyplot as plt

# Load the training results
results_convnet = pd.read_csv('resultsex03convnet.csv')
results_vggnet = pd.read_csv('resultsex04_vggnet.csv')
results_resnet = pd.read_csv('results_resnet18.csv')


# Plotting the results
plt.figure(figsize=(12, 8))


# Test Accuracy
plt.plot(results_vggnet['epoch'], results_vggnet['test_accuracy'], label='Test Accuracy VGGNET')
plt.plot(results_convnet['Epoch'], results_convnet['Accuracy'], label='Test Accuracy ConvNet' )
plt.plot(results_resnet['Epoch'], results_resnet['test_accuracy'], label='Test Accuracy ResNet' )



plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy between Nets')
plt.legend()

plt.tight_layout()
plt.savefig("exercise5_2.pdf")