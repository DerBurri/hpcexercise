import pandas as pd
import matplotlib.pyplot as plt

# Load the training results
results_convnet = pd.read_csv('resultsex03convnet.csv')
results_vggnet = pd.read_csv('resultsex04_vggnet.csv')
results_resnet = pd.read_csv('results_resnet18.csv')


# Plotting the results
plt.figure(figsize=(12, 8))


# Training Time
plt.plot(results_vggnet['epoch'], results_vggnet['train_time'].cumsum(), label='Training Time VGGNET')
plt.plot(results_convnet['Epoch'], results_convnet['Total Time'], label='Training Time ConvNet' )
plt.plot(results_resnet['Epoch'], results_resnet['total_time'], label='Training Time ResNet' )



plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Time difference between Nets')
plt.legend()

plt.tight_layout()
plt.savefig("exercise5_2_time.pdf")