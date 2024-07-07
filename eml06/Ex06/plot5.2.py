import pandas as pd
import matplotlib.pyplot as plt

results_pruned = []
results_structured_pruned = []


# Load the training results
for i in range(2,7):
    results_pruned.append(pd.read_csv(f'results_resnet18_pruned 0.{i}.csv'))
    results_structured_pruned.append( pd.read_csv(f'results_resnet18_structured_pruned0.{i}.csv'))

results_unpruned = pd.read_csv('results_resnet18.csv')

results_unpruned.describe()

# Plotting the results
fig, axs = plt.subplots(2, 1, figsize=(12, 16))

# Plot pruned results
axs[0].plot(results_unpruned['Epoch'], results_unpruned['Test Accuracy'], label='Unpruned ResNet18')
for i in range(5):
    axs[0].plot(results_pruned[i]['Epoch'], results_pruned[i]['Test Accuracy'], label=f'Pruned ResNet18 0.{i+2}')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Test Accuracy for Pruned ResNet18')
axs[0].legend()

# Plot structured pruned results
axs[1].plot(results_unpruned['Epoch'], results_unpruned['Test Accuracy'], label='Unpruned ResNet18')
for i in range(5):
    axs[1].plot(results_structured_pruned[i]['Epoch'], results_structured_pruned[i]['Test Accuracy'], label=f'Structured Pruned ResNet18 0.{i+2}')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Test Accuracy for Structured Pruned ResNet18')
axs[1].legend()

plt.tight_layout()
plt.savefig("plot5.1.pdf")


# Test Accuracy
#plt.plot(results_vggnet['epoch'], results_vggnet['test_accuracy'], label='Test Accuracy VGGNET')


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy between Different Pruning Rates')
plt.legend()

plt.tight_layout()
plt.savefig("plot5.1.pdf")