import pandas as pd
import matplotlib.pyplot as plt

# Load the training results
results_batch = pd.read_csv('results_batchnorm.csv')
results_instance_sgd = pd.read_csv('results_instancenorm_sgd.csv')
results_instance = pd.read_csv('results_instancenorm.csv')
results_nonorm = pd.read_csv('results_nonorm.csv')
results_groupNormg_1 = pd.read_csv('results_groupnormg=1.csv')
results_groupNormg_32 = pd.read_csv('results_groupnormg=32.csv')
results_identitysgd = pd.read_csv('results_identity.csv')


# Plotting the results
plt.figure(figsize=(12, 8))

# Train and Test Loss
plt.subplot(2, 1, 1)
#plt.plot(results_batch['Epoch'], results_batch['train_loss'], label='Train Loss Batch Normalization')
plt.plot(results_batch['Epoch'], results_batch['Test Loss'], label='Test Loss Batch Normalization')

#plt.plot(results_instance_sgd['Epoch'], results_instance_sgd['train_loss'], label='Train Loss Instance Normalizatin SGD')
plt.plot(results_instance_sgd['Epoch'], results_instance_sgd['Test Loss'], label='Test Loss Instance Normalizatin SGD')

#plt.plot(results_instance['Epoch'], results_instance['train_loss'], label='Train Loss Instance Normalization')
plt.plot(results_instance['Epoch'], results_instance['Test Loss'], label='Test Loss Instance Normalization')

#plt.plot(results_nonorm['Epoch'], results_nonorm['train_loss'], label='Train Loss No Normalization')
plt.plot(results_nonorm['Epoch'], results_nonorm['Test Loss'], label='Test Loss No Normalization')

#plt.plot(results_groupNormg_1['Epoch'], results_groupNormg_1['train_loss'], label='Train Loss Group Normalization G=1')
plt.plot(results_groupNormg_1['Epoch'], results_groupNormg_1['Test Loss'], label='Test Loss Group Normalization G=1')

#plt.plot(results_groupNormg_32['Epoch'], results_groupNormg_32['train_loss'], label='Train Loss Group Normalization G=32')
plt.plot(results_groupNormg_32['Epoch'], results_groupNormg_32['Test Loss'], label='Test Loss Group Normalization G=32')

#plt.plot(results_identitysgd['Epoch'], results_identitysgd['train_loss'], label='Train Loss Identity Normalization')
plt.plot(results_identitysgd['Epoch'], results_identitysgd['Test Loss'], label='Test Loss Identity Normalization')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()

# Test Accuracy
plt.subplot(2, 1, 2)
plt.plot(results_batch['Epoch'], results_batch['Test Accuracy'], label='Test Accuracy Batch Normalization' )
plt.plot(results_instance_sgd['Epoch'], results_instance_sgd['Test Accuracy'], label='Instance Normalization SGD')
plt.plot(results_instance['Epoch'], results_instance['Test Accuracy'], label='Test Accuracy Instance Normalization' )
plt.plot(results_nonorm['Epoch'], results_nonorm['Test Accuracy'], label='Test Accuracy No Normalization' )
plt.plot(results_groupNormg_1['Epoch'], results_groupNormg_1['Test Accuracy'], label='Test Accuracy GroupNormalization G=1' )
plt.plot(results_groupNormg_32['Epoch'], results_groupNormg_32['Test Accuracy'], label='Test Accuracy Group Normlization G=32' )
plt.plot(results_identitysgd['Epoch'], results_identitysgd['Test Accuracy'], label='Test Accuracy Identity Normalizaion SGD' )


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy with different Normalization Techniques')
plt.legend()

plt.tight_layout()
plt.savefig("exercise5_1.pdf")