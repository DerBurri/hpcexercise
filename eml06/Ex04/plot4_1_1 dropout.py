import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Load the training results
results = []
for i in range(1,9):
    filename = f'exercse4_1_10.{i}.csv.csv'
    results.append(pd.read_csv(filename))

# Plotting the results
plt.figure(figsize=(6, 8))

# Definieren Sie eine Liste von Farben
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
# Test Accuracy
ax = plt.subplot(1, 1, 1)
for i in range(0,8):
    label = results[i].drop_duplicates(subset=['dropout_probability'])
    plt.plot(results[i]['epoch'], results[i]['test_accuracy'], label=f'Dropout Probability ${label.iloc[0]["dropout_probability"]}', color=colors[i])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy with different Dropout Probabilities')
plt.legend()

# Erstellen eines eingebetteten Achsenfensters
axins = inset_axes(ax, width="50%", height="30%",  bbox_to_anchor=(0.6, 0.2, 0.4, 0.6), bbox_transform=ax.transAxes)

for i in range(0, 8):
    axins.plot(results[i]['epoch'], results[i]['test_accuracy'], color=colors[i])

# Zoom in on the last epochs
last_epochs = 5  # Number of last epochs to display
max_epoch = max([max(result['epoch']) for result in results])
axins.set_xlim(max_epoch - last_epochs, max_epoch)

# Set tighter y-limits for the inset
y_min = min([result[result['epoch'] >= max_epoch - last_epochs]['test_accuracy'].min() for result in results])
y_max = max([result[result['epoch'] >= max_epoch - last_epochs]['test_accuracy'].max() for result in results])
axins.set_ylim(y_min, y_max)

plt.savefig("exercise4_1_1dropout.pdf")
#plt.show()