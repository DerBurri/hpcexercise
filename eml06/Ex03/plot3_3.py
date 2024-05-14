import matplotlib.pyplot as plt
import pandas as pd

# Read accuracy data from CSV files for each optimizer
sgd_df = pd.read_csv("SGD_Accuracy.csv")
adam_df = pd.read_csv("Adam_Accuracy.csv")
adagrad_df = pd.read_csv("Adagrad_Accuracy.csv")

# Define learning rates for each optimizer
sgd_learning_rates = [0.001, 0.01, 0.05, 0.1]
adam_learning_rates = [0.001, 0.01, 0.05, 0.1]
adagrad_learning_rates = [0.001, 0.01, 0.05, 0.1]

# Function to plot accuracy over epochs for each optimizer and learning rate
def plot_accuracy_over_epochs(df, optimizer_name, learning_rates):
    plt.figure(figsize=(10, 6))
    for lr in learning_rates:
        lr_data = df[df['Learning Rate'] == lr]
        plt.plot(lr_data['Epoch'], lr_data['Accuracy'], label=f'LR={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Over Epochs for {optimizer_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{optimizer_name}_Accuracy.pdf")  # Save the plot as PDF
    plt.show()

# Plot accuracy over epochs for SGD optimizer
plot_accuracy_over_epochs(sgd_df, 'SGD', sgd_learning_rates)

# Plot accuracy over epochs for Adam optimizer
plot_accuracy_over_epochs(adam_df, 'Adam', adam_learning_rates)

# Plot accuracy over epochs for Adagrad optimizer
plot_accuracy_over_epochs(adagrad_df, 'Adagrad', adagrad_learning_rates)
