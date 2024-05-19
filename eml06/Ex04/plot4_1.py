import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_file):
    # Load the data from CSV file
    data = pd.read_csv(csv_file)

    # Extract values
    train_loss = data['train_loss']
    train_acc = data['train_acc']
    test_loss = data['test_loss']
    test_acc = data['test_acc']

    epochs = range(1, len(train_loss) // len(train_acc) + 1)

    # Plot training and testing loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot([i * len(train_acc) for i in epochs], test_loss[:len(epochs)], label='Testing Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot([i * len(train_acc) for i in epochs], test_acc[:len(epochs)], label='Testing Accuracy')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_results('training_results.csv')
