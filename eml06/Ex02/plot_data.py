import matplotlib.pyplot as plt
import numpy as np

def plot_learning_rates(log_file):
  """
  Plots accuracy over test epochs for each learning rate in a log file.

  Args:
    log_file (str): Path to the log file containing training information.
  """
  data = {}  # Dictionary to store learning rate and accuracy data
  with open(log_file, 'r') as f:
    for line in f:
      if "Learning Rate" in line:
        learning_rate = float(line.split(":")[-1].strip())
        if learning_rate not in data.keys():
          data[learning_rate] = []
      if "Average loss" in line:
        accuracy = float(line.split(",")[0].split(":")[2].strip())
        data[learning_rate].append(accuracy)

  # Check if data is valid
  if not data:
    raise ValueError("Log file doesn't contain learning rate or test accuracy information.")

  # Plot for each learning rate
  for learning_rate, accuracies in data.items():
    plt.plot(range(1,len(accuracies)+1), np.array(accuracies), label=f"Learning Rate: {learning_rate}")
  plt.xlabel("Epoch")
  plt.ylabel("Training Loss")
  plt.title("Training Loss over Epochs")
  plt.legend()
  plt.grid(True)
  plt.savefig("plot.pdf")

# Example usage
log_file = "metrics-default.log"
plot_learning_rates(log_file)
