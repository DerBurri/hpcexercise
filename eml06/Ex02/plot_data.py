import matplotlib.pyplot as plt

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
        data[learning_rate] = []
      elif "Accuracy" in line:
        accuracy = float(line.split("%")[-2].split("(")[-1].strip())
        data[learning_rate].append(accuracy)

  # Check if data is valid
  if not data:
    raise ValueError("Log file doesn't contain learning rate or test accuracy information.")

  # Plot for each learning rate
  for learning_rate, accuracies in data.items():
    plt.plot(range(len(accuracies)), accuracies, label=f"Learning Rate: {learning_rate}")
  plt.xlabel("Epoch")
  plt.ylabel("Test Accuracy")
  plt.title("Test Accuracy over Epochs for Different Learning Rates")
  plt.legend()
  plt.grid(True)
  plt.show()

# Example usage
log_file = "eml06/Ex02/metrics-adaptive-learningrate.log"
plot_learning_rates(log_file)
