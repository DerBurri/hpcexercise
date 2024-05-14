from __future__ import print_function
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Define your CNN model
class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(18432, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        print("WILL USE CUDA!!")
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Define learning rates for different optimizers
    learning_rates = [0.001, 0.01, 0.05, 0.1]

    # Define the optimizers
    optimizers = [ optim.SGD, optim.Adam, optim.Adagrad]

    # Define a list to store accuracy dataframes for each optimizer
    accuracy_dfs = []

    for optimizer_class in optimizers:
        # Define a list to store accuracy data for different learning rates
        accuracy_data = []

        for lr in learning_rates:
            # Load CIFAR-10 dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
            dataset_test = datasets.CIFAR10('../data', train=False, transform=transform)
            train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
            test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

            # Instantiate model and optimizer
            model = CNN(3).to(device)
            optimizer = optimizer_class(model.parameters(), lr=lr)

            # Training and testing
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                test_loss, accuracy = test(model, device, test_loader)
                accuracy_data.append((lr, epoch, accuracy))

        # Create a DataFrame for accuracy data and append it to the list
        df = pd.DataFrame(accuracy_data, columns=['Learning Rate', 'Epoch', 'Accuracy'])
        accuracy_dfs.append(df)

    # Save accuracy dataframes for each optimizer to CSV files
    for i, optimizer_class in enumerate(optimizers):
        df = accuracy_dfs[i]
        optimizer_name = optimizer_class.__name__
        df.to_csv(f"{optimizer_name}_Accuracy.csv", index=False)

if __name__ == '__main__':
    main()
