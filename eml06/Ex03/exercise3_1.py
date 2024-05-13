from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import pandas as pd


class MLP(nn.Module):
    def __init__(self,input_shape):
        super(MLP,self).__init__()
        self.linear0 = nn.Linear(input_shape, 512)       #28*28, 512, batch_size, lr)
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
      x = torch.flatten(x, 1)
      x = self.linear0(x)
      x = torch.sigmoid(x)
      x = self.linear1(x)
      x = torch.sigmoid(x)
      x = self.linear2(x)
      x = F.log_softmax(x, dim=1)
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
    accuracy = 100. * correct /len(test_loader.dataset)
    print('\n Accuracy: {:.0f}%\n'.format(accuracy))
    return accuracy


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset',type=str, default='cifar10', metavar='N',
                        help='which dataset do you want to use? (CIFAR10 or MNIST)')
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
        
    mlp_accuracies = []
    elapsed_times = []

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('../data', train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

        model_mlp = MLP(32*32*3).to(device)
        optimizer_mlp = optim.SGD(model_mlp.parameters(), lr=args.lr)

        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            train(args, model_mlp, device, train_loader, optimizer_mlp, epoch)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            mlp_acc = test(model_mlp, device, test_loader)
            mlp_accuracies.append(mlp_acc)

        data = {"time": elapsed_times, "accuracy": mlp_accuracies}
        df = pd.DataFrame(data)
        if use_cuda:
            df.to_csv("gpu_data.csv")
        else:
            df.to_csv("cpu_data.csv")

    else:
        print("Using MNIST")
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)


        dataset_test = datasets.MNIST('../data', train=False,
                       transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size = args.batch_size)
        test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size = args.batch_size)

        model_mlp = MLP(input_shape=28*28).to(device)
        optimizer_mlp = optim.SGD(model_mlp.parameters(), lr=args.lr)

        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            train(args, model_mlp, device, train_loader, optimizer_mlp, epoch)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            mlp_acc = test(model_mlp, device, test_loader)
            mlp_accuracies.append(mlp_acc)

        data = {"time": elapsed_times, "accuracy": mlp_accuracies}
        df = pd.DataFrame(data)
        if use_cuda:
            df.to_csv("gpu_data.csv")
        else:
            df.to_csv("cpu_data.csv")



if __name__ == '__main__':
    main()
