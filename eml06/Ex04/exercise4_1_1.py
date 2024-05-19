from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import pandas as pd

# Define the VGG11 model
class VGG11(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(512 * 1 * 1, 4096), nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 10),  # 10 classes for CIFAR-10
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch, results):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader.dataset)
    results['train_loss'].append(avg_train_loss)

def test(model, device, test_loader, epoch, results):
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
    results['test_loss'].append(test_loss)
    results['test_accuracy'].append(accuracy)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout_p', type=float, default=0.0,
                        help='dropout_p (default: 0.0)')
    parser.add_argument('--L2_reg', type=float, default=None,
                        help='L2_reg (default: None)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_transforms = transforms.Compose([transforms.ToTensor()])
    train_transforms = [transforms.ToTensor()]
    train_transforms = transforms.Compose(train_transforms)

    dataset_train = datasets.SVHN('../data', split='train', download=True,
                       transform=train_transforms)
    dataset_test = datasets.SVHN('../data', split='test', download=True,
                       transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = VGG11(dropout_p=args.dropout_p).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = {'epoch': [], 'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    print(f'Starting training at: {time.time():.4f}')
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, results)
        test(model, device, test_loader, epoch, results)
        results['epoch'].append(epoch)
        print(f"Epoch {epoch}: Train Loss {results['train_loss'][-1]}, Test Loss {results['test_loss'][-1]}, Test Accuracy {results['test_accuracy'][-1]}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('exercise4_1_1.csv', index=False)
    print('Training results saved to exercise4_1_1.csv')

    if (args.L2_reg is not None):
        f_name = f'trained_VGG11_L2-{args.L2_reg}.pt'
        torch.save(model.state_dict(), f_name)
        print(f'Saved model to: {f_name}')


if __name__ == '__main__':
    main()
