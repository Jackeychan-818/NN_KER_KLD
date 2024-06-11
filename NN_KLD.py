import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Compute_RFF_NDR:
    def __init__(self, xs, rgl, l):
        self.l = l
        self.rgl = rgl
        self.device = xs.device
        n_xs, d = xs.shape
        self.omega = torch.normal(0, 1, (d, l)).to(self.device) / l
        self.theta = (torch.rand(l) * 2 * math.pi).to(self.device)
        self.update_rff(xs)

    def update_rff(self, xs):
        n_xs, _ = xs.shape
        self.rff_xs = torch.cos((xs @ self.omega) + self.theta.repeat((n_xs, 1)))
        self.inv_matrix = torch.linalg.inv(self.rff_xs.T @ self.rff_xs / n_xs + (self.rgl * torch.eye(self.l).to(self.device)))

        self.rff_loss_constant = self.inv_matrix.clone().detach()
    def compute_loss(self):
        return torch.mean(self.rff_xs, dim=0) @ (self.rff_loss_constant @ torch.mean(self.rff_xs, dim=0))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self._initialize_fc_layers()  # Initialize fully connected layers with correct size

    def _initialize_fc_layers(self):
        sample_input = torch.zeros(1, 3, 32, 32)
        sample_output = self._forward_conv_layers(sample_input)
        flattened_size = sample_output.numel()
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 10)

    def _forward_conv_layers(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

    # initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_losses = []

    # training process
    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)  # forward

            # cross entropy
            ce_loss = criterion(output, target)

            # Update RFF for the current batch
            rff_ndr = Compute_RFF_NDR(output, rgl=0.1, l=128)
            rff_ndr.update_rff(output)  # Update the inverse matrix
            rff_loss = rff_ndr.compute_loss()

            # Combined loss
            loss = ce_loss + 0.1 * rff_loss
            loss.backward()  # backward
            optimizer.step()  # update the parameters

            epoch_loss += loss.item()

        # average loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f'Train Epoch: {epoch} \tLoss: {avg_epoch_loss:.6f}')

    # save parameters
    torch.save(model.state_dict(), 'cifar10_cnn_rff_ndr.pth')

    # testing procedure
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            ce_loss = criterion(output, target)

            rff_ndr = Compute_RFF_NDR(output, rgl=0.1, l=128)
            
            rff_ndr.update_rff(output)  #
            
            rff_loss = rff_ndr.compute_loss()
            
            loss = ce_loss + 0.1 * rff_loss
            test_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    # plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
