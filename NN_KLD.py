import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as pltimport
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Function to set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set the seed
set_seed(42)

class NetHook(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs = module_out

    def clear(self):
        self.outputs = []


class Compute_RFF_NDR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs, ys, rgl, l):
        n_xs, d = xs.shape
        n_ys, _ = ys.shape
        self.omega = torch.normal(0, 1, (d, l)).to(xs.device) / l
        self.theta = (torch.rand(l) * 2 * math.pi).to(xs.device)

        self.rff_xs = torch.cos((xs @ self.omega) + self.theta.repeat((n_xs, 1)))
        self.rff_ys = torch.cos((ys @ self.omega) + self.theta.repeat((n_ys, 1)))
        self.w = (torch.linalg.inv(self.rff_ys.T @ self.rff_ys / n_ys + (rgl * torch.eye(l).to(xs.device))).detach()
             @ (torch.mean(self.rff_xs, dim=0))) # - torch.mean(self.rff_ys, dim=0)))
        return (torch.mean(self.rff_xs, dim=0))@ self.w

    def update_rff(self, xs):
        n_xs, _ = xs.shape
        rff_xs = torch.cos((xs @ self.omega) + self.theta.repeat((n_xs, 1)))
        inv_matrix = torch.linalg.inv(self.rff_xs.T @ self.rff_xs / n_xs + (self.rgl * torch.eye(self.l).to(self.device)))

        rff_loss_constant =inv_matrix.clone().detach()
        return rff_xs,rff_loss_constant

    def compute_loss(self):
        return torch.mean(self.rff_xs, dim=0) @ (self.rff_loss_constant @ torch.mean(self.rff_xs, dim=0))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True, drop_last=True)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False, drop_last=True)
    hooker = NetHook()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    model.fc1.register_forward_hook(hooker)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()


    train_losses = []
    compute_RFF_NDR = Compute_RFF_NDR()

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            index=(target==0)| (target==1)
            data=data[index]
            target=target[index]
            optimizer.zero_grad()
            out = model(data)
            ref_output = hooker.outputs

            ce_loss = criterion(out, target)

            # loss function
            rff_loss = -compute_RFF_NDR(ref_output[target == 0], ref_output[target == 1], rgl=0.1, l=128)

            loss = rff_loss + 0.5 * ce_loss  # loss function
            #loss = rff_loss# loss function
            loss.backward()  # backward
            optimizer.step()  # upadate

            epoch_loss += loss.item()
            # print(f"batch nums:{i}")
            # print(f"ce_loss{ce_loss}", f"rff_loss:{rff_loss}")
            # print(f"rff_loss:{rff_loss}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f'Train Epoch: {epoch} \tLoss: {avg_epoch_loss:.6f}')

    torch.save(model.state_dict(), 'mnist_cnn_rff_ndr.pth')

    # Test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            index=(target==0)| (target==1)
            data=data[index]
            target=target[index]
            output = model(data)
            test_output = hooker.outputs

            rff_loss = -compute_RFF_NDR(test_output[target == 0], test_output[target == 1], rgl=0.1, l=128)
            loss = rff_loss

            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
