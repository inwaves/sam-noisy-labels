import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CIFAR:
    def __init__(self, batch_size, threads):
        data_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# TODO: Add support for CIFAR-100

