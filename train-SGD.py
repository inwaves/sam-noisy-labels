import torch

from smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from net import Net
from cifar import CIFAR10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants.
batch_size = 128
epochs = 10

# Hyperparameters.
rho = 0.05
adaptive = False
lr = 0.1
momentum = 0.9
weight_decay = 0.0005
epochs = 200
label_smoothing = 0.1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = Log(log_each=10)
model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, lr, epochs) # Learning rate scheduler.
dataset = CIFAR10(batch_size, 0)


if __name__ == '__main__':
    train()


