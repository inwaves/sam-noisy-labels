import torch

from smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from net import Net
from cifar import Cifar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants.
batch_size = 4
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
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, rho=rho, adaptive=adaptive, lr=lr, momentum=momentum,
                weight_decay=weight_decay)
scheduler = StepLR(optimizer, lr, epochs) # Learning rate scheduler.
dataset = Cifar(batch_size, 0)


def train():
    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()
        log.train(len_dataset=len(dataset.train))

        # For CIFAR-10 we expect images of size 3x32x32.

        # Iterate over the batches in the training set.
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            # print(inputs.shape)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        # Set the model to eval mode.
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        # Begin testing the model on the test set.
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

        log.flush()


if __name__ == '__main__':
    train()


