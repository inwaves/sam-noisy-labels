import argparse
import sys
import torch

from smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.neighbourhood_scheduler import NeighbourhoodScheduler
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from net import Net
from cifar import CIFAR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append("..")


def setup(batch_size, threads, initial_rho, adaptive, momentum, weight_decay, lr, epochs, label_type):
    """Initialises dataset, model, optimiser, schedulers and log."""

    dataset = CIFAR(batch_size, label_type, threads)
    model = Net()
    base_optimizer = torch.optim.SGD
    optimiser = SAM(model.parameters(), base_optimizer, rho=initial_rho, adaptive=adaptive, lr=lr, momentum=momentum,
                    weight_decay=weight_decay)

    f = open("log.txt", "w")
    log = Log(log_each=10, file_writer=f)

    # Schedulers.
    scheduler = StepLR(optimiser, lr, epochs)  # Learning rate scheduler.
    nb_scheduler = NeighbourhoodScheduler(initial_rho, epochs, optimiser)

    return dataset, model, optimiser, log, scheduler, nb_scheduler


# TODO: Add option to train with plain SGD.
def train(dataset, model, optimiser, log, scheduler, nb_scheduler, epochs, label_smoothing):
    """Trains model using sharpness-aware minimisation (SAM)."""
    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()
        log.train(len_dataset=len(dataset.train))

        # Iterate over the batches in the training set.
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # First forward-backward step: finds the adversarial point.
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=label_smoothing)
            loss.mean().backward()
            optimiser.first_step(zero_grad=True)

            # Second forward-backward step: using gradient value at the adversarial point,
            # update the weights at the current point.
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=label_smoothing).mean().backward()
            optimiser.second_step(zero_grad=True)

            # Calculate the accuracy, log the results, and apply the learning
            # rate scheduler and the neighbourhood radius scheduler.
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)
                nb_scheduler(epoch)

        #####################
        # TRAINING COMPLETE #
        #####################
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

        # Write to log the final statistics.
        log.flush()


if __name__ == '__main__':
    # Start by parsing CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--initial_rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--label_type", default="clean", type=str, help="Type of CIFAR labels to use: clean, aggregate,"
                                                                        " or worst.")
    args = parser.parse_args()

    # Use CLI args to initialise arguments for the training process.
    train_args = setup(args.batch_size,
                       args.threads,
                       args.initial_rho,
                       args.adaptive,
                       args.momentum,
                       args.weight_decay,
                       args.learning_rate,
                       args.epochs,
                       args.label_type)

    # Train model.
    train(*train_args, args.epochs, args.label_smoothing)
