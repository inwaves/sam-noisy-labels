import argparse
import sys
import torch

from smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.neighbourhood_scheduler import NeighbourhoodScheduler
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from cifar10net import Cifar10Net
from cifar100net import Cifar100Net
from cifar import CIFAR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append("..")


def setup(source_dataset, batch_size, label_type, threads, optimiser_choice, learning_rate,
          momentum, weight_decay, initial_rho, adaptive, epochs):
    """ Sets up the training process. """
    dataset = CIFAR(source_dataset, batch_size, label_type, threads)

    model = Cifar10Net() if source_dataset == "cifar10" else Cifar100Net()

    if optimiser_choice == "SAM":
        base_optimiser = torch.optim.SGD
        optimiser = SAM(model.parameters(),
                        base_optimiser,
                        rho=initial_rho,
                        adaptive=adaptive,
                        lr=learning_rate,
                        momentum=momentum,
                        weight_decay=weight_decay)
    else:
        optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    f = open("log.txt", "w")
    log = Log(log_each=10, file_writer=f)

    # Schedulers.
    scheduler = StepLR(optimiser, learning_rate, epochs)  # Learning rate scheduler.
    nb_scheduler = NeighbourhoodScheduler(initial_rho, epochs, optimiser)

    return dataset, model, optimiser, log, scheduler, nb_scheduler


def train_sgd(dataset, model, optimiser, log, scheduler, nb_scheduler, epochs, label_smoothing):
    """Trains model using stochastic gradient descent."""

    for epoch in range(epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        # Iterate over training set.
        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, label_smoothing)

            loss.mean().backward()
            optimiser.step()
            optimiser.zero_grad()

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


def train_sam(dataset, model, optimiser, log, scheduler, nb_scheduler, epochs, label_smoothing):
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
    parser.add_argument("--dataset", default="cifar10", type=str, help="Select from cifar10 or cifar100.")
    parser.add_argument("--epochs", default=10, type=int, help="Total number of epochs.")
    parser.add_argument("--initial_rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--label_type", default="clean", type=str, help="Type of CIFAR labels to use: clean, aggregate,"
                                                                        " or worst.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--optimiser-choice", default="SAM", type=str, help="Select from SAM or SGD.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    args = parser.parse_args()

    training_params = setup(args.dataset, args.batch_size, args.label_type, args.threads,  # Dataset arguments.
                            args.optimiser_choice, args.learning_rate, args.momentum, args.weight_decay,  # Optimiser.
                            args.initial_rho, args.adaptive,  # SAM-specific arguments.
                            args.epochs)  # Training arguments.

    # Run the training loop.
    if args.optimiser_choice == "SAM":
        train_sam(*training_params, args.epochs, args.label_smoothing)
    else:
        train_sgd(*training_params, args.epochs, args.label_smoothing)
