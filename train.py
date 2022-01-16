import argparse
import sys
import torch

from loss.smooth_cross_entropy import smooth_crossentropy
from utility.log import Log
from utility.neighbourhood_scheduler import ExponentialDecayNeighbourhoodSchedule, StepDecayNeighbourhoodSchedule, \
    StepIncreaseNeighbourhoodSchedule, ConstantNeighbourhoodSchedule
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from optimiser.sam import SAM
from data.cifar import CIFAR
import models.resnet as resnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append("..")


def setup(source_dataset, batch_size, label_type, threads, optimiser, learning_rate,
          momentum, weight_decay, initial_rho, adaptive, rho_scheduler, k, epochs):
    """ Sets up the training process. """
    dataset = CIFAR(source_dataset, batch_size, label_type, threads)

    model = resnet.resnet32().to(device)

    if optimiser == "SAM":
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

    if rho_scheduler == "exponential":
        nb_scheduler = ExponentialDecayNeighbourhoodSchedule(initial_rho, epochs, k, optimiser)
    elif rho_scheduler == "stepdecay":
        nb_scheduler = StepDecayNeighbourhoodSchedule(initial_rho, epochs, optimiser)
    elif rho_scheduler == "stepincrease":
        nb_scheduler = StepIncreaseNeighbourhoodSchedule(initial_rho, epochs, optimiser)
    else:
        nb_scheduler = ConstantNeighbourhoodSchedule(initial_rho, optimiser)

    return dataset, model, optimiser, log, scheduler, nb_scheduler


def train_sgd(dataset, model, optimiser, log, scheduler, nb_scheduler, epochs, label_smoothing, bootstrapped):
    """Trains model using stochastic gradient descent."""

    bootstrapped_targets, all_predictions = [], []
    if bootstrapped:
        print(f"Bootstrapping an SGD run.\n")
        bootstrapped_targets = train_sgd(dataset=dataset,
                                         model=model,
                                         optimiser=optimiser,
                                         log=log,
                                         scheduler=scheduler,
                                         nb_scheduler=nb_scheduler,
                                         epochs=epochs,
                                         label_smoothing=label_smoothing,
                                         bootstrapped=False)
        print(f"Length: {len(bootstrapped_targets)}: first element: {bootstrapped_targets[0]}")

    for epoch in range(epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        # Iterate over training set.
        for ind, batch in enumerate(dataset.train):
            inputs, targets = (b.to(device) for b in batch)
            targets = bootstrapped_targets[ind].to(device) if bootstrapped else targets

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
        # Save predictions for bootstrapping.
        with torch.no_grad():
            if not bootstrapped:
                for batch in dataset.train:
                    inputs, _ = (b.to(device) for b in batch)
                    all_predictions.extend(model(inputs))

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

        # Write final statistics.
        log.flush()

        if bootstrapped:
            return None
        return all_predictions


def train_sam(dataset, model, optimiser, log, scheduler, nb_scheduler, epochs, label_smoothing, bootstrapped):
    """Trains model using sharpness-aware minimisation (SAM)."""

    bootstrapped_targets, all_predictions = [], []
    if bootstrapped:
        log.writeline(f"Bootstrapping a SAM run.")
        bootstrapped_targets = train_sam(dataset=dataset,
                                         model=model,
                                         optimiser=optimiser,
                                         log=log,
                                         scheduler=scheduler,
                                         nb_scheduler=nb_scheduler,
                                         epochs=epochs,
                                         label_smoothing=label_smoothing,
                                         bootstrapped=False)

    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()
        log.train(len_dataset=len(dataset.train))

        # Iterate over the batches in the training set.
        for ind, batch in enumerate(dataset.train):
            # Inputs is shape (batch_size, 3, 32, 32). Targets is shape (batch_size,).
            inputs, targets = (b.to(device) for b in batch)
            targets = bootstrapped_targets[ind].to(device) if bootstrapped else targets

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
        # Save predictions for bootstrapping.
        if not bootstrapped:
            for batch in dataset.train:
                inputs, _ = (b.to(device) for b in batch)
                all_predictions.extend(model(inputs))

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

        if bootstrapped:
            return None
        return all_predictions


def train_swa():
    pass


if __name__ == '__main__':
    # Start by parsing CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--bootstrapped", default=False, type=bool,
                        help="Run bootstrapped training: train once on the original labels, then again on the "
                             "predicted labels from the first run.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Select from cifar10 or cifar100.")
    parser.add_argument("--epochs", default=10, type=int, help="Total number of epochs.")
    parser.add_argument("--initial_rho", default=0.5, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--k", default=0.1, type=float,
                        help="Parameter for the exponential decay schedule for ρ.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--label_type", default="clean", type=str, help="Type of CIFAR labels to use: clean, aggregate,"
                                                                        " or worse.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--noise_level", default=0.2, type=float,
                        help="When label type is 'blue', each label is flipped with probability equal to this "
                             "parameter.")
    parser.add_argument("--optimiser", default="SAM", type=str, help="Select from SAM or SGD.")
    parser.add_argument("--rho_scheduler", default=0.5, type=float,
                        help="Neighbourhood size scheduler type: exponential, stepdecay, stepincrease")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    args = parser.parse_args()

    training_params = setup(args.dataset, args.batch_size, args.label_type, args.threads,  # Dataset arguments.
                            args.optimiser, args.learning_rate, args.momentum, args.weight_decay,  # Optimiser.
                            args.initial_rho, args.adaptive, args.rho_scheduler, args.k,  # SAM-specific arguments.
                            args.epochs)  # Training arguments.

    # Run the training loop.
    if args.optimiser == "SAM":
        train_sam(*training_params, args.epochs, args.label_smoothing, args.bootstrapped)
    else:
        train_sgd(*training_params, args.epochs, args.label_smoothing, args.bootstrapped)
