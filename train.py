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
from utility.utils import parse_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sys.path.append("..")


def setup(source_dataset, noise, batch_size, label_type, threads, optimiser_choice, learning_rate,
          momentum, weight_decay, initial_rho, adaptive, rho_schedule_type, epochs):
    """ Sets up the training process. """
    dataset = CIFAR(source_dataset, batch_size, label_type, noise, threads)

    model = resnet.resnet32().to(device) if source_dataset == "cifar10" \
        else resnet.ResNet(resnet.BasicBlock, [5, 5, 5], 100).to(device)

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

    experiment_filename = f"{source_dataset}_{label_type}_{noise}_{optimiser_choice}_{adaptive}_{initial_rho}"
    f = open(f"{experiment_filename}.txt", "a+")
    log = Log(log_each=10, file_writer=f)

    # Schedulers.
    scheduler = StepLR(optimiser, learning_rate, epochs)  # Learning rate scheduler.

    if rho_schedule_type == "exponential":
        rho_schedule = ExponentialDecayNeighbourhoodSchedule(initial_rho, epochs, optimiser)
    elif rho_schedule_type == "stepdecay":
        rho_schedule = StepDecayNeighbourhoodSchedule(initial_rho, epochs, optimiser)
    elif rho_schedule_type == "stepincrease":
        rho_schedule = StepIncreaseNeighbourhoodSchedule(initial_rho, epochs, optimiser)
    else:
        rho_schedule = ConstantNeighbourhoodSchedule(initial_rho, optimiser)

    return dataset, model, optimiser, log, scheduler, rho_schedule


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

        # Write final statistics.
        log.flush()


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
            print(f"Loss: {loss}, loss_mean: {loss.mean()}")
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
    args = parse_args()

    training_params = setup(args.dataset, args.noise, args.batch_size, args.label_type, args.threads,
                            # Dataset arguments.
                            args.optimiser_choice, args.learning_rate, args.momentum, args.weight_decay,  # Optimiser.
                            args.initial_rho, args.adaptive, args.rho_scheduler,  # SAM-specific arguments.
                            args.epochs)  # Training arguments.

    # Run the training loop.
    if args.optimiser_choice == "SAM":
        train_sam(*training_params, args.epochs, args.label_smoothing)
    else:
        train_sgd(*training_params, args.epochs, args.label_smoothing)
