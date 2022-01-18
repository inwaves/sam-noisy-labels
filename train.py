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
          momentum, weight_decay, initial_rho, adaptive, rho_schedule_type, k, epochs):
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

    experiment_filename = f"./output/{source_dataset}_{label_type}_{str(noise).replace('.', 'p')}_" \
                          f"{optimiser_choice}_{adaptive}_{str(initial_rho).replace('.', 'p')}"
    f = open(f"{experiment_filename}.txt", "a+")
    log = Log(log_each=10, file_writer=f)

    # Schedulers.
    scheduler = StepLR(optimiser, learning_rate, epochs)  # Learning rate scheduler.

    if rho_schedule_type == "exponential":
        rho_schedule = ExponentialDecayNeighbourhoodSchedule(initial_rho, epochs, k, optimiser)
    elif rho_schedule_type == "stepdecay":
        rho_schedule = StepDecayNeighbourhoodSchedule(initial_rho, epochs, optimiser)
    elif rho_schedule_type == "stepincrease":
        rho_schedule = StepIncreaseNeighbourhoodSchedule(initial_rho, epochs, optimiser)
    else:
        rho_schedule = ConstantNeighbourhoodSchedule(initial_rho, optimiser)

    return dataset, model, optimiser, log, scheduler, rho_schedule


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

                    # Argmax is needed because the original targets are labels,
                    # whereas the output here is in 10-/100-class softmax form.
                    all_predictions.append(torch.argmax(model(inputs), 1))

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
        with torch.no_grad():
            if not bootstrapped:
                for batch in dataset.train:
                    inputs, _ = (b.to(device) for b in batch)

                    # Argmax is needed because the original targets are labels,
                    # whereas the output here is in 10-/100-class softmax form.
                    all_predictions.append(torch.argmax(model(inputs), 1))

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
    args = parse_args()

    training_params = setup(args.dataset, args.noise, args.batch_size, args.label_type, args.threads,
                            # Dataset arguments.
                            args.optimiser_choice, args.learning_rate, args.momentum, args.weight_decay,  # Optimiser.
                            args.initial_rho, args.adaptive, args.rho_scheduler, args.k,  # SAM-specific arguments.
                            args.epochs)  # Training arguments.

    # Run the training loop.
    if args.optimiser_choice == "SAM":
        train_sam(*training_params, args.epochs, args.label_smoothing, args.bootstrapped)
    else:
        train_sgd(*training_params, args.epochs, args.label_smoothing, args.bootstrapped)
