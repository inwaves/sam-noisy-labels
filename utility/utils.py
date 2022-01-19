import argparse
import random
import torch


def initialize(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def parse_args():
    """Parses command-line arguments corresponding to experiment parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--bootstrapped", default=False, type=bool,
                        help="Determines whether to run the bootstrapped version of the algorithm.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Select from cifar10 or cifar100.")
    parser.add_argument("--epochs", default=10, type=int, help="Total number of epochs.")
    parser.add_argument("--initial_rho", default=0.5, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--k", default=0.1, type=float, help="Parameter for exponential decay.")
    parser.add_argument("--rho_scheduler", default="constant", type=str,
                        help="Neighbourhood size scheduler type: exponential, stepdecay, stepincrease, constant.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--label_type", default="clean", type=str, help="Type of CIFAR labels to use: clean, aggregate,"
                                                                        " or worse.")
    parser.add_argument("--learning_rate", default=1.0, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--noise", default=0.2, type=float, help="Level of noise as float in [0, 1].")
    parser.add_argument("--optimiser-choice", default="SAM", type=str, help="Select from SAM or SGD.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="L2 weight decay.")
    args = parser.parse_args()

    return args