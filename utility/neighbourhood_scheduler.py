import math


class NeighbourhoodScheduler:
    def __init__(self, rho: float, total_epochs: int, optimiser):
        self.total_epochs = total_epochs
        self.optimiser = optimiser
        self.initial_rho = rho
        # FIXME: this needs to have a different name.

    def __call__(self, rho: float):
        for param_group in self.optimiser.param_groups:
            param_group["rho"] = rho

    def rho(self) -> float:
        return self.optimiser.param_groups[0]["rho"]


class ExponentialDecayNeighbourhoodSchedule(NeighbourhoodScheduler):
    def __init__(self, rho: float, total_epochs: int, k: float, optimiser):
        super().__init__(rho, total_epochs, optimiser)
        self.k = k

    def __call__(self, epoch: int):
        rho = self.initial_rho * math.exp(-self.k * epoch)
        super().__call__(rho)

    def rho(self):
        return self.optimiser.param_groups[0]["rho"]


class StepDecayNeighbourhoodSchedule(NeighbourhoodScheduler):
    def __init__(self, rho: float, total_epochs: int, optimiser):
        super().__init__(rho, total_epochs, optimiser)

    def __call__(self, epoch: int):
        """Right now this schedule seems arbitrary, I should
                inform it from the paper values."""
        if epoch < self.total_epochs // 3:
            rho = self.initial_rho
        elif epoch < 2 * self.total_epochs // 3:
            rho = self.initial_rho * 0.1
        else:
            rho = self.initial_rho * 0.01

        super().__call__(rho)

    def rho(self):
        return self.optimiser.param_groups[0]["rho"]


class StepIncreaseNeighbourhoodSchedule(NeighbourhoodScheduler):
    def __init__(self, rho: float, total_epochs: int, optimiser):
        super().__init__(rho, total_epochs, optimiser)

    def __call__(self, epoch: int):
        if epoch < self.total_epochs // 3:
            rho = self.initial_rho
        elif epoch < 2 * self.total_epochs // 3:
            rho = self.initial_rho * 10
        else:
            rho = self.initial_rho * 100

        super().__call__(rho)

    def rho(self):
        return self.optimiser.param_groups[0]["rho"]


class ConstantNeighbourhoodSchedule(NeighbourhoodScheduler):
    def __init__(self, rho: float, optimiser):
        self.optimiser = optimiser
        self.initial_rho = rho

    def __call__(self, epoch: int):
        super().__call__(self.initial_rho)

    def rho(self):
        return self.optimiser.param_groups[0]["rho"]
