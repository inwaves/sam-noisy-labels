class NeighbourhoodScheduler:
    def __init__(self, rho: float, total_epochs: int, optimiser):
        self.total_epochs = total_epochs
        self.optimiser = optimiser
        self.rho = rho

    def __call__(self, epoch: int):
        """Right now this schedule seems arbitrary, I should
        inform it from the paper values."""
        if epoch < self.total_epochs // 3:
            rho = self.rho
        elif epoch < 2 * self.total_epochs // 3:
            rho = self.rho * 0.1    # What should this be?
        else:
            rho = self.rho * 0.01

        for param_group in self.optimiser.param_groups:
            param_group["rho"] = rho

    def rho(self) -> float:
        return self.optimiser.param_groups[0]["rho"]
