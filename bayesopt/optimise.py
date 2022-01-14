import GPy
import GPyOpt
import numpy as np
from numpy.random import seed

seed(12345)

# TODO: Implement bayesian optimisation for ρ.

def fit_gp(hyperparameters):
    """This is the function that GPyOpt will optimise.
        parameter hyperparameters: A vector containing the hyperparameters to optimise.
        return: The root mean squared error of the model.
    """

    # TODO: add support for multiple loss functions.
    rmse = 0
    print(f"Hyperparams shape: {hyperparameters.shape}")
    for i in range(hyperparameters.shape[0]):
        kernel = GPy.kern.RBF(1, lengthscale=hyperparameters[i, 0], variance=hyperparameters[i, 3]) + \
                 GPy.kern.StdPeriodic(1, lengthscale=hyperparameters[i, 1]) * \
                 GPy.kern.PeriodicMatern32(1, lengthscale=hyperparameters[i, 2], variance=hyperparameters[i, 4])
        model = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, normalizer=True,
                                        noise_var=0.05)
        y_mean, y_std = model.predict(x_all.reshape(-1, 1))
        y_pred = draw_samples(y_mean, y_std)

        # Calculate RMSE — does it even make sense to use RMSE since
        # a GP doesn't make exact predictions, but specifies distributions
        # over functions?
        rmse += np.sqrt(np.square(y_all - y_pred).mean())

        # This used to be rmse += np.sum(all_y - y_pred)
        # which isn't the actual RMSE, but gave the positive definite error fewer times.

    print(f"RMSE: {rmse}")
    return rmse

def optimise(maximum_iterations=10, dom_tuples=None, acquisition_type="LCB", acquisition_weight=0.1):
    if dom_tuples is None:
        dom_tuples = [(0., 5.), (0., 1.), (0., 5.), (0., 1.), (0., 1.)]

    domain = [
        {'name': 'lengthscale1', 'type': 'continuous', 'domain': dom_tuples[0]},
        {'name': 'lengthscale2', 'type': 'continuous', 'domain': dom_tuples[1]},
        {'name': 'lengthscale3', 'type': 'continuous', 'domain': dom_tuples[2]},
        {'name': 'variance1', 'type': 'continuous', 'domain': dom_tuples[3]},
        {'name': 'variance2', 'type': 'continuous', 'domain': dom_tuples[4]}]

    # TODO: add support for multiple models
    # TODO: add support for multiple optimizers

    opt = GPyOpt.methods.BayesianOptimization(f=fit_gp,  # function to optimize
                                              domain=domain,  # box-constraints of the problem
                                              acquisition_type=acquisition_type,
                                              acquisition_weight=acquisition_weight)

    # Optimise the hyperparameters.
    opt.run_optimization(max_iter=maximum_iterations)
    opt.plot_convergence()  # TODO: get these out to a file.

    # Get the optimised hyperparameters.
    x_best = opt.X[np.argmin(opt.Y)]

    kernel = GPy.kern.RBF(1, lengthscale=optimal_hparams[0], variance=optimal_hparams[3]) + \
             GPy.kern.StdPeriodic(1, lengthscale=optimal_hparams[1]) * \
             GPy.kern.PeriodicMatern32(1, lengthscale=optimal_hparams[2], variance=optimal_hparams[4])

    model = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, normalizer=True,
                                    noise_var=0.05)

    y_mean, y_std = model.predict(x_all.reshape(-1, 1))
    y_pred = draw_samples(y_mean, y_std)

    # Calculate RMSE — does it even make sense to use RMSE since
    # a GP doesn't make exact predictions, but specifies distributions
    # over functions?
    best_rmse = np.sqrt(np.square(y_all - y_pred).mean())

    return x_best, best_rmse, y_mean, y_std