import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Noisy Gaussian function
def noisy_gaussian(x, snr=1000000):
    mu = np.array([0, 0, 0])
    sigma = np.array([1, 1, 1])
    f = np.exp(-np.sum(((x - mu) ** 2) / (2 * sigma ** 2)))
    noisy_f = f + np.random.normal(0, 1 / snr, f.shape)
    return noisy_f

# Expected Improvement acquisition function
def expected_improvement(x, gp, y_max):
    x = x.reshape(1, -1)
    mu, sigma = gp.predict(x)
    sigma = sigma.reshape(-1, 1)

    if sigma > 0:
        Z = (mu - y_max) / sigma
        ei = (mu - y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
    else:
        ei = 0.0

    return -ei

# Bayesian Optimization
def bayesian_optimization(n_iter, bounds, snr, n_init, method):
    # Initialize data
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, bounds.shape[0]))
    Y = np.array([noisy_gaussian(x, snr) for x in X])

    # Create and train Gaussian Process model
    kernel = GPy.kern.RBF(input_dim=bounds.shape[0], variance=1.0, lengthscale=1.0)
    gp = GPy.models.GPRegression(X, Y[:, None], kernel)
    gp.optimize()

    for _ in range(n_iter):
        y_max = np.max(Y)

        # Find the next point with the highest Expected Improvement
        result = minimize(
            fun=expected_improvement,
            x0=np.random.uniform(bounds[:, 0], bounds[:, 1], size=bounds.shape[0]),
            bounds=bounds,
            args=(gp, y_max),
            method=method
        )
        x_next = result.x

        # Evaluate the function at the next point
        y_next = noisy_gaussian(x_next, snr)

        # Update the Gaussian Process model with the new data point
        X = np.vstack((X, x_next))
        Y = np.append(Y, y_next)
        gp.set_XY(X, Y[:, None])
        gp.optimize()

    # Return the point with the maximum observed value
    max_idx = np.argmax(Y)
    return X[max_idx], Y[max_idx]

# Define the search space bounds
bounds = np.array([[-2, 2], [-2, 2], [-2, 2]])

# Run Bayesian Optimization
n_iter = 100
n_init = 200
method = 'Powell'
snr_range = np.arange(1, 10 , 2)
#snr_range=[20]
n_trials = 10
success_prob = []

for snr in snr_range:
    success_count = 0
    for trial in range(n_trials):
        x_max, y_max = bayesian_optimization(n_iter, bounds, snr, n_init, method)
        distance = np.sqrt(x_max[0]**2 + x_max[1]**2 + x_max[2]**2)
        if distance < 0.225:
            success_count += 1
    print(snr,success_count / n_trials)
    success_prob.append(success_count / n_trials)

plt.plot(snr_range, success_prob, marker='o')
plt.xlabel('SNR')
plt.ylabel('Probability of Success')
plt.title('Bayesian Optimization Performance vs SNR')
plt.grid(True)
plt.show()
