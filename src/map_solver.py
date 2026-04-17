import numpy as np

class MAP:
    def __init__(self, log_likelihood, log_prior):
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior

    def log_posterior(self, theta, data):
        return self.log_likelihood(theta, data) + self.log_prior(theta)
    
    def posterior(self, theta, data):
        return np.exp(self.log_posterior(theta, data))

# Bernoulli for Coin Flipping

def log_bernoulli_likelihood(theta , data):
    """
    theta: The probability of result 1
    data: An array of (0, 1)
    """
    data = np.array(data) # Array of (0, 1)

    if theta <= 0 or theta >= 1:
        return -np.inf
    
    m = np.sum(data) # Number of 1
    n = len(data) - m # Number of 0

    return m * np.log(theta) + n * np.log(1 - theta)

def log_bernoulli_prior(theta, alpha=2, beta=2): # Beta Distribution
    """
    theta: The probability of result 1
    alpha: Number of result 1
    beta: Number of result 0
    """

    if theta <= 0 or theta >= 1:
        return -np.inf
    
    return (alpha - 1) * np.log(theta) + (beta - 1) * np.log(1 - theta)

# Gaussian for Sensor Fusion

def log_gaussian_likelihood(theta, data, sigma=1):
    """
    theta: Mean
    sigma: Standard Deviation (Equals 1 cause it does not effect the final answer)
    data: An array of values
    """

    data = np.array(data)

    return - sum((data - theta)**2) / (2 * sigma**2)

def log_gaussian_prior(theta, mu=0, sigma=1):
    """
    theta: Mean
    sigma: Standard Deviation
    mu: Mean (Equals 0 for easy caculation)
    """

    return - sum((theta - mu)**2) / (2 * sigma**2)

# Linear Regression